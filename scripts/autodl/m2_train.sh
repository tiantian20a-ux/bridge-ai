#!/usr/bin/env bash
# M2: the whole GPU part of bidding training in one go, on an AutoDL instance.
#
#   1. preflight  - every input and tool checked in a minute, before any GPU time is spent
#   2. sl         - supervised learning on the OpenSpiel WBridge5 data (brl sl.py)
#   3. rl         - PPO with fictitious self-play from our SL model (brl ppo.py,
#                   the settings of brl's model-pretrained-rl-with-fsp)
#   4. eval       - brl eval.py: ours against brl's released models
#
# Each finished stage leaves a marker, so running the script again resumes
# at the first unfinished stage. Run it detached so it survives the SSH session:
#
#   nohup bash m2_train.sh > /root/autodl-tmp/m2/run.log 2>&1 &
#
# Progress: tail -f /root/autodl-tmp/m2/progress.log
#
# Environment:
#   BRL_DIR        brl checkout (default /root/autodl-tmp/brl)
#   OUT            run directory (default /root/autodl-tmp/m2)
#   OPENSPIEL_DIR  directory holding train.txt and test.txt (default $OUT/openspiel)
#   CONDA_ENV      conda environment with brl's requirements (default brl)
#   SHUTDOWN_WHEN_DONE=1  power the instance off after the last stage (or on failure)
#   SL_ITERATIONS, RL_TIMESTEPS, NUM_ENVS  override for a short trial run
#   ALLOW_CPU=1    skip the GPU check (trial runs only); CONDA_ENV= uses the current python
set -uo pipefail

BRL_DIR=${BRL_DIR:-/root/autodl-tmp/brl}
OUT=${OUT:-/root/autodl-tmp/m2}
OPENSPIEL_DIR=${OPENSPIEL_DIR:-$OUT/openspiel}
CONDA_ENV=${CONDA_ENV-brl}
SL_ITERATIONS=${SL_ITERATIONS:-400000}
RL_TIMESTEPS=${RL_TIMESTEPS:-2621440000}
NUM_ENVS=${NUM_ENVS:-8192}
OPENSPIEL_SHA_TRAIN=59b0fb726b8887271a51f97af6e4bab425a54c0b712bf83e544df5ede95d4bbb
OPENSPIEL_SHA_TEST=41ee45364d71ec2608c474eda52d3c8d4302c6ca46cb23bcbaf66a8ac6386352

mkdir -p "$OUT"
PROGRESS=$OUT/progress.log
export WANDB_MODE=disabled

log() { echo "$(date '+%F %T') $*" | tee -a "$PROGRESS"; }

finish() {
  local status=$1
  log "exit status $status"
  if [ "${SHUTDOWN_WHEN_DONE:-0}" = 1 ]; then
    log "powering off"
    sync
    shutdown -h now
  fi
  exit "$status"
}

fail() { log "FAILED: $*"; finish 1; }

stage_done() { [ -e "$OUT/.done-$1" ]; }
mark_done() { touch "$OUT/.done-$1"; log "stage $1 done"; }

if [ -n "$CONDA_ENV" ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" || fail "conda env $CONDA_ENV not found"
fi
export ALLOW_CPU=${ALLOW_CPU:-0}
cd "$BRL_DIR" || fail "no brl checkout at $BRL_DIR"

# ---------------------------------------------------------------- preflight
log "preflight"
python - <<'EOF' || fail "python environment (see run.log)"
import jax, pgx, pyspiel, haiku, optax, distrax, wandb, omegaconf  # noqa: F401
devs = jax.devices()
import os
assert devs[0].platform == "gpu" or os.environ["ALLOW_CPU"] == "1", f"JAX sees no GPU: {devs}"
print("jax", jax.__version__, "pgx", pgx.__version__, devs)
EOF
ls dds_results/train_*.npy > /dev/null 2>&1 || fail "no dds_results/train_*.npy under $BRL_DIR"
[ -e dds_results/test_000.npy ] || fail "no dds_results/test_000.npy (needed by eval)"
[ "$(ls dds_results/train_*.npy | wc -l)" -ge 2 ] || log "warning: only one DDS train file; is this the random sample table?"
[ -e bridge_models/model-sl.pkl ] || fail "brl's released models are missing from $BRL_DIR/bridge_models"
if ! stage_done sl; then
  for f in train test; do
    [ -s "$OPENSPIEL_DIR/$f.txt" ] || fail "missing $OPENSPIEL_DIR/$f.txt (OpenSpiel bridge data, see docs)"
  done
  echo "$OPENSPIEL_SHA_TRAIN  $OPENSPIEL_DIR/train.txt" | sha256sum -c --quiet || fail "train.txt checksum"
  echo "$OPENSPIEL_SHA_TEST  $OPENSPIEL_DIR/test.txt" | sha256sum -c --quiet || fail "test.txt checksum"
fi
avail=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc 0-9)
[ "$avail" -ge 10 ] || fail "only ${avail}G free under $OUT; RL checkpoints need about 2G, SL about 1G"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$PROGRESS"
log "preflight ok"

# ---------------------------------------------------------------- sl
SL_MODEL=$OUT/sl/params-$SL_ITERATIONS.pkl
if ! stage_done sl; then
  log "sl: $SL_ITERATIONS iterations"
  python sl.py iterations="$SL_ITERATIONS" train_batch=128 learning_rate=0.0001 \
    eval_every=$(( SL_ITERATIONS < 10000 ? SL_ITERATIONS : 10000 )) data_path="$OPENSPIEL_DIR" save_path="$OUT/sl" \
    || fail "sl.py"
  [ -s "$SL_MODEL" ] || fail "sl.py finished without $SL_MODEL"
  mark_done sl
fi

# ---------------------------------------------------------------- rl
# ppo.py cannot resume. If the instance stopped mid-run, a rerun starts a new
# run m2-fsp-from-<N> from the latest checkpoint (N updates done) for the
# remaining updates, with every earlier checkpoint copied into its model zoo
# so fictitious self-play keeps its past opponents. Adam's state restarts.
RL_DIR=$OUT/rl
STEPS_PER_UPDATE=$((NUM_ENVS * 32))
TOTAL_UPDATES=$((RL_TIMESTEPS / STEPS_PER_UPDATE))

latest_checkpoint() {  # prints "<updates done> <path>" for the furthest checkpoint, or nothing
  local best=0 path="" d off f n
  for d in "$RL_DIR"/m2-fsp "$RL_DIR"/m2-fsp-from-*; do
    [ -d "$d/rl_params" ] || continue
    off=0
    case $d in *-from-*) off=$((10#${d##*-from-})) ;; esac
    for f in "$d"/rl_params/params-*.pkl; do
      [ -e "$f" ] || continue
      n=$(basename "$f" .pkl)
      n=$((off + 10#${n#params-}))
      if [ "$n" -gt "$best" ]; then best=$n; path=$f; fi
    done
  done
  [ -n "$path" ] && echo "$best $path"
}

if ! stage_done rl; then
  read -r DONE_UPDATES INIT_MODEL <<< "$(latest_checkpoint)"
  DONE_UPDATES=${DONE_UPDATES:-0}
  if [ "$DONE_UPDATES" -eq 0 ]; then
    EXP=m2-fsp
    INIT_MODEL=$SL_MODEL
    # a run that died before its first checkpoint: start it again
    [ -d "$RL_DIR/$EXP" ] && mv "$RL_DIR/$EXP" "$RL_DIR/$EXP.partial-$(date +%s)"
  else
    EXP=m2-fsp-from-$(printf %08d "$DONE_UPDATES")
    [ -d "$RL_DIR/$EXP" ] && mv "$RL_DIR/$EXP" "$RL_DIR/$EXP.partial-$(date +%s)"
  fi
  REMAINING=$((TOTAL_UPDATES - DONE_UPDATES))
  log "rl: PPO + FSP, $REMAINING of $TOTAL_UPDATES updates left, from $INIT_MODEL (run $EXP)"
  # ppo.py creates log_path/exp_name only when tracking with wandb, then mkdirs
  # rl_params inside it non-recursively, so make the parent but not rl_params.
  mkdir -p "$RL_DIR/$EXP"
  python ppo.py num_envs="$NUM_ENVS" num_steps=32 minibatch_size=1024 \
    total_timesteps=$((REMAINING * STEPS_PER_UPDATE)) update_epochs=10 lr=0.000001 gamma=1 gae_lambda=0.95 ent_coef=0.001 \
    VE_COEF=0.5 num_eval_envs=100 eval_opp_model_path=bridge_models/model-sl.pkl num_eval_step=10 \
    load_initial_model=True initial_model_path="$INIT_MODEL" \
    log_path="$RL_DIR" exp_name="$EXP" ratio_model_zoo=1 \
    save_model=True save_model_interval=100 track=False &
  PPO_PID=$!
  if [ "$DONE_UPDATES" -gt 0 ]; then
    # Seed the zoo once ppo.py has made rl_params (well before its first update,
    # which waits for JIT compilation). "old-" names never clash with its own.
    for _ in $(seq 600); do [ -d "$RL_DIR/$EXP/rl_params" ] && break; kill -0 $PPO_PID 2>/dev/null || break; sleep 1; done
    for d in "$RL_DIR"/m2-fsp "$RL_DIR"/m2-fsp-from-*; do
      [ "$d" = "$RL_DIR/$EXP" ] && continue
      for f in "$d"/rl_params/params-*.pkl; do
        [ -e "$f" ] && cp "$f" "$RL_DIR/$EXP/rl_params/old-$(basename "$d")-$(basename "$f")"
      done
    done
    log "zoo seeded with $(ls "$RL_DIR/$EXP/rl_params" | grep -c '^old-') earlier checkpoints"
  fi
  wait $PPO_PID || fail "ppo.py"
  mark_done rl
fi
read -r _ RL_MODEL <<< "$(latest_checkpoint)"
[ -n "${RL_MODEL:-}" ] || fail "no RL checkpoint under $RL_DIR"
log "rl model: $RL_MODEL"

# ---------------------------------------------------------------- eval
if ! stage_done eval; then
  mkdir -p "$OUT/final"
  cp "$SL_MODEL" "$OUT/final/m2-sl.pkl"
  cp "$RL_MODEL" "$OUT/final/m2-fsp.pkl"
  for pair in "m2-sl.pkl model-sl.pkl" "m2-fsp.pkl model-sl.pkl" "m2-fsp.pkl model-pretrained-rl-with-fsp.pkl"; do
    set -- $pair
    log "eval $1 vs $2"
    # 10000 envs needs about 42 GB; 1000 boards gives about ±0.2 IMPs
    python eval.py team1_model_path="$OUT/final/$1" team2_model_path="bridge_models/$2" num_eval_envs=1000 \
      > "$OUT/eval-$1-$2.log" 2>&1 || log "eval failed, see $OUT/eval-$1-$2.log"
    grep -E "^IMP" "$OUT/eval-$1-$2.log" | tee -a "$PROGRESS"
  done
  mark_done eval
fi

log "all stages done; models in $OUT/final"
finish 0
