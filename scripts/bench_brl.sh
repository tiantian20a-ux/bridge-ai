#!/usr/bin/env bash
# brl 叫牌 PPO 训练测速，适用于 AutoDL vGPU-32GB + Miniconda / Python 3.10 镜像。
# 跑 100 次和 300 次更新各一遍，用两次耗时之差算出每次更新的耗时（扣掉 JIT 编译等固定开销），
# 再推算一次完整训练（1 万次更新）要多少小时、多少钱。预计总耗时 10 到 40 分钟。
# 用法：bash bench_brl.sh [每小时价格，默认 1.58]
set -euo pipefail

PRICE="${1:-1.58}"
WORK=/root/autodl-tmp
STEPS_PER_UPDATE=$((8192 * 32))

# AutoDL 的学术资源加速，用来访问 GitHub 和 Google 的下载地址；没有就跳过
source /etc/network_turbo 2>/dev/null || true

cd "$WORK"
[ -d brl ] || git clone https://github.com/harukaki/brl.git
cd brl

pip install -r requirements.txt
# requirements.txt 里的 jaxlib 是 CPU 版，换成同版本的 CUDA 12 版
pip install "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', '没有用上 GPU'"
python -c "from pgx.bridge_bidding import download_dds_results; download_dds_results()"

export WANDB_MODE=disabled

run() {
  local updates=$1
  local start end
  start=$(date +%s)
  python ppo.py num_envs=8192 num_steps=32 minibatch_size=1024 \
    total_timesteps=$((STEPS_PER_UPDATE * updates)) update_epochs=10 lr=0.00001 gamma=1 \
    gae_lambda=0.95 ent_coef=0.001 VE_COEF=0.5 num_eval_envs=100 \
    eval_opp_model_path="bridge_models/model-sl.pkl" num_eval_step=10 \
    load_initial_model=False log_path="bench_log" exp_name="bench${updates}" save_model=False \
    > "bench_${updates}.log" 2>&1
  end=$(date +%s)
  echo $((end - start))
}

echo "跑 100 次更新……"
T100=$(run 100)
echo "  用时 ${T100} 秒"
echo "跑 300 次更新……"
T300=$(run 300)
echo "  用时 ${T300} 秒"

python - "$T100" "$T300" "$PRICE" "$STEPS_PER_UPDATE" <<'PY'
import sys
t100, t300, price, spu = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
per_update = (t300 - t100) / 200
hours = per_update * 10000 / 3600
print(f"每次更新 {per_update:.2f} 秒，约 {spu / per_update:,.0f} 步/秒")
print(f"一次完整训练（1 万次更新，26.2 亿步）约 {hours:.1f} 小时，按 {price} 元/时约 {hours * price:.0f} 元")
PY
echo "把上面三行结果发回来即可。出错的话把 bench_100.log 的最后 30 行发回来。"
