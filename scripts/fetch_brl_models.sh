#!/usr/bin/env bash
# Download brl's released bidding models (Apache-2.0, github.com/harukaki/brl)
# into data/brl_models/. Each file is about 15 MB.
#
#   bash scripts/fetch_brl_models.sh            # all five
#   bash scripts/fetch_brl_models.sh model-sl   # just some
#
# Set BRL_MIRROR to a GitHub proxy prefix where raw.githubusercontent.com is
# unreachable (e.g. on AutoDL).
set -euo pipefail

REV=${BRL_REV:-main}
BASE=${BRL_MIRROR:-}https://raw.githubusercontent.com/harukaki/brl/$REV/bridge_models
OUT=$(cd "$(dirname "$0")/.." && pwd)/data/brl_models
mkdir -p "$OUT"

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
  models=(model-sl model-pretrained-rl model-pretrained-rl-with-fsp model-pretrained-rl-with-pfsp model-from-scratch-rl)
fi

for m in "${models[@]}"; do
  if [ -s "$OUT/$m.pkl" ]; then
    echo "have $m.pkl"
    continue
  fi
  echo "fetching $m.pkl"
  curl -fL --retry 3 -o "$OUT/$m.pkl.part" "$BASE/$m.pkl"
  mv "$OUT/$m.pkl.part" "$OUT/$m.pkl"
done
