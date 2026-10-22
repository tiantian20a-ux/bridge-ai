#!/usr/bin/env bash
# 在 AutoDL 实例上下载 Pgx 的真实双明手（DDS）数据，替换测速时用的随机表。
# 实例连不上 Google Drive（pgx 1.4.0 的 download_dds_results 用的地址），
# 这里改从 Hugging Face 上的同名文件下载（Pgx 新版本已改用这个地址），先走国内镜像 hf-mirror.com。
# 数据放在文件存储 /root/autodl-fs（实例释放后还在），brl 目录下的 dds_results 链接过去。
# 用法：bash fetch_dds_data.sh [brl 目录，默认 /root/autodl-tmp/brl] [conda 环境，默认 brl]
set -euo pipefail

BRL="${1:-/root/autodl-tmp/brl}"
ENV="${2:-brl}"
STORE=/root/autodl-fs/dds_results
REPO=datasets/sotetsuk/dds_dataset
FILES=(dds_results_2.5M.npy dds_results_500K.npy)

mkdir -p "$STORE/raw"
cd "$STORE/raw"

fetch() {
  local name=$1 host
  for host in https://hf-mirror.com https://huggingface.co; do
    [ "$host" = https://huggingface.co ] && { source /etc/network_turbo 2>/dev/null || true; }
    echo "从 $host 下载 $name ……"
    if curl -fL --retry 3 -C - -o "$name" "$host/$REPO/resolve/main/$name"; then
      curl -fsL "$host/$REPO/raw/main/README.md" -o README.md 2>/dev/null || true
      return 0
    fi
  done
  echo "下载 $name 失败" >&2
  return 1
}

for f in "${FILES[@]}"; do
  [ -s "$f" ] || fetch "$f"
done
ls -lh "$STORE/raw"

# 用 pgx 自己的切分逻辑生成 train_000.npy … 和 test_000.npy …（每份 10 万副），
# 只是把它的下载函数换成从本地复制。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV"
cd "$STORE"
python - <<'PY'
import os, shutil
import pgx.bridge_bidding as bb

local = {bb.DDS_RESULTS_TRAIN_URL: "raw/dds_results_2.5M.npy",
         bb.DDS_RESULTS_TEST_URL: "raw/dds_results_500K.npy"}
bb._download = lambda url, fname: shutil.copy(local[url], fname)
for f in ("dds_results_2.5M.npy", "dds_results_500K.npy"):
    if os.path.exists(os.path.join("split", f)):
        os.remove(os.path.join("split", f))
bb.download_dds_results("split")
for f in ("dds_results_2.5M.npy", "dds_results_500K.npy"):
    os.remove(os.path.join("split", f))  # 原始文件在 raw/ 里已有一份
n = sorted(os.listdir("split"))
print(f"{len(n)} 个分片：{n[0]} … {n[-1]}")

# 冒烟测试：pgx 1.4.0 能加载分片并走完一局叫牌（四家都 Pass）
import jax
env = bb.BridgeBidding("split/train_000.npy")
state = env.init(jax.random.PRNGKey(0))
for _ in range(4):
    state = env.step(state, 0)
assert bool(state.terminated), "四家 Pass 后应结束"
print("pgx 加载真实数据正常")
PY

# brl 默认从 ./dds_results 读；测速时的随机表先挪开留底
cd "$BRL"
if [ -e dds_results ] && [ ! -L dds_results ]; then
  mv dds_results "dds_results_random_$(date +%m%d%H%M)"
fi
ln -sfn "$STORE/split" dds_results
echo "完成：$BRL/dds_results -> $STORE/split"
[ -s "$STORE/raw/README.md" ] && { echo "---- 数据集说明（含许可） ----"; head -40 "$STORE/raw/README.md"; }
