#!/usr/bin/env bash
# Prepare a fresh AutoDL instance for scripts/autodl/m2_train.sh.
#
#   bash setup_instance.sh
#
# Installs what earlier runs needed (2026-10-23): a Python 3.10 conda env
# "brl", brl's requirements with setuptools<70, JAX 0.4.23 with CUDA 12 from
# the USTC PyPI mirror (Google's JAX index is unreachable from AutoDL), the
# brl checkout and the Pgx DDS tables from hf-mirror. Safe to run again.
# pip uses the USTC mirror: on 2026-10-23 Aliyun dropped to tens of kB/s for
# the large CUDA wheels while USTC gave about 1.3 MB/s. Override with PIP_INDEX.
#
# Not done here: the OpenSpiel WBridge5 data (train.txt, test.txt) comes from
# storage.googleapis.com, which AutoDL cannot reach; copy it in from a machine
# that can, into /root/autodl-tmp/m2/openspiel/.
set -euo pipefail

WORK=/root/autodl-tmp
ENV=brl
PIP_INDEX=${PIP_INDEX:-https://mirrors.ustc.edu.cn/pypi/simple}
HERE=$(cd "$(dirname "$0")" && pwd)

# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^$ENV "; then
  conda create -y -n "$ENV" python=3.10
fi
conda activate "$ENV"

cd "$WORK"
# AutoDL's GitHub proxy, only for the clone: left on, it slows the pip mirrors to a crawl
[ -d brl ] || (source /etc/network_turbo 2>/dev/null || true; git clone https://github.com/harukaki/brl.git)
cd brl

pip install -i "$PIP_INDEX" "setuptools<70"
pip install -i "$PIP_INDEX" -r requirements.txt
pip install -i "$PIP_INDEX" "jax==0.4.23" "jaxlib==0.4.23" "jax-cuda12-plugin==0.4.23" "jax-cuda12-pjrt==0.4.23" \
  nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 \
  "nvidia-cudnn-cu12>=8.9,<9" nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12
python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'JAX sees no GPU'"

if [ ! -e dds_results/train_000.npy ] || [ "$(ls dds_results/train_*.npy | wc -l)" -lt 2 ]; then
  bash "$HERE/../fetch_dds_data.sh" "$WORK/brl" "$ENV"
fi
ls dds_results | head
echo "setup done"
