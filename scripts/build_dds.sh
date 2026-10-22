#!/usr/bin/env bash
# Build the DDS 2.9.0 double-dummy solver as a shared library into third_party/dds/.
# Linux (g++) and macOS (clang). Uses C++11 std::thread only, so no Boost or OpenMP is needed.
# Usage: bash scripts/build_dds.sh
set -euo pipefail

DDS_TAG=v2.9.0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/third_party/dds"
SRC="$OUT/src-$DDS_TAG"

mkdir -p "$OUT"
[ -d "$SRC" ] || git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$DDS_TAG" https://github.com/dds-bridge/dds.git "$SRC"

case "$(uname -s)" in
  Linux)  MAKEFILE=Makefile_linux_shared;    TARGET=linux; LIB=libdds.so ;;
  Darwin) MAKEFILE=Makefile_Mac_clang_shared; TARGET=macos; LIB=libdds.dylib ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

cd "$SRC/src"
# DDS 2.9.0 reads free memory from the third line of `free -k`, which is the Swap line on
# current procps, so it sees 0 MB and starts no solver threads. Read MemAvailable instead.
sed -i.orig -e "s#\"free -k | tail -n+3 | head -n1 | awk '{print \$NF}'\"#\"awk '/MemAvailable/ {print \$2}' /proc/meminfo\"#" System.cpp
grep -q MemAvailable System.cpp || { echo "failed to patch System.cpp" >&2; exit 1; }
# The library destructor calls FreeMemory() after the C++ globals it walks may already be
# destroyed, which segfaults every process at exit. The OS frees that memory anyway.
sed -i.orig -e '/libEnd(void)/,/^}/s/^  FreeMemory();/  \/\/ FreeMemory();  (bridge-ai: crashes at exit)/' dds.cpp
grep -q "bridge-ai: crashes at exit" dds.cpp || { echo "failed to patch dds.cpp" >&2; exit 1; }
sed -e 's/^THREADING\t= .*/THREADING\t= $(THR_STL)/' \
    -e 's/^THREAD_COMPILE\t= .*/THREAD_COMPILE\t=/' \
    -e 's/^THREAD_LINK\t= .*/THREAD_LINK\t= -pthread/' \
    -e 's/ -fopenmp//' \
    "Makefiles/$MAKEFILE" > Makefile
make clean >/dev/null 2>&1 || true
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
make -j"$JOBS" "$TARGET" >"$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }
cp libdds.so "$OUT/$LIB"
cp "$SRC/LICENSE" "$OUT/LICENSE"
echo "built $OUT/$LIB"
