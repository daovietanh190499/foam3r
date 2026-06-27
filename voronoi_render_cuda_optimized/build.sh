#!/usr/bin/env bash
# Build optimized voronoi_cuda extension.
# Usage: ./build.sh [native|SM list]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"
ARCH="${1:-}"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "/mnt/data/foam3r/.venv/bin/python" ]]; then
    PYTHON="/mnt/data/foam3r/.venv/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi
fi

mkdir -p "$BUILD"
cd "$BUILD"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PYTHON")
if [[ -n "$ARCH" ]]; then
  if [[ "$ARCH" == "native" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES=native)
  else
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
  fi
fi

cmake "${CMAKE_ARGS[@]}" ..
cmake --build . -j"$(nproc)"
echo "Built: $BUILD/voronoi_cuda*.so"
