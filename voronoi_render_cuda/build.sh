#!/usr/bin/env bash
# Build CUDA extension. Usage:
#   ./build.sh                  # common SM (75-90)
#   ./build.sh native           # current GPU only
#   ./build.sh "86;89"          # custom list
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"
ARCH="${1:-}"

mkdir -p "$BUILD"
cd "$BUILD"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "$ARCH" ]]; then
  if [[ "$ARCH" == "native" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES=native)
  else
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
  fi
fi

cmake "${CMAKE_ARGS[@]}" -DPython_EXECUTABLE=/mnt/data/foam3r/.venv/bin/python ..
cmake --build . -j"$(nproc)"

# Copy module next to package for easy import
cp -f voronoi_render_cuda*.so "$ROOT/" 2>/dev/null || cp -f voronoi_render_cuda*.so "$ROOT/../voronoi_render_cuda/" || true
echo "Built: $BUILD/voronoi_render_cuda*.so"
