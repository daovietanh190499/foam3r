#pragma once

#include "voronoi/scene.h"

namespace voronoi {

void prefetch_adjacent_diff_cuda(const SceneGpu& scene, cudaStream_t stream);

}  // namespace voronoi
