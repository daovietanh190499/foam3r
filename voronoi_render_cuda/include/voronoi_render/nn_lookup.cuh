#pragma once

#include "voronoi_render/scene.h"

namespace voronoi {

// Tiled 1-NN on GPU; reuses scene.positions already resident (~0 extra scene VRAM).
// origins: device, num_origins * 3 (xyz)
// out_indices: device, num_origins int32
void nearest_site_indices_cuda(
    const SceneGpu& scene,
    const float* d_origins,
    int num_origins,
    int* d_out_indices
);

// Same, reading origins from packed rays (num_rays * 6, xyz at [0:3]).
void nearest_site_indices_from_rays_cuda(
    const SceneGpu& scene,
    const float* d_rays,
    int num_rays,
    int* d_out_indices
);

}  // namespace voronoi
