#pragma once

#include "voronoi_render/types.h"

namespace voronoi {

struct SceneGpu {
    int num_sites = 0;
    int max_degree = 0;
    int sh_degree = 3;
    int sh_dim = kShAttrSize;
    float weight_threshold = 0.001f;
    int max_steps = 128;

    float* positions = nullptr;     // num_sites * 3
    float* sh_attrs = nullptr;      // num_sites * sh_dim
    float* density = nullptr;       // num_sites
    int* nbr_idx = nullptr;         // num_sites * max_degree
    float* nbr_diff = nullptr;      // num_sites * max_degree * 3
    unsigned char* nbr_valid = nullptr;  // num_sites * max_degree

    // Uniform spatial grid for GPU NN (~8-10 MB extra; no duplicate positions).
    float grid_min[3] = {0, 0, 0};
    float grid_inv_cell[3] = {1, 1, 1};
    int grid_dims[3] = {1, 1, 1};
    int grid_num_cells = 0;
    int* grid_offsets = nullptr;    // grid_num_cells + 1
    int* grid_indices = nullptr;    // num_sites
};

SceneGpu load_scene_gpu(const char* path);
void free_scene_gpu(SceneGpu& scene);

void trace_rays_cuda(
    const SceneGpu& scene,
    const float* rays,       // num_rays * 6  [ox,oy,oz, dx,dy,dz]
    const int* start_cells,  // num_rays
    int num_rays,
    float* out_rgba          // num_rays * 4
);

}  // namespace voronoi
