#pragma once

#include "voronoi/types.h"

#include <cuda_runtime.h>

namespace voronoi {

struct SceneGpu {
    int num_sites = 0;
    int max_degree = 0;
    int sh_degree = 3;
    int sh_dim = kShAttrSize;
    float weight_threshold = 0.001f;
    int max_steps = 128;

    float* positions = nullptr;
    float* sh_attrs = nullptr;
    float* density = nullptr;

    // CSR Delaunay adjacency (RadFoam-style)
    int adjacency_size = 0;
    uint32_t* adjacency = nullptr;
    uint32_t* adjacency_offsets = nullptr;
    float* adjacent_diff = nullptr;  // 3 floats per global edge index

    // Spatial grid NN
    float grid_min[3] = {0, 0, 0};
    float grid_inv_cell[3] = {1, 1, 1};
    int grid_dims[3] = {1, 1, 1};
    int grid_num_cells = 0;
    int* grid_offsets = nullptr;
    int* grid_indices = nullptr;
};

SceneGpu load_scene_gpu(const char* path);
void free_scene_gpu(SceneGpu& scene);

void prefetch_adjacent_diff_cuda(const SceneGpu& scene, cudaStream_t stream = nullptr);

void trace_fused_cuda(
    const SceneGpu& scene,
    const float* d_rays,
    int num_rays,
    float* d_out_rgba,
    cudaStream_t stream = nullptr
);

void trace_cuda(
    const SceneGpu& scene,
    const float* d_rays,
    const int* d_start_cells,
    int num_rays,
    float* d_out_rgba,
    cudaStream_t stream = nullptr
);

void nearest_site_indices_cuda(
    const SceneGpu& scene,
    const float* d_origins,
    int num_origins,
    int* d_out_indices,
    cudaStream_t stream = nullptr
);

}  // namespace voronoi
