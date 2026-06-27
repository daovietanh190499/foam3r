#include "voronoi_render/nn_lookup.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace voronoi {

namespace {

constexpr int kBlockThreads = 256;

__device__ int grid_cell_id(const SceneGpu& scene, float ox, float oy, float oz) {
    int ix = static_cast<int>((ox - scene.grid_min[0]) * scene.grid_inv_cell[0]);
    int iy = static_cast<int>((oy - scene.grid_min[1]) * scene.grid_inv_cell[1]);
    int iz = static_cast<int>((oz - scene.grid_min[2]) * scene.grid_inv_cell[2]);
    ix = max(0, min(ix, scene.grid_dims[0] - 1));
    iy = max(0, min(iy, scene.grid_dims[1] - 1));
    iz = max(0, min(iz, scene.grid_dims[2] - 1));
    return ix + scene.grid_dims[0] * (iy + iz * scene.grid_dims[1]);
}

__device__ void consider_site(
    const float* positions,
    int site,
    float ox,
    float oy,
    float oz,
    int& best_j,
    float& best_d2
) {
    const float* p = positions + site * 3;
    const float dx = p[0] - ox;
    const float dy = p[1] - oy;
    const float dz = p[2] - oz;
    const float d2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    if (d2 < best_d2) {
        best_d2 = d2;
        best_j = site;
    }
}

__device__ void nearest_grid_query(
    const SceneGpu& scene,
    float ox,
    float oy,
    float oz,
    int& best_j,
    float& best_d2
) {
    const int nx = scene.grid_dims[0];
    const int ny = scene.grid_dims[1];
    const int nz = scene.grid_dims[2];
    const int stride_y = nx;
    const int stride_z = nx * ny;

    int cx = static_cast<int>((ox - scene.grid_min[0]) * scene.grid_inv_cell[0]);
    int cy = static_cast<int>((oy - scene.grid_min[1]) * scene.grid_inv_cell[1]);
    int cz = static_cast<int>((oz - scene.grid_min[2]) * scene.grid_inv_cell[2]);
    cx = max(0, min(cx, nx - 1));
    cy = max(0, min(cy, ny - 1));
    cz = max(0, min(cz, nz - 1));

      constexpr int kRadius = 1;
    for (int dz = -kRadius; dz <= kRadius; ++dz) {
        for (int dy = -kRadius; dy <= kRadius; ++dy) {
            for (int dx = -kRadius; dx <= kRadius; ++dx) {
                const int ix = cx + dx;
                const int iy = cy + dy;
                const int iz = cz + dz;
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) continue;
                const int cell = ix + stride_y * (iy + iz * ny);
                const int start = scene.grid_offsets[cell];
                const int end = scene.grid_offsets[cell + 1];
                for (int k = start; k < end; ++k) {
                    consider_site(scene.positions, scene.grid_indices[k], ox, oy, oz, best_j, best_d2);
                }
            }
        }
    }
}

__global__ void nearest_site_grid_kernel(
    SceneGpu scene,
    const float* __restrict__ origins,
    int num_origins,
    int* __restrict__ out_indices
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_origins) return;

    const float ox = origins[i * 3 + 0];
    const float oy = origins[i * 3 + 1];
    const float oz = origins[i * 3 + 2];

    int best_j = 0;
    float best_d2 = FLT_MAX;
    nearest_grid_query(scene, ox, oy, oz, best_j, best_d2);
    out_indices[i] = best_j;
}

__global__ void nearest_from_rays_grid_kernel(
    SceneGpu scene,
    const float* __restrict__ rays,
    int num_rays,
    int* __restrict__ out_indices
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    const int base = i * 6;
    int best_j = 0;
    float best_d2 = FLT_MAX;
    nearest_grid_query(scene, rays[base], rays[base + 1], rays[base + 2], best_j, best_d2);
    out_indices[i] = best_j;
}

}  // namespace

void nearest_site_indices_cuda(
    const SceneGpu& scene,
    const float* d_origins,
    int num_origins,
    int* d_out_indices
) {
    if (num_origins <= 0) return;
    const int blocks = (num_origins + kBlockThreads - 1) / kBlockThreads;
    nearest_site_grid_kernel<<<blocks, kBlockThreads>>>(scene, d_origins, num_origins, d_out_indices);
    cudaDeviceSynchronize();
}

void nearest_site_indices_from_rays_cuda(
    const SceneGpu& scene,
    const float* d_rays,
    int num_rays,
    int* d_out_indices
) {
    if (num_rays <= 0) return;
    const int blocks = (num_rays + kBlockThreads - 1) / kBlockThreads;
    nearest_from_rays_grid_kernel<<<blocks, kBlockThreads>>>(scene, d_rays, num_rays, d_out_indices);
    cudaDeviceSynchronize();
}

}  // namespace voronoi
