#include "voronoi/nn.cuh"
#include "voronoi/scene.h"
#include "voronoi/sh.cuh"

#include <cuda_runtime.h>

namespace voronoi {

namespace {

constexpr int kThreads = 128;
constexpr int kFaceChunkSize = 8;

__device__ __forceinline__ void voronoi_walk(
    const SceneGpu& scene,
    float3 origin,
    float3 dir,
    const float sh_basis_vals[kShBasisSize],
    int start_cell,
    float* out_rgba
) {
    float t0 = 0.0f;
    float transmittance = 1.0f;
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    bool alive = true;
    int current = start_cell;

    const int max_steps = scene.max_steps;
    const int n = scene.num_sites;
    const float wt = scene.weight_threshold;

    for (int step = 0; step < max_steps && alive; ++step) {
        int idx = current;
        if (idx < 0 || idx >= n) {
            if (idx < 0) idx = 0;
            else break;
        }

        const float3 primal = load_pos(scene.positions, idx);
        float t1 = 1e30f;
        int next_cell = idx;
        bool any_face = false;

        const uint32_t adj_begin = __ldg(&scene.adjacency_offsets[idx]);
        const uint32_t adj_end = __ldg(&scene.adjacency_offsets[idx + 1]);
        const uint32_t num_faces = adj_end - adj_begin;

        for (uint32_t base = 0; base < num_faces; base += kFaceChunkSize) {
#pragma unroll
            for (int j = 0; j < kFaceChunkSize; ++j) {
                const uint32_t f = base + static_cast<uint32_t>(j);
                if (f >= num_faces) continue;

                const uint32_t e = adj_begin + f;
                const float* diff_p = scene.adjacent_diff + e * 3;
                const float3 diff = {__ldg(diff_p), __ldg(diff_p + 1), __ldg(diff_p + 2)};
                const float3 face_origin = add3(primal, mul3f(diff, 0.5f));
                const float dp = dot3(diff, dir);
                if (dp <= 0.0f) continue;

                const float num = dot3(sub3(face_origin, origin), diff);
                const float t_hit = num / dp;
                if (t_hit < t1) {
                    t1 = t_hit;
                    next_cell = static_cast<int>(__ldg(&scene.adjacency[e]));
                    any_face = true;
                }
            }
        }

        if (!any_face) break;

        const bool advance = alive;
        const bool contrib = advance && (t1 > t0);

        if (contrib) {
            const float* sh_vals = scene.sh_attrs + idx * scene.sh_dim;
            float site_rgb[3];
            eval_sh_rgb(sh_basis_vals, sh_vals, scene.sh_dim, site_rgb);

            const float delta_t = fmaxf(t1 - t0, 0.0f);
            const float sigma = __ldg(&scene.density[idx]);
            const float alpha = 1.0f - __expf(-sigma * delta_t);
            const float weight = transmittance * alpha;

            rgb[0] += weight * site_rgb[0];
            rgb[1] += weight * site_rgb[1];
            rgb[2] += weight * site_rgb[2];
            transmittance *= (1.0f - alpha);
        }

        if (advance) {
            t0 = fmaxf(t0, t1);
            current = next_cell;
        }
        alive = contrib ? (transmittance > wt) : advance;
    }

    const float opacity = fminf(fmaxf(1.0f - transmittance, 0.0f), 1.0f);
    out_rgba[0] = rgb[0];
    out_rgba[1] = rgb[1];
    out_rgba[2] = rgb[2];
    out_rgba[3] = opacity;
}

__global__ void fused_trace_kernel(
    SceneGpu scene,
    const float* __restrict__ rays,
    int num_rays,
    float* __restrict__ out_rgba
) {
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    const int base = ray_id * 6;
    const float3 origin = {__ldg(rays + base), __ldg(rays + base + 1), __ldg(rays + base + 2)};
    const float3 dir = normalize3({__ldg(rays + base + 3), __ldg(rays + base + 4), __ldg(rays + base + 5)});

    float sh_basis_vals[kShBasisSize];
    sh_basis(scene.sh_degree, dir, sh_basis_vals);

    int best_j = 0;
    float best_d2 = 1e30f;
    nearest_grid_query(scene, origin.x, origin.y, origin.z, best_j, best_d2);

    voronoi_walk(scene, origin, dir, sh_basis_vals, best_j, out_rgba + ray_id * 4);
}

__global__ void trace_kernel(
    SceneGpu scene,
    const float* __restrict__ rays,
    const int* __restrict__ start_cells,
    int num_rays,
    float* __restrict__ out_rgba
) {
    const int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    const int base = ray_id * 6;
    const float3 origin = {__ldg(rays + base), __ldg(rays + base + 1), __ldg(rays + base + 2)};
    const float3 dir = normalize3({__ldg(rays + base + 3), __ldg(rays + base + 4), __ldg(rays + base + 5)});

    float sh_basis_vals[kShBasisSize];
    sh_basis(scene.sh_degree, dir, sh_basis_vals);

    voronoi_walk(scene, origin, dir, sh_basis_vals, __ldg(&start_cells[ray_id]), out_rgba + ray_id * 4);
}

__global__ void nearest_kernel(
    SceneGpu scene,
    const float* __restrict__ origins,
    int num_origins,
    int* __restrict__ out_indices
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_origins) return;

    const float ox = __ldg(origins + i * 3);
    const float oy = __ldg(origins + i * 3 + 1);
    const float oz = __ldg(origins + i * 3 + 2);

    int best_j = 0;
    float best_d2 = 1e30f;
    nearest_grid_query(scene, ox, oy, oz, best_j, best_d2);
    out_indices[i] = best_j;
}

void launch_config(int num_rays, int& blocks, int& threads) {
    threads = kThreads;
    blocks = (num_rays + threads - 1) / threads;
}

}  // namespace

void trace_fused_cuda(
    const SceneGpu& scene,
    const float* d_rays,
    int num_rays,
    float* d_out_rgba,
    cudaStream_t stream
) {
    if (num_rays <= 0) return;
    int blocks, threads;
    launch_config(num_rays, blocks, threads);
    fused_trace_kernel<<<blocks, threads, 0, stream>>>(scene, d_rays, num_rays, d_out_rgba);
}

void trace_cuda(
    const SceneGpu& scene,
    const float* d_rays,
    const int* d_start_cells,
    int num_rays,
    float* d_out_rgba,
    cudaStream_t stream
) {
    if (num_rays <= 0) return;
    int blocks, threads;
    launch_config(num_rays, blocks, threads);
    trace_kernel<<<blocks, threads, 0, stream>>>(scene, d_rays, d_start_cells, num_rays, d_out_rgba);
}

void nearest_site_indices_cuda(
    const SceneGpu& scene,
    const float* d_origins,
    int num_origins,
    int* d_out_indices,
    cudaStream_t stream
) {
    if (num_origins <= 0) return;
    int blocks, threads;
    launch_config(num_origins, blocks, threads);
    nearest_kernel<<<blocks, threads, 0, stream>>>(scene, d_origins, num_origins, d_out_indices);
}

}  // namespace voronoi
