#include "voronoi_render/scene.h"
#include "voronoi_render/sh.cuh"

#include <cuda_runtime.h>

namespace voronoi {

__device__ float3 load_float3(const float* ptr, int idx) {
    const float* p = ptr + idx * 3;
    return {p[0], p[1], p[2]};
}

__global__ void trace_rays_kernel(
    const SceneGpu scene,
    const float* rays,
    const int* start_cells,
    int num_rays,
    float* out_rgba
) {
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    float3 origin = load_float3(rays, ray_id * 2);
    float3 dir = normalize3(load_float3(rays, ray_id * 2 + 1));

    float sh_basis_vals[kShBasisSize];
    sh_basis(scene.sh_degree, dir, sh_basis_vals);

    float t0 = 0.0f;
    float transmittance = 1.0f;
    float rgb[3] = {0.0f, 0.0f, 0.0f};
    bool alive = true;
    int current = start_cells[ray_id];

    const int max_steps = scene.max_steps;
    const int n = scene.num_sites;
    const int md = scene.max_degree;

    for (int step = 0; step < max_steps && alive; ++step) {
        int idx = current;
        if (idx < 0 || idx >= n) {
            // Match Python: clamp negative to 0 for neighbor lookup only.
            if (idx < 0) idx = 0;
            else break;
        }

        float3 primal = load_float3(scene.positions, idx);
        float t1 = 1e30f;
        int next_cell = idx;
        bool any_face = false;

        for (int f = 0; f < md; ++f) {
            int flat = idx * md + f;
            if (!scene.nbr_valid[flat]) continue;

            float3 diff = load_float3(scene.nbr_diff, flat);
            float3 face_origin = add3(primal, mul3f(diff, 0.5f));
            float3 face_normal = diff;
            float dp = dot3(face_normal, dir);
            if (dp <= 0.0f) continue;

            float num = dot3(sub3(face_origin, origin), face_normal);
            float t_hit = num / dp;
            if (t_hit < t1) {
                t1 = t_hit;
                next_cell = scene.nbr_idx[flat];
                any_face = true;
            }
        }

        if (!any_face) break;

        bool advance = alive;
        bool contrib = advance && (t1 > t0);

        if (contrib) {
            const float* sh_vals = scene.sh_attrs + idx * scene.sh_dim;
            float site_rgb[3];
            eval_sh_rgb(sh_basis_vals, sh_vals, scene.sh_dim, site_rgb);

            float delta_t = fmaxf(t1 - t0, 0.0f);
            float sigma = scene.density[idx];
            float alpha = 1.0f - expf(-sigma * delta_t);
            float weight = transmittance * alpha;

            rgb[0] += weight * site_rgb[0];
            rgb[1] += weight * site_rgb[1];
            rgb[2] += weight * site_rgb[2];
            transmittance *= (1.0f - alpha);
        }

        if (advance) {
            t0 = fmaxf(t0, t1);
            current = next_cell;
        }
        alive = contrib ? (transmittance > scene.weight_threshold) : advance;
    }

    float opacity = fminf(fmaxf(1.0f - transmittance, 0.0f), 1.0f);
    float* out = out_rgba + ray_id * 4;
    out[0] = rgb[0];
    out[1] = rgb[1];
    out[2] = rgb[2];
    out[3] = opacity;
}

void trace_rays_cuda(
    const SceneGpu& scene,
    const float* rays,
    const int* start_cells,
    int num_rays,
    float* out_rgba
) {
    const int threads = 256;
    const int blocks = (num_rays + threads - 1) / threads;
    trace_rays_kernel<<<blocks, threads>>>(scene, rays, start_cells, num_rays, out_rgba);
    cudaDeviceSynchronize();
}

}  // namespace voronoi
