#include "voronoi/adjacency.cuh"
#include "voronoi/math.cuh"

#include <cuda_runtime.h>

namespace voronoi {

namespace {

constexpr int kPrefetchThreads = 256;

__global__ void prefetch_adjacent_diff_kernel(
    const float* __restrict__ positions,
    int num_sites,
    const uint32_t* __restrict__ adjacency,
    const uint32_t* __restrict__ adjacency_offsets,
    float* __restrict__ adjacent_diff
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sites) return;

    const float3 p = load_pos(positions, i);
    const uint32_t start = __ldg(&adjacency_offsets[i]);
    const uint32_t end = __ldg(&adjacency_offsets[i + 1]);

    for (uint32_t e = start; e < end; ++e) {
        const uint32_t adj = __ldg(&adjacency[e]);
        const float3 q = load_pos(positions, adj);
        const float3 diff = sub3(q, p);
        float* out = adjacent_diff + e * 3;
        out[0] = diff.x;
        out[1] = diff.y;
        out[2] = diff.z;
    }
}

}  // namespace

void prefetch_adjacent_diff_cuda(const SceneGpu& scene, cudaStream_t stream) {
    if (scene.num_sites <= 0 || scene.adjacency_size <= 0) return;
    const int blocks = (scene.num_sites + kPrefetchThreads - 1) / kPrefetchThreads;
    prefetch_adjacent_diff_kernel<<<blocks, kPrefetchThreads, 0, stream>>>(
        scene.positions,
        scene.num_sites,
        scene.adjacency,
        scene.adjacency_offsets,
        scene.adjacent_diff
    );
}

}  // namespace voronoi
