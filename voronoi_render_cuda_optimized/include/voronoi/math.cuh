#pragma once

#include "voronoi/types.h"

namespace voronoi {

__device__ inline float3 make_f3(float x, float y, float z) { return {x, y, z}; }

__device__ inline float dot3(float3 a, float3 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, a.z * b.z));
}

__device__ inline float3 sub3(float3 a, float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline float3 add3(float3 a, float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ inline float3 mul3f(float3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ inline float3 normalize3(float3 v) {
    const float n = rsqrtf(fmaxf(dot3(v, v), 1e-20f));
    return mul3f(v, n);
}

__device__ inline float3 load_pos(const float* __restrict__ positions, int idx) {
    const float* p = positions + idx * 3;
    return {__ldg(p), __ldg(p + 1), __ldg(p + 2)};
}

}  // namespace voronoi
