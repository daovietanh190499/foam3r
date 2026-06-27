#pragma once

#include <cstdint>

namespace voronoi {

constexpr uint32_t kVortMagic = 0x54524F56u;
constexpr uint32_t kVortVersion = 1;
constexpr uint32_t kVortVersionCsr = 2;
constexpr int kMaxShDegree = 3;
constexpr int kShBasisSize = 16;
constexpr int kShAttrSize = 48;

struct SceneHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_sites;
    uint32_t max_degree;  // v1: pad width; v2: max degree hint
    uint32_t sh_degree;
    float weight_threshold;
    float max_steps;
    uint32_t sh_dim;
};

struct SceneHeaderV2 {
    uint32_t magic;
    uint32_t version;
    uint32_t num_sites;
    uint32_t adjacency_size;
    uint32_t sh_degree;
    float weight_threshold;
    float max_steps;
    uint32_t sh_dim;
};

struct float3 {
    float x, y, z;
};

}  // namespace voronoi
