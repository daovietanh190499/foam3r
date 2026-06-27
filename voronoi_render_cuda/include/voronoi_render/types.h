#pragma once

#include <cstdint>

namespace voronoi {

constexpr uint32_t kVortMagic = 0x54524F56u;  // "VORT"
constexpr uint32_t kVortVersion = 1;
constexpr int kMaxShDegree = 3;
constexpr int kShBasisSize = 16;
constexpr int kShAttrSize = 48;  // 3 * 16 interleaved

struct SceneHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_sites;
    uint32_t max_degree;
    uint32_t sh_degree;
    float weight_threshold;
    float max_steps;
    uint32_t sh_dim;
};

struct float3 {
    float x, y, z;
};

}  // namespace voronoi
