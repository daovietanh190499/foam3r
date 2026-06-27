#pragma once

#include "voronoi/scene.h"

namespace voronoi {

__device__ inline void nearest_grid_query(
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
                if (static_cast<unsigned>(ix) >= static_cast<unsigned>(nx) ||
                    static_cast<unsigned>(iy) >= static_cast<unsigned>(ny) ||
                    static_cast<unsigned>(iz) >= static_cast<unsigned>(nz)) {
                    continue;
                }
                const int cell = ix + nx * (iy + iz * ny);
                const int start = __ldg(&scene.grid_offsets[cell]);
                const int end = __ldg(&scene.grid_offsets[cell + 1]);
                for (int k = start; k < end; ++k) {
                    const int site = __ldg(&scene.grid_indices[k]);
                    const float* p = scene.positions + site * 3;
                    const float px = __ldg(p);
                    const float py = __ldg(p + 1);
                    const float pz = __ldg(p + 2);
                    const float ddx = px - ox;
                    const float ddy = py - oy;
                    const float ddz = pz - oz;
                    const float d2 = fmaf(ddx, ddx, fmaf(ddy, ddy, ddz * ddz));
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_j = site;
                    }
                }
            }
        }
    }
}

}  // namespace voronoi
