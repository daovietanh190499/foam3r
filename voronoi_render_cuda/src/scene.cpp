#include "voronoi_render/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace voronoi {

static void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

static void build_spatial_grid(const std::vector<float>& pos, int n, SceneGpu& scene) {
    if (n <= 0) return;

    float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
    float mx[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            const float v = pos[i * 3 + c];
            mn[c] = std::min(mn[c], v);
            mx[c] = std::max(mx[c], v);
        }
    }

    const float pad = 1e-4f;
    for (int c = 0; c < 3; ++c) {
        mn[c] -= pad;
        mx[c] += pad;
        scene.grid_min[c] = mn[c];
    }

    // ~8-12 sites / cell for ~2M sites → ~250k cells.
    const int nx = 72;
    const int ny = 72;
    const int nz = 48;
    scene.grid_dims[0] = nx;
    scene.grid_dims[1] = ny;
    scene.grid_dims[2] = nz;
    scene.grid_num_cells = nx * ny * nz;

    float span[3] = {mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]};
    for (int c = 0; c < 3; ++c) {
        if (span[c] < 1e-6f) span[c] = 1.0f;
        scene.grid_inv_cell[c] = static_cast<float>(scene.grid_dims[c]) / span[c];
    }

    std::vector<int> counts(scene.grid_num_cells, 0);
    std::vector<int> cell_of_site(n);
    for (int i = 0; i < n; ++i) {
        int ix = static_cast<int>((pos[i * 3 + 0] - mn[0]) * scene.grid_inv_cell[0]);
        int iy = static_cast<int>((pos[i * 3 + 1] - mn[1]) * scene.grid_inv_cell[1]);
        int iz = static_cast<int>((pos[i * 3 + 2] - mn[2]) * scene.grid_inv_cell[2]);
        ix = std::max(0, std::min(ix, nx - 1));
        iy = std::max(0, std::min(iy, ny - 1));
        iz = std::max(0, std::min(iz, nz - 1));
        const int cell = ix + nx * (iy + iz * ny);
        cell_of_site[i] = cell;
        counts[cell]++;
    }

    std::vector<int> offsets(scene.grid_num_cells + 1, 0);
    for (int c = 0; c < scene.grid_num_cells; ++c) {
        offsets[c + 1] = offsets[c] + counts[c];
    }
    std::vector<int> cursor = offsets;
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) {
        const int cell = cell_of_site[i];
        indices[cursor[cell]++] = i;
    }

    cuda_check(cudaMalloc(&scene.grid_offsets, offsets.size() * sizeof(int)), "cudaMalloc grid_offsets");
    cuda_check(cudaMalloc(&scene.grid_indices, indices.size() * sizeof(int)), "cudaMalloc grid_indices");
    cuda_check(cudaMemcpy(scene.grid_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy grid_offsets");
    cuda_check(cudaMemcpy(scene.grid_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy grid_indices");
}

SceneGpu load_scene_gpu(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot open scene: ") + path);

    SceneHeader hdr{};
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (hdr.magic != kVortMagic || hdr.version != kVortVersion) {
        throw std::runtime_error("Invalid .vort header");
    }

    const int n = static_cast<int>(hdr.num_sites);
    const int md = static_cast<int>(hdr.max_degree);
    const int sh_dim = static_cast<int>(hdr.sh_dim);

    std::vector<float> pos(n * 3), sh(n * sh_dim), dens(n);
    std::vector<int> nbr_idx(n * md);
    std::vector<float> nbr_diff(n * md * 3);
    std::vector<unsigned char> nbr_valid(n * md);

    in.read(reinterpret_cast<char*>(pos.data()), pos.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(sh.data()), sh.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(dens.data()), dens.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(nbr_idx.data()), nbr_idx.size() * sizeof(int));
    in.read(reinterpret_cast<char*>(nbr_diff.data()), nbr_diff.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(nbr_valid.data()), nbr_valid.size());

    SceneGpu scene{};
    scene.num_sites = n;
    scene.max_degree = md;
    scene.sh_degree = static_cast<int>(hdr.sh_degree);
    scene.sh_dim = sh_dim;
    scene.weight_threshold = hdr.weight_threshold;
    scene.max_steps = static_cast<int>(hdr.max_steps);

    cuda_check(cudaMalloc(&scene.positions, pos.size() * sizeof(float)), "cudaMalloc positions");
    cuda_check(cudaMalloc(&scene.sh_attrs, sh.size() * sizeof(float)), "cudaMalloc sh");
    cuda_check(cudaMalloc(&scene.density, dens.size() * sizeof(float)), "cudaMalloc density");
    cuda_check(cudaMalloc(&scene.nbr_idx, nbr_idx.size() * sizeof(int)), "cudaMalloc nbr_idx");
    cuda_check(cudaMalloc(&scene.nbr_diff, nbr_diff.size() * sizeof(float)), "cudaMalloc nbr_diff");
    cuda_check(cudaMalloc(&scene.nbr_valid, nbr_valid.size()), "cudaMalloc nbr_valid");

    cuda_check(cudaMemcpy(scene.positions, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy positions");
    cuda_check(cudaMemcpy(scene.sh_attrs, sh.data(), sh.size() * sizeof(float), cudaMemcpyHostToDevice), "copy sh");
    cuda_check(cudaMemcpy(scene.density, dens.data(), dens.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy density");
    cuda_check(cudaMemcpy(scene.nbr_idx, nbr_idx.data(), nbr_idx.size() * sizeof(int), cudaMemcpyHostToDevice),
               "copy nbr_idx");
    cuda_check(cudaMemcpy(scene.nbr_diff, nbr_diff.data(), nbr_diff.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy nbr_diff");
    cuda_check(cudaMemcpy(scene.nbr_valid, nbr_valid.data(), nbr_valid.size(), cudaMemcpyHostToDevice),
               "copy nbr_valid");

    build_spatial_grid(pos, n, scene);

    return scene;
}

void free_scene_gpu(SceneGpu& scene) {
    if (scene.positions) cudaFree(scene.positions);
    if (scene.sh_attrs) cudaFree(scene.sh_attrs);
    if (scene.density) cudaFree(scene.density);
    if (scene.nbr_idx) cudaFree(scene.nbr_idx);
    if (scene.nbr_diff) cudaFree(scene.nbr_diff);
    if (scene.nbr_valid) cudaFree(scene.nbr_valid);
    if (scene.grid_offsets) cudaFree(scene.grid_offsets);
    if (scene.grid_indices) cudaFree(scene.grid_indices);
    scene = {};
}

}  // namespace voronoi
