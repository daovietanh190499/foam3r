#include "voronoi/scene.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace voronoi {

static void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

static int max_degree_from_offsets(const std::vector<uint32_t>& offsets) {
    int max_deg = 0;
    for (size_t i = 1; i < offsets.size(); ++i) {
        max_deg = std::max(max_deg, static_cast<int>(offsets[i] - offsets[i - 1]));
    }
    return max_deg;
}

static void padded_to_csr(
    int n,
    int md,
    const std::vector<int>& nbr_idx,
    const std::vector<unsigned char>& nbr_valid,
    std::vector<uint32_t>& adjacency,
    std::vector<uint32_t>& offsets
) {
    offsets.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        int count = 0;
        for (int f = 0; f < md; ++f) {
            if (nbr_valid[i * md + f]) ++count;
        }
        offsets[i + 1] = offsets[i] + static_cast<uint32_t>(count);
    }

    adjacency.resize(offsets[n]);
    for (int i = 0; i < n; ++i) {
        uint32_t e = offsets[i];
        for (int f = 0; f < md; ++f) {
            if (!nbr_valid[i * md + f]) continue;
            adjacency[e++] = static_cast<uint32_t>(nbr_idx[i * md + f]);
        }
    }
}

static void upload_csr(
    SceneGpu& scene,
    const std::vector<uint32_t>& adjacency,
    const std::vector<uint32_t>& offsets
) {
    scene.adjacency_size = static_cast<int>(adjacency.size());
    scene.max_degree = max_degree_from_offsets(offsets);

    cuda_check(cudaMalloc(&scene.adjacency, adjacency.size() * sizeof(uint32_t)), "adjacency");
    cuda_check(cudaMalloc(&scene.adjacency_offsets, offsets.size() * sizeof(uint32_t)), "adjacency_offsets");
    cuda_check(
        cudaMalloc(&scene.adjacent_diff, adjacency.size() * 3 * sizeof(float)),
        "adjacent_diff"
    );

    cuda_check(
        cudaMemcpy(scene.adjacency, adjacency.data(), adjacency.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
        "adjacency H2D"
    );
    cuda_check(
        cudaMemcpy(
            scene.adjacency_offsets,
            offsets.data(),
            offsets.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ),
        "adjacency_offsets H2D"
    );

    prefetch_adjacent_diff_cuda(scene, nullptr);
    cuda_check(cudaDeviceSynchronize(), "prefetch adjacent_diff");
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

    const int nx = 72, ny = 72, nz = 48;
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

    cuda_check(cudaMalloc(&scene.grid_offsets, offsets.size() * sizeof(int)), "grid_offsets");
    cuda_check(cudaMalloc(&scene.grid_indices, indices.size() * sizeof(int)), "grid_indices");
    cuda_check(cudaMemcpy(scene.grid_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice),
               "grid_offsets H2D");
    cuda_check(cudaMemcpy(scene.grid_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice),
               "grid_indices H2D");
}

SceneGpu load_scene_gpu(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot open scene: ") + path);

    uint32_t magic = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kVortMagic) throw std::runtime_error("Invalid .vort magic");

    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    SceneGpu scene{};
    std::vector<float> pos;
    std::vector<uint32_t> adjacency;
    std::vector<uint32_t> adj_offsets;

    if (version == kVortVersion) {
        SceneHeader hdr{};
        hdr.magic = magic;
        hdr.version = version;
        in.read(reinterpret_cast<char*>(&hdr) + 8, sizeof(hdr) - 8);

        const int n = static_cast<int>(hdr.num_sites);
        const int md = static_cast<int>(hdr.max_degree);
        const int sh_dim = static_cast<int>(hdr.sh_dim);

        std::vector<float> sh(n * sh_dim), dens(n);
        std::vector<int> nbr_idx(n * md);
        std::vector<float> nbr_diff(n * md * 3);
        std::vector<unsigned char> nbr_valid(n * md);

        pos.resize(n * 3);
        in.read(reinterpret_cast<char*>(pos.data()), pos.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(sh.data()), sh.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(dens.data()), dens.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(nbr_idx.data()), nbr_idx.size() * sizeof(int));
        in.read(reinterpret_cast<char*>(nbr_diff.data()), nbr_diff.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(nbr_valid.data()), nbr_valid.size());

        padded_to_csr(n, md, nbr_idx, nbr_valid, adjacency, adj_offsets);

        scene.num_sites = n;
        scene.sh_degree = static_cast<int>(hdr.sh_degree);
        scene.sh_dim = sh_dim;
        scene.weight_threshold = hdr.weight_threshold;
        scene.max_steps = static_cast<int>(hdr.max_steps);

        cuda_check(cudaMalloc(&scene.positions, pos.size() * sizeof(float)), "positions");
        cuda_check(cudaMalloc(&scene.sh_attrs, sh.size() * sizeof(float)), "sh");
        cuda_check(cudaMalloc(&scene.density, dens.size() * sizeof(float)), "density");
        cuda_check(cudaMemcpy(scene.positions, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "positions H2D");
        cuda_check(cudaMemcpy(scene.sh_attrs, sh.data(), sh.size() * sizeof(float), cudaMemcpyHostToDevice), "sh H2D");
        cuda_check(cudaMemcpy(scene.density, dens.data(), dens.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "density H2D");
    } else if (version == kVortVersionCsr) {
        SceneHeaderV2 hdr{};
        hdr.magic = magic;
        hdr.version = version;
        in.read(reinterpret_cast<char*>(&hdr) + 8, sizeof(hdr) - 8);

        const int n = static_cast<int>(hdr.num_sites);
        const int adj_size = static_cast<int>(hdr.adjacency_size);
        const int sh_dim = static_cast<int>(hdr.sh_dim);

        pos.resize(n * 3);
        std::vector<float> sh(n * sh_dim), dens(n);
        adjacency.resize(adj_size);
        adj_offsets.resize(n + 1);

        in.read(reinterpret_cast<char*>(pos.data()), pos.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(sh.data()), sh.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(dens.data()), dens.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(adjacency.data()), adjacency.size() * sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(adj_offsets.data()), adj_offsets.size() * sizeof(uint32_t));

        scene.num_sites = n;
        scene.sh_degree = static_cast<int>(hdr.sh_degree);
        scene.sh_dim = sh_dim;
        scene.weight_threshold = hdr.weight_threshold;
        scene.max_steps = static_cast<int>(hdr.max_steps);

        cuda_check(cudaMalloc(&scene.positions, pos.size() * sizeof(float)), "positions");
        cuda_check(cudaMalloc(&scene.sh_attrs, sh.size() * sizeof(float)), "sh");
        cuda_check(cudaMalloc(&scene.density, dens.size() * sizeof(float)), "density");
        cuda_check(cudaMemcpy(scene.positions, pos.data(), pos.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "positions H2D");
        cuda_check(cudaMemcpy(scene.sh_attrs, sh.data(), sh.size() * sizeof(float), cudaMemcpyHostToDevice), "sh H2D");
        cuda_check(cudaMemcpy(scene.density, dens.data(), dens.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "density H2D");
    } else {
        throw std::runtime_error("Unsupported .vort version");
    }

    upload_csr(scene, adjacency, adj_offsets);
    build_spatial_grid(pos, scene.num_sites, scene);
    return scene;
}

void free_scene_gpu(SceneGpu& scene) {
    if (scene.positions) cudaFree(scene.positions);
    if (scene.sh_attrs) cudaFree(scene.sh_attrs);
    if (scene.density) cudaFree(scene.density);
    if (scene.adjacency) cudaFree(scene.adjacency);
    if (scene.adjacency_offsets) cudaFree(scene.adjacency_offsets);
    if (scene.adjacent_diff) cudaFree(scene.adjacent_diff);
    if (scene.grid_offsets) cudaFree(scene.grid_offsets);
    if (scene.grid_indices) cudaFree(scene.grid_indices);
    scene = {};
}

}  // namespace voronoi
