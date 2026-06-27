#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "voronoi_render/nn_lookup.cuh"
#include "voronoi_render/scene.h"

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

class CudaScene {
public:
    explicit CudaScene(const std::string& path) : scene_(voronoi::load_scene_gpu(path.c_str())) {}
    ~CudaScene() { voronoi::free_scene_gpu(scene_); }

    py::array_t<int> nearest_site_indices(
        py::array_t<float, py::array::c_style | py::array::forcecast> origins
    ) {
        auto buf = origins.request();
        if (buf.ndim != 2 || buf.shape[1] != 3) {
            throw std::runtime_error("origins must be (N, 3)");
        }
        const int n = static_cast<int>(buf.shape[0]);

        float* d_origins = nullptr;
        int* d_out = nullptr;
        cuda_check(cudaMalloc(&d_origins, n * 3 * sizeof(float)), "cudaMalloc origins");
        cuda_check(cudaMalloc(&d_out, n * sizeof(int)), "cudaMalloc nn out");

        cuda_check(
            cudaMemcpy(d_origins, buf.ptr, n * 3 * sizeof(float), cudaMemcpyHostToDevice),
            "copy origins");

        voronoi::nearest_site_indices_cuda(scene_, d_origins, n, d_out);

        std::vector<int> out(n);
        cuda_check(cudaMemcpy(out.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost), "copy nn");

        cudaFree(d_origins);
        cudaFree(d_out);

        py::array_t<int> result(n);
        std::memcpy(result.mutable_data(), out.data(), n * sizeof(int));
        return result;
    }

    py::array_t<float> trace(
        py::array_t<float, py::array::c_style | py::array::forcecast> rays,
        py::object start_cells = py::none()
    ) {
        auto rbuf = rays.request();
        if (rbuf.ndim != 2 || rbuf.shape[1] != 6) {
            throw std::runtime_error("rays must be (N, 6)");
        }

        const int num_rays = static_cast<int>(rbuf.shape[0]);
        float* d_rays = nullptr;
        int* d_start = nullptr;
        float* d_out = nullptr;
        bool own_start = false;

        cuda_check(cudaMalloc(&d_rays, num_rays * 6 * sizeof(float)), "cudaMalloc rays");
        cuda_check(cudaMalloc(&d_start, num_rays * sizeof(int)), "cudaMalloc start");
        cuda_check(cudaMalloc(&d_out, num_rays * 4 * sizeof(float)), "cudaMalloc out");

        cuda_check(
            cudaMemcpy(d_rays, rbuf.ptr, num_rays * 6 * sizeof(float), cudaMemcpyHostToDevice),
            "copy rays");

        if (start_cells.is_none()) {
            voronoi::nearest_site_indices_from_rays_cuda(scene_, d_rays, num_rays, d_start);
            own_start = true;
        } else {
            auto sbuf = start_cells.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>().request();
            if (sbuf.ndim != 1 || sbuf.shape[0] != num_rays) {
                cudaFree(d_rays);
                cudaFree(d_start);
                cudaFree(d_out);
                throw std::runtime_error("start_cells must be (N,)");
            }
            cuda_check(
                cudaMemcpy(d_start, sbuf.ptr, num_rays * sizeof(int), cudaMemcpyHostToDevice),
                "copy start");
        }

        voronoi::trace_rays_cuda(scene_, d_rays, d_start, num_rays, d_out);

        std::vector<float> out(num_rays * 4);
        cuda_check(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");

        cudaFree(d_rays);
        cudaFree(d_start);
        cudaFree(d_out);

        (void)own_start;
        py::array_t<float> result({num_rays, 4});
        std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(float));
        return result;
    }

    int num_sites() const { return scene_.num_sites; }
    int max_degree() const { return scene_.max_degree; }

private:
    voronoi::SceneGpu scene_;
};

PYBIND11_MODULE(voronoi_render_cuda, m) {
    m.doc() = "CUDA Voronoi volume renderer";
    py::class_<CudaScene>(m, "CudaScene")
        .def(py::init<const std::string&>())
        .def("nearest_site_indices", &CudaScene::nearest_site_indices,
             "GPU 1-NN using scene positions (tiled, no extra site buffer).")
        .def("trace", &CudaScene::trace, py::arg("rays"), py::arg("start_cells") = py::none(),
             "Trace rays. If start_cells is None, NN lookup runs on GPU first.")
        .def_property_readonly("num_sites", &CudaScene::num_sites)
        .def_property_readonly("max_degree", &CudaScene::max_degree);
}
