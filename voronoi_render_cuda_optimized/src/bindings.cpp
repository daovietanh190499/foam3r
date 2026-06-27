#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "voronoi/scene.h"

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

struct DevicePool {
    float* d_rays = nullptr;
    int* d_start = nullptr;
    float* d_out = nullptr;
    size_t cap = 0;

    ~DevicePool() { free_all(); }

    void free_all() {
        if (d_rays) cudaFree(d_rays);
        if (d_start) cudaFree(d_start);
        if (d_out) cudaFree(d_out);
        d_rays = nullptr;
        d_start = nullptr;
        d_out = nullptr;
        cap = 0;
    }

    void ensure(int num_rays) {
        if (static_cast<size_t>(num_rays) <= cap) return;
        free_all();
        cap = static_cast<size_t>(num_rays);
        cuda_check(cudaMalloc(&d_rays, cap * 6 * sizeof(float)), "pool rays");
        cuda_check(cudaMalloc(&d_start, cap * sizeof(int)), "pool start");
        cuda_check(cudaMalloc(&d_out, cap * 4 * sizeof(float)), "pool out");
    }
};

class CudaScene {
public:
    explicit CudaScene(const std::string& path) : scene_(voronoi::load_scene_gpu(path.c_str())) {
        cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    }

    ~CudaScene() {
        if (stream_) cudaStreamDestroy(stream_);
        voronoi::free_scene_gpu(scene_);
    }

    CudaScene(const CudaScene&) = delete;
    CudaScene& operator=(const CudaScene&) = delete;

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
        cuda_check(cudaMalloc(&d_origins, n * 3 * sizeof(float)), "malloc origins");
        cuda_check(cudaMalloc(&d_out, n * sizeof(int)), "malloc nn out");
        cuda_check(cudaMemcpyAsync(d_origins, buf.ptr, n * 3 * sizeof(float), cudaMemcpyHostToDevice, stream_),
                   "H2D origins");

        voronoi::nearest_site_indices_cuda(scene_, d_origins, n, d_out, stream_);
        cuda_check(cudaStreamSynchronize(stream_), "sync nn");

        py::array_t<int> result(n);
        cuda_check(cudaMemcpy(result.mutable_data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost), "D2H nn");

        cudaFree(d_origins);
        cudaFree(d_out);
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
        pool_.ensure(num_rays);

        cuda_check(
            cudaMemcpyAsync(pool_.d_rays, rbuf.ptr, num_rays * 6 * sizeof(float), cudaMemcpyHostToDevice, stream_),
            "H2D rays");

        if (start_cells.is_none()) {
            voronoi::trace_fused_cuda(scene_, pool_.d_rays, num_rays, pool_.d_out, stream_);
        } else {
            auto sbuf = start_cells.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>().request();
            if (sbuf.ndim != 1 || sbuf.shape[0] != num_rays) {
                throw std::runtime_error("start_cells must be (N,)");
            }
            cuda_check(
                cudaMemcpyAsync(pool_.d_start, sbuf.ptr, num_rays * sizeof(int), cudaMemcpyHostToDevice, stream_),
                "H2D start");
            voronoi::trace_cuda(scene_, pool_.d_rays, pool_.d_start, num_rays, pool_.d_out, stream_);
        }

        cuda_check(cudaStreamSynchronize(stream_), "sync trace");

        py::array_t<float> result({num_rays, 4});
        cuda_check(
            cudaMemcpyAsync(result.mutable_data(), pool_.d_out, num_rays * 4 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_),
            "D2H out");
        cuda_check(cudaStreamSynchronize(stream_), "sync out");
        return result;
    }

    int num_sites() const { return scene_.num_sites; }
    int max_degree() const { return scene_.max_degree; }

private:
    voronoi::SceneGpu scene_;
    cudaStream_t stream_ = nullptr;
    DevicePool pool_;
};

PYBIND11_MODULE(voronoi_cuda, m) {
    m.doc() = "Optimized CUDA Voronoi renderer (fused NN+trace, buffer pool)";
    py::class_<CudaScene>(m, "VoronoiScene")
        .def(py::init<const std::string&>())
        .def("nearest_site_indices", &CudaScene::nearest_site_indices)
        .def("trace", &CudaScene::trace, py::arg("rays"), py::arg("start_cells") = py::none(),
             "Fused GPU NN+trace when start_cells is None.")
        .def_property_readonly("num_sites", &CudaScene::num_sites)
        .def_property_readonly("max_degree", &CudaScene::max_degree);
}
