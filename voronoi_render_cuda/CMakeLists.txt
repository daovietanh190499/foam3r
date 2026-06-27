cmake_minimum_required(VERSION 3.18)
project(voronoi_render_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Override: cmake -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}" NO_DEFAULT_PATH)

add_library(voronoi_render_cuda_core STATIC
  src/scene.cpp
  src/trace.cu
  src/nn_lookup.cu
)
target_include_directories(voronoi_render_cuda_core PUBLIC include)
target_link_libraries(voronoi_render_cuda_core PUBLIC CUDA::cudart)

pybind11_add_module(voronoi_render_cuda src/bindings.cpp)
target_link_libraries(voronoi_render_cuda PRIVATE voronoi_render_cuda_core CUDA::cudart)
target_include_directories(voronoi_render_cuda PRIVATE include)
