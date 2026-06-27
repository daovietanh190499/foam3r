cmake_minimum_required(VERSION 3.18)
project(voronoi_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

add_library(voronoi_cuda_core STATIC
  src/scene.cpp
  src/adjacency.cu
  src/trace_fused.cu
)
target_include_directories(voronoi_cuda_core PUBLIC include)
target_link_libraries(voronoi_cuda_core PUBLIC CUDA::cudart)
target_compile_options(voronoi_cuda_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

pybind11_add_module(voronoi_cuda src/bindings.cpp)
target_link_libraries(voronoi_cuda PRIVATE voronoi_cuda_core CUDA::cudart)
target_include_directories(voronoi_cuda PRIVATE include)
target_compile_options(voronoi_cuda PRIVATE -O3)
