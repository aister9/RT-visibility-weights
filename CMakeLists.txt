cmake_minimum_required(VERSION 3.14)

project(rtVis LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit 11.8 REQUIRED)

#OPTIX
include(FetchContent)
add_definitions(-DBYTE=uint8_t)
FetchContent_Declare(
        optix
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG v7.4.0
)
FetchContent_MakeAvailable(optix)
set(OptiX_INCLUDE_DIR "${optix_SOURCE_DIR}/include")

FetchContent_Declare(
    cuBQL
    GIT_REPOSITORY https://github.com/NVIDIA/cuBQL.git
    GIT_TAG        main    
)
FetchContent_MakeAvailable(cuBQL)

add_library(optix_common INTERFACE)
target_include_directories(optix_common INTERFACE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OptiX_INCLUDE_DIR}
)
target_link_libraries(optix_common INTERFACE
    CUDA::cudart
    CUDA::cuda_driver
)

add_subdirectory(optix_base)
add_subdirectory(rt_visibility)

option(RTVIS_BUILD_PYTHON "Build Python bindings" OFF)
if(RTVIS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

add_executable(rt_visibility_test
    test.cpp
)
target_link_libraries(rt_visibility_test PRIVATE
    rt_visibility_lib
)
target_include_directories(rt_visibility_test PRIVATE
    ${CMAKE_SOURCE_DIR}
)
target_compile_definitions(rt_visibility_test PRIVATE
    RTVIS_PTX_DIR="${CMAKE_BINARY_DIR}/ptx/"
    RTVIS_TEST_DATASET="${CMAKE_SOURCE_DIR}/testDataset/temp_decimate.bin"
)
add_dependencies(rt_visibility_test copy_ptx_to_ptx_dir)
