include(FetchContent)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(Torch QUIET)
if(NOT Torch_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PREFIX_RESULT
    )

    if(NOT TORCH_PREFIX_RESULT EQUAL 0 OR TORCH_CMAKE_PREFIX_PATH STREQUAL "")
        message(FATAL_ERROR
            "Torch CMake package was not found. "
            "Install PyTorch for the selected Python interpreter or set CMAKE_PREFIX_PATH to your libtorch installation."
        )
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    find_package(Torch REQUIRED)
endif()

pybind11_add_module(rtvis_python MODULE binding.cpp)

target_link_libraries(rtvis_python PRIVATE
    rt_visibility_lib
    ${TORCH_LIBRARIES}
)

target_include_directories(rtvis_python PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(rtvis_python PRIVATE cxx_std_17)
target_compile_definitions(rtvis_python PRIVATE
    RTVIS_PTX_DIR="${CMAKE_BINARY_DIR}/ptx/"
)

if(MSVC)
    target_compile_options(rtvis_python PRIVATE /bigobj)
endif()

add_dependencies(rtvis_python copy_ptx_to_ptx_dir)
