# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Define the project
project(optix_tracer LANGUAGES CXX CUDA)

# Include directories
include_directories(./)

add_library(rt_visibility_lib
    visibility_impl.cu
    ../geometry/triangle_utils.cu
)
target_link_libraries(rt_visibility_lib PUBLIC
    optixBaseLib
    optix_common
)
target_include_directories(rt_visibility_lib PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/geometry
)

# Add the library
add_library(optix_tracer_baseline OBJECT trace_baseline.cu)
target_link_libraries(optix_tracer_baseline PRIVATE
    optixBaseLib
)
# Set the property for PTX compilation
set_property(TARGET optix_tracer_baseline PROPERTY CUDA_PTX_COMPILATION ON)

# Add a custom command to copy the *.ptx file to the ptx directory
add_custom_target(copy_baseline_ptx_to_ptx_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/ptx
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:optix_tracer_baseline>
            ${CMAKE_BINARY_DIR}/ptx/trace_baseline.ptx
    COMMENT "Copying trace_baseline.ptx to ptx directory"
)
# Copy only after the forward.ptx file is generated
add_dependencies(copy_baseline_ptx_to_ptx_dir optix_tracer_baseline)
