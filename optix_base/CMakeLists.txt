add_library(optixBaseLib
    OptiX_Base.cpp
)
target_link_libraries(optixBaseLib PUBLIC
    optix_common
)
target_include_directories(optixBaseLib PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
