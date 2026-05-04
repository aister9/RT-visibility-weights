#pragma once

#include "3rdParty/helper_math.h"
#include "geometry/tetrahedron.h"

namespace SPIN{

    class RTVisiblity{
        public:

        static void preprocess(int P, int C, int T,
            const float3* vertices,
            const float3* cameras,
            const uint* camNums, // length = P, number of cameras that see each point
            const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
            const Tetrahedron* cells,
            const cell_info_t* cell_infos,
            const uint infInd,
            uint3 *&resTri, uint2* &resTetTriInfo, uint2* &faceTriInfo, size_t &triSize,
            uint2*& seg, uint& segCount
        );

        static void preprocessWithoutUnique(int P, int C, int T,
            const float3* vertices,
            const float3* cameras,
            const uint* camNums, // length = P, number of cameras that see each point
            const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
            const Tetrahedron* cells,
            const cell_info_t* cell_infos,
            const uint infInd,
            uint3 *&resTri, uint2* &resTetTriInfo, uint2* &faceTriInfo, size_t &triSize,
            uint2*& seg, uint& segCount
        );

        static void compute(int P, int C, int T,
            const float3* vertices,
            const float3* cameras,
            const uint* camNums,
            const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
            const Tetrahedron* cells,
            const cell_info_t* cell_infos,
            const uint infInd
        );

        static void compute_n_chunks(int P, int C, int T,
            const float3* vertices,
            const float3* cameras,
            const uint* camNums,
            const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
            const Tetrahedron* cells,
            const cell_info_t* cell_infos,
            const uint infInd
        );
    };

}
