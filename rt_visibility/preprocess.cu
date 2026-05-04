#include "visibility.h"
#include "3rdParty/TimeChecker.h"
#include "geometry/triangle_utils.h"

#include <iostream>
#include <map>

__global__ void __kernel_build_segment__(uint vSize, uint* camVec, uint* pCamNum, uint2* seg) {
	int tIdx = blockIdx.x * blockDim.x + threadIdx.x; // segment idx

	if (tIdx >= vSize)
		return;

	for (uint start = pCamNum[tIdx]; start < pCamNum[tIdx + 1]; start++) {
		seg[start] = make_uint2(tIdx, camVec[start]);
	}
}

void buildSegment(uint vSize, uint* camVec, uint* pCamNum, uint2*& seg, uint& segCount) {
	cudaMemcpy(&segCount, pCamNum + vSize, sizeof(uint), cudaMemcpyDeviceToHost);

	cudaMalloc(&seg, sizeof(uint2) * segCount);

	uint2 layout = { 256, ceil(segCount / 256.) }; // block, grid
	__kernel_build_segment__ << <layout.y, layout.x >> > (vSize, camVec, pCamNum, seg);
	cudaCheckError(cudaDeviceSynchronize());
}

void SPIN::RTVisiblity::preprocess(int P, int C, int T,
    const float3* vertices,
    const float3* cameras,
    const uint* camNums, // length = P, number of cameras that see each point
    const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
    const Tetrahedron* cells,
    const cell_info_t* cell_infos,
    const uint infInd,
    uint3 *&resTri, uint2* &resTetTriInfo, uint2* &faceTriInfo, size_t &triSize,
    uint2*& seg, uint& segCount
) {
    uint3* dTris; // 4 Tris per 1 Tet
    cudaMalloc(&dTris, sizeof(uint3) * T* 4);
    uint2* dTetTriInfo;
    cudaMalloc(&dTetTriInfo, sizeof(uint2) * T * 4);

    auto divideTime = SPIN::TimeCheck([&] {
        triangle_divider((Tetrahedron*)cells, dTris, dTetTriInfo, T, T * 4); //Needs remove the infinite triangles
        });

    auto compactionTime = SPIN::TimeCheck([&] {
        remove_infinite_triangles(dTris, dTetTriInfo, T * 4, resTri, resTetTriInfo, triSize, infInd);
        });
    std::cout << "Remove infinite : " << T * 4 << " to " << triSize << std::endl;
    std::cout << "Elapsed (triangle divide) : " << divideTime << " ms" << std::endl;
    std::cout << "Elapsed (triangle compaction) : " << compactionTime << " ms" << std::endl;

    //for test
    size_t uniqueTriSize = 0;
    auto findUniqueTime = SPIN::TimeCheck([&]
        {
            uint3 *newTris;
            get_unique_triangles(resTri, triSize, newTris, faceTriInfo, uniqueTriSize);
            cudaFree(resTri);
            resTri = newTris;
        }
    );
    std::cout << "Triangle Count : " << triSize << " to " << uniqueTriSize << std::endl;
    std::cout << "Elapsed (triangle unique) : " << findUniqueTime << " ms" << std::endl;
    triSize = uniqueTriSize;

    uint* pCamNum;
	cudaMalloc(&pCamNum, sizeof(uint) * (P + 1));
	prefixSum((uint*)camNums, pCamNum, P); // prefix sum to get the start index of each point's camera list in the flattened cam list
    buildSegment(P, (uint*)pointViews, (uint*)pCamNum, seg, segCount);
    cudaFree(pCamNum);
    cudaFree(dTris);
    cudaFree(dTetTriInfo);
}


void SPIN::RTVisiblity::preprocessWithoutUnique(int P, int C, int T,
    const float3* vertices,
    const float3* cameras,
    const uint* camNums, // length = P, number of cameras that see each point
    const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
    const Tetrahedron* cells,
    const cell_info_t* cell_infos,
    const uint infInd,
    uint3 *&resTri, uint2* &resTetTriInfo, uint2* &faceTriInfo, size_t &triSize,
    uint2*& seg, uint& segCount
) {
    uint3* dTris; // 4 Tris per 1 Tet
    cudaMalloc(&dTris, sizeof(uint3) * T* 4);
    uint2* dTetTriInfo;
    cudaMalloc(&dTetTriInfo, sizeof(uint2) * T * 4);

    auto divideTime = SPIN::TimeCheck([&] {
        triangle_divider((Tetrahedron*)cells, dTris, dTetTriInfo, T, T * 4); //Needs remove the infinite triangles
        });

    auto compactionTime = SPIN::TimeCheck([&] {
        remove_infinite_triangles(dTris, dTetTriInfo, T * 4, resTri, resTetTriInfo, triSize, infInd);
        });
    std::cout << "Remove infinite : " << T * 4 << " to " << triSize << std::endl;
    std::cout << "Elapsed (triangle divide) : " << divideTime << " ms" << std::endl;
    std::cout << "Elapsed (triangle compaction) : " << compactionTime << " ms" << std::endl;

    faceTriInfo = nullptr;

    uint* pCamNum;
	cudaMalloc(&pCamNum, sizeof(uint) * (P + 1));
	prefixSum((uint*)camNums, pCamNum, P); // prefix sum to get the start index of each point's camera list in the flattened cam list
    buildSegment(P, (uint*)pointViews, (uint*)pCamNum, seg, segCount);
    cudaFree(pCamNum);
    cudaFree(dTris);
    cudaFree(dTetTriInfo);
}