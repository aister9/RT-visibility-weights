#include "visibility.h"
#include "3rdParty/TimeChecker.h"
#include "geometry/triangle_utils.h"

#include "params.h"

#include "program.h"
#include "geometry/triangleMeshGAS.h"

#include <iostream>
#include <map>


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DummyRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) TriangleMeshRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    FaceMeshSBTData data;
};

__global__ void __kernel_source_sink_update__(cell_info_t* cell_infos, const uint2* segMinMaxIDX, const float2* segMinMaxDist, const uint segSize){
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x; // segment idx

	if (tIdx >= segSize)
		return;

    const uint2 minMaxIdx = segMinMaxIDX[tIdx];
    const float2 minMaxDist = segMinMaxDist[tIdx];

    if(minMaxIdx.x != UINT_MAX && minMaxDist.x < 1e8f){
        cell_infos[minMaxIdx.x].s = (float)(INT_MAX / 8);
    }
    if(minMaxIdx.y != UINT_MAX && minMaxDist.y >= 0.0f){
        atomicAdd(&cell_infos[minMaxIdx.y].t, 1.0f);
    }
}

void SPIN::RTVisiblity::compute_n_chunks(int P, int C, int T,
            const float3* vertices,
            const float3* cameras,
            const uint* camNums,
            const uint* pointViews, // flattened list of camera indices that see each point, length = sum(camNums)
            const Tetrahedron* cells,
            const cell_info_t* cell_infos,
            const uint infInd
        ){
    std::unordered_map<std::string, size_t> memoryCheck;

    uint3* resTri;
    uint2* resTetTriInfo;
    uint2* faceTriInfo;
    size_t triSize;
    uint2* seg;
    uint segCount;
    preprocess(P, C, T,
        vertices, cameras, camNums, pointViews,
        cells, cell_infos, infInd,
        resTri, resTetTriInfo, faceTriInfo, triSize, seg, segCount); 

    float sigma;
    float inv2SigmaSq;
    auto sigmaComputeTime = SPIN::TimeCheck([&] {
        sigma = sigma_calculate((float3*)vertices, resTri, triSize);
        inv2SigmaSq = (0.5f / (sigma * sigma));
        });
    std::cout << "Elapsed (sigma) : " << sigmaComputeTime << " ms" << std::endl;

    RTVisibilityProgram* program = (RTVisibilityProgram*)optixGlobalParams.programList["Triangle"];

    uint nChunk = 3;
    std::vector<uint> chunkPerSize(nChunk);
    for (uint i = 0; i < nChunk; i++) {
        chunkPerSize[i] = triSize / nChunk + (i < triSize % nChunk);
    }

    uint3* trianglePointer = resTri;
    uint2* lut2Pointer = faceTriInfo;
    
    RTVisParamsNChunks param;
    CUDABuffer launchParamBuffer;
    launchParamBuffer.alloc(sizeof(RTVisParamsNChunks));

    std::vector<uint2> segMinMaxIdxHost(segCount, make_uint2(UINT_MAX, UINT_MAX));
    std::vector<float2> segMinMaxDistanceHost(segCount, make_float2(1e8f, -1.0f));
    
    param.dTet = (Tetrahedron*)cells;
    param.dWeight = (cell_info_t*)cell_infos;
    param.sigma = sigma;
    param.inv2SigmaSq = inv2SigmaSq;
    param.cam = (float3*)cameras;
    param.cSize = C;
    param.vertex = (float3*)vertices;
    param.vSize = P;
    param.segment = seg;
    param.sSize = segCount;
    param.segMinMaxIdx = upload(segMinMaxIdxHost);
    param.segMinMaxDistance = upload(segMinMaxDistanceHost);
    param.traversable = 0;

    memoryCheck["Segment"] = sizeof(uint2) * param.sSize;
    memoryCheck["Segment_Support"] = sizeof(uint2) * param.sSize + sizeof(float2) * param.sSize;
	memoryCheck["LaunchParam"] = launchParamBuffer.sizeInBytes;

    for(int i = 0; i<nChunk; i++){
        TriangleMeshGAS gas;

        auto buildGASTime = SPIN::TimeCheck([&] {
            gas.build((float3*)vertices, trianglePointer, P, chunkPerSize[i]);
            program->updateHitProgramRecord({(float3*)vertices, trianglePointer, resTetTriInfo, lut2Pointer }); // resTetTriInfo follows original Tri infomation
            //program->updateHitProgramRecord({(float3*)vertices, resTri, resTetTriInfo, nullptr });
        });
        std::cout << "Elapsed (build GAS) : " << buildGASTime << " ms" << std::endl;

        //Compute Memory peak
        memoryCheck["GAS_AS"] = std::max(memoryCheck["GAS_AS"], gas.asBuffer.sizeInBytes);
        memoryCheck["GAS_temp"] = std::max(memoryCheck["GAS_temp"], gas.tempBuffer.sizeInBytes);
        memoryCheck["GAS_output"] = std::max(memoryCheck["GAS_output"], gas.outputBuffer.sizeInBytes);

        param.traversable = gas.handle;
        launchParamBuffer.upload(&param,1);

        int sqrtSize = ceil(sqrt(param.sSize));
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        auto traverseTime = SPIN::TimeCheck([&]() {
            cudaEventRecord(start, 0);
            OPTIX_CHECK(optixLaunch(program->pipeline, 0,
                launchParamBuffer.d_pointer(),
                launchParamBuffer.sizeInBytes,
                &program->sbt,
                sqrtSize,
                sqrtSize,
                1
            ));
            cudaEventRecord(end, 0);
            cudaCheckError(cudaStreamSynchronize(0)); //illigeal memory access
        });
        cudaEventSynchronize(end);
        float traverseTimeEvent;
        cudaEventElapsedTime(&traverseTimeEvent, start, end);
        std::cout << "Traverse: " << traverseTime << "ms (" << traverseTimeEvent <<" ms for CUDA event)" << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    
        trianglePointer += chunkPerSize[i];
        lut2Pointer += chunkPerSize[i];
    }

    const uint blockSize = 256;
    const uint gridSize = static_cast<uint>((param.sSize + blockSize - 1) / blockSize);
    __kernel_source_sink_update__<<<gridSize, blockSize>>>(
        param.dWeight,
        param.segMinMaxIdx,
        param.segMinMaxDistance,
        param.sSize
    );
    cudaCheckError(cudaDeviceSynchronize());

    //launchParamBuffer.free();
    cudaFree(resTri);
    cudaFree(resTetTriInfo);
    ///////////////////////////////////////////////////////////
    cudaFree(param.segment);
    cudaFree(param.segMinMaxIdx);
    cudaFree(param.segMinMaxDistance);

    //globalTimeParams["PREPROCESSING"] = divideTime + findUniqueTime + sigmaComputeTime + buildGASTime + buildSegmentTime;
    //globalTimeParams["TRAVERSE"] = traverseTime;

    memoryCheck["Vertices"] = sizeof(float3) * P;  // vertices
    memoryCheck["Cameras"] = sizeof(float3) * C;  // cameras
    memoryCheck["Tetrahedrons"] = sizeof(Tetrahedron) * T;  // tetrahedrons
    memoryCheck["Tetrahedron_Weights"] = sizeof(cell_info_t) * T;  // weights

    memoryCheck["pCamNum"] = sizeof(uint) * (P + 1);
    memoryCheck["camNum"] = sizeof(uint) * P;
    memoryCheck["camVec"] = sizeof(uint) * segCount; // flattened camera list == segment list

	std::cout << "---- Memory Check ----" << std::endl;
	size_t totalMem = 0;
	for (auto& mem : memoryCheck) {
		std::cout << mem.first << " : " << mem.second / (1024.0f * 1024.0f) << " MB" << std::endl;
		totalMem += mem.second;
	}
	std::cout << "Total : " << totalMem / (1024.0f * 1024.0f) << " MB" << std::endl;
}
