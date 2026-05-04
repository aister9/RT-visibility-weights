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

RTVisibilityProgram::RTVisibilityProgram(OptiXProgramCompileOption programOption) : OptiXPrograms(programOption) {
    //Set SBT because HD not needed SBT

    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<DummyRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        DummyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------ 
    // build miss records
    // ------------------------------------------------------------------
    std::vector<DummyRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        DummyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(DummyRecord);
    sbt.missRecordCount = (int)missRecords.size();
}

void RTVisibilityProgram::updateHitProgramRecord(FaceMeshSBTData data) {
   // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------
    TriangleMeshRecord hitgroupRecords;

    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &hitgroupRecords));

    hitgroupRecords.data = data;

    std::vector<TriangleMeshRecord> record = { hitgroupRecords };

    hitgroupRecordsBuffer.alloc_and_upload(record);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(TriangleMeshRecord);
    sbt.hitgroupRecordCount = (int)record.size();
    // ------------------------------------------------------------------
}


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

void preprocess(int P, int C, int T,
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

void SPIN::RTVisiblity::compute(int P, int C, int T,
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
    TriangleMeshGAS gas;

    auto buildGASTime = SPIN::TimeCheck([&] {
        gas.build((float3*)vertices, resTri, P, triSize);
        program->updateHitProgramRecord({(float3*)vertices, resTri, resTetTriInfo, faceTriInfo });
        //program->updateHitProgramRecord({(float3*)vertices, resTri, resTetTriInfo, nullptr });
        });
    std::cout << "Elapsed (build GAS) : " << buildGASTime << " ms" << std::endl;

    memoryCheck["GAS_AS"] = gas.asBuffer.sizeInBytes;
    memoryCheck["GAS_temp"] = gas.tempBuffer.sizeInBytes;
    memoryCheck["GAS_output"] = gas.outputBuffer.sizeInBytes;

    RTVisParams param;
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
    param.traversable = gas.handle;
    
    memoryCheck["Segment"] = sizeof(uint2) * param.sSize;

    CUDABuffer launchParamBuffer;
    launchParamBuffer.alloc(sizeof(RTVisParams));
    launchParamBuffer.upload(&param,1);

	memoryCheck["LaunchParam"] = launchParamBuffer.sizeInBytes;

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

   /* std::vector<cell_info_t> check = download(param.dWeight, gpuMetadata.tetSize);
    for (auto& c : check) {
        std::cout << c.s << " "<< c.f[0] << " " << c.f[1] << " " << c.f[2] << " " << c.f[3] << " "  << c.t << std::endl;
    }*/

    //launchParamBuffer.free();
    cudaFree(resTri);
    cudaFree(resTetTriInfo);
    ///////////////////////////////////////////////////////////
    cudaFree(param.segment);

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