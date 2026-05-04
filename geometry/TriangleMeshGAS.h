#pragma once
#include "3rdParty/helper_math.h"
#include "OptiX_Base.h"

class TriangleMeshGAS {
public:
	CUDABuffer asBuffer;
	CUDABuffer tempBuffer;
	CUDABuffer outputBuffer;
	size_t temp_buffer_size;
	size_t output_buffer_size;

	OptixTraversableHandle handle{0};

	void build(float3* dVertices, uint3* dIndices, size_t vSize, size_t tSize){
        	OptixTraversableHandle asHandle{ 0 };

	OptixBuildInput triangleInput = {};
	triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

	CUdeviceptr d_vertices = (CUdeviceptr)dVertices;
	CUdeviceptr d_faces = (CUdeviceptr)dIndices;

	triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
	triangleInput.triangleArray.numVertices = vSize;
	triangleInput.triangleArray.vertexBuffers = &d_vertices;

	triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	triangleInput.triangleArray.indexStrideInBytes = sizeof(uint3);
	triangleInput.triangleArray.numIndexTriplets = tSize;
	triangleInput.triangleArray.indexBuffer = d_faces;

	//We change the material information when instance step
	uint32_t triangleInputFlags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;

	//Each GAS only have one SBT in this implementation
	triangleInput.triangleArray.flags = &triangleInputFlags;
	triangleInput.triangleArray.numSbtRecords = 1;
	triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
	triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
	triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;


	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSize;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		optixGlobalParams.optixContext,
		&accelOptions,
		&triangleInput,
		1,
		&blasBufferSize));

	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));

	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();

	temp_buffer_size = blasBufferSize.tempSizeInBytes;
	output_buffer_size = blasBufferSize.outputSizeInBytes;
	tempBuffer.alloc(blasBufferSize.tempSizeInBytes);
	outputBuffer.alloc(blasBufferSize.outputSizeInBytes);

	cudaEvent_t start, end;
	cudaEventCreate(&start); cudaEventCreate(&end);

	cudaEventRecord(start, 0);
	OPTIX_CHECK(optixAccelBuild(optixGlobalParams.optixContext,
		0,
		&accelOptions,
		&triangleInput,
		1,
		tempBuffer.d_pointer(),
		tempBuffer.sizeInBytes,

		outputBuffer.d_pointer(),
		outputBuffer.sizeInBytes,

		&asHandle,

		&emitDesc, 1
	));
	cudaEventRecord(end, 0);
	CUDA_SYNC_CHECK();

	float times;
	cudaEventElapsedTime(&times, start, end);

	//std::cout << "Build accel time : " << times << "ms" << std::endl;

	cudaEventDestroy(start); cudaEventDestroy(end);

	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);

	asBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixGlobalParams.optixContext,
		0,
		asHandle,
		asBuffer.d_pointer(),
		asBuffer.sizeInBytes,
		&asHandle));
	CUDA_SYNC_CHECK();

	compactedSizeBuffer.free();
	outputBuffer.free();
	tempBuffer.free();

	handle = asHandle;
    }
};
