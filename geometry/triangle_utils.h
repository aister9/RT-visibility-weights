#pragma once
#include "3rdParty/helper_math.h"
#include "tetrahedron.h"
#include <vector>

#ifndef cudaCheckError
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#endif

void triangle_divider(Tetrahedron* tetraList, uint3* triangleList, uint2* tetra_tri_info, size_t tetraSize, size_t triangleSize);
void remove_infinite_triangles(uint3* triangleList, uint2* tetTrInfo, size_t triangleSize, uint3*& resTri, uint2*& resTetTriInfo, size_t& resSize, int infInd);
void get_unique_triangles(uint3* originTriangleList,
    size_t triangleSize,
	uint3*& resTri,
	uint2*& lut,
	size_t& resSize);
float sigma_calculate(float3* vlist, uint3* triList, size_t triangleSize);
void findConvexHull(Tetrahedron* tetraList, size_t tetSize, Tetrahedron*& convexHull, size_t& chSize, int infInd);
void prefixSum(uint* input, uint* output, size_t size);

template <typename T>
T* upload(const std::vector<T>& data) {
	T* res;
	cudaMalloc(&res, sizeof(T) * data.size());
	cudaMemcpy(res, data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice);
	return res;
}

template <typename T>
std::vector<T> download(const T* deviceData, size_t dataSize) {
	std::vector<T> res(dataSize);
	cudaMemcpy(res.data(), deviceData, sizeof(T) * dataSize, cudaMemcpyDeviceToHost);
	return res;
}