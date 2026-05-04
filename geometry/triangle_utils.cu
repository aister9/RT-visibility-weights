#include "triangle_utils.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/gather.h>
#include <thrust/count.h>
#include <thrust/remove.h>
#include <algorithm>
#include <limits>

__global__ void triangle_divider_kernel(Tetrahedron* tetraList, uint3* triangleList, uint2* tetra_tri_info, size_t tetraSize, size_t triangleSize) {
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	while (tidx < tetraSize) {

		if (tidx >= tetraSize)
			return;

		for (int i = 0; i < 4; i++) {
			const uint targetIDX = tidx + i * tetraSize;
			//const uint targetIDX = tidx * 4 + i;
			getTriangle(tetraList[tidx], i, triangleList[targetIDX]);
			tetra_tri_info[targetIDX] = make_uint2(tidx, i);
		}

		tidx += 1024 * 1024;
	}
}

void triangle_divider(Tetrahedron* tetraList, uint3* triangleList, uint2* tetra_tri_info, size_t tetraSize, size_t triangleSize) {
	triangle_divider_kernel << <1024, 1024 >> > (tetraList, triangleList, tetra_tri_info, tetraSize, triangleSize);
	cudaDeviceSynchronize();
}

struct removeInfTriFunctor {
	int infInd;
	__host__ __device__ removeInfTriFunctor(int inf) : infInd(inf) {}

	__host__ __device__
		bool operator()(const uint3& tri) const {
		return !hasInfinite(tri, infInd);
	}
};

void remove_infinite_triangles(uint3* triangleList, uint2* tetTrInfo, size_t triangleSize, uint3*& resTri, uint2*& resTetTriInfo, size_t& resSize, int infInd) {
	auto d_tris = thrust::device_pointer_cast(triangleList);
	auto d_info = thrust::device_pointer_cast(tetTrInfo);

	cudaMalloc(&resTri, sizeof(uint3) * triangleSize);
	cudaMalloc(&resTetTriInfo, sizeof(uint2) * triangleSize);

	auto d_out_tris = thrust::device_pointer_cast(resTri);
	auto d_out_info = thrust::device_pointer_cast(resTetTriInfo);

	auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(d_tris, d_info));
	auto zip_out = thrust::make_zip_iterator(thrust::make_tuple(d_out_tris, d_out_info));

	removeInfTriFunctor pred(infInd);

	auto end_it = thrust::copy_if(
		zip_in, zip_in + triangleSize,
		d_tris,
		zip_out,
		pred
	);

	resSize = thrust::get<0>(end_it.get_iterator_tuple()) - d_out_tris;
}

struct unique_val {
	uint firstIdx;
	uint secondIdx;
	uint3 tri;
};

struct make_key {
	const uint3* tris;
	__host__ __device__ uint3 operator()(const uint& idx) const {
		uint a = tris[idx].x, b = tris[idx].y, c = tris[idx].z;
		if (a > b) thrust::swap(a, b);
		if (b > c) thrust::swap(b, c);
		if (a > b) thrust::swap(a, b);
		return make_uint3(a, b, c);
	}
};

struct key_less {
	const uint3* keys;
	__host__ __device__ bool operator()(uint lhs, uint rhs) const {
		const uint3 k1 = keys[lhs];
		const uint3 k2 = keys[rhs];
		if (k1.x != k2.x) return k1.x < k2.x;
		if (k1.y != k2.y) return k1.y < k2.y;
		return k1.z < k2.z;
	}
};

struct key_equal {
	__host__ __device__ bool operator()(const uint3& a, const uint3& b) const {
		return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
	}
};


//원본 triangle list는 그대로 놔둔다. ==> 왜냐하면 triTetList는 모든 삼각형에 존재해야하고 unique만 사용해도 재사용되기 때문에
void get_unique_triangles(uint3* originTriangleList,
	size_t triangleSize,
	uint3*& resTri,
	uint2*& lut,
	size_t& resSize) {
	using namespace thrust;

	// indices 0..triangleSize-1
	device_vector<uint> indices(triangleSize);
	sequence(indices.begin(), indices.end());

	// sorted keys per triangle (order-insensitive identifier)
	device_vector<uint3> keys(triangleSize);
	transform(device, indices.begin(), indices.end(), keys.begin(), make_key{ originTriangleList });

	// sort indices by key (stable keeps earlier indices first for equal keys)
	stable_sort(device, indices.begin(), indices.end(), key_less{ raw_pointer_cast(keys.data()) });

	// gather keys in sorted order
	device_vector<uint3> sortedKeys(triangleSize);
	gather(indices.begin(), indices.end(), keys.begin(), sortedKeys.begin());

	// values aligned with sorted indices
	device_vector<unique_val> vals(triangleSize);
	transform(device, indices.begin(), indices.end(), vals.begin(),
		[] __host__ __device__ (uint idx) {
			unique_val v;
			v.firstIdx = idx;
			v.secondIdx = std::numeric_limits<uint>::max();
			v.tri = uint3{ 0,0,0 }; // placeholder, filled below
			return v;
		});

	// fill triangle orientation from first occurrence
	for_each(device, vals.begin(), vals.end(), [originTriangleList] __host__ __device__ (unique_val& v) {
		v.tri = originTriangleList[v.firstIdx];
		});

	// reduce by key to keep first and second indices
	device_vector<uint3> uniqueKeys(triangleSize);
	device_vector<unique_val> uniqueVals(triangleSize);
	auto new_end = reduce_by_key(device,
		sortedKeys.begin(), sortedKeys.end(),
		vals.begin(),
		uniqueKeys.begin(), uniqueVals.begin(),
		key_equal(),
		[] __host__ __device__ (const unique_val& a, const unique_val& b) {
			unique_val out;
			const bool aFirst = (a.firstIdx <= b.firstIdx);
			out.firstIdx = aFirst ? a.firstIdx : b.firstIdx;
			out.tri = aFirst ? a.tri : b.tri;

			uint second = std::numeric_limits<uint>::max();
			auto upd = [&second](uint cand) {
				if (cand < second) second = cand;
			};
			if (a.firstIdx != out.firstIdx) upd(a.firstIdx);
			if (b.firstIdx != out.firstIdx) upd(b.firstIdx);
			if (a.secondIdx != std::numeric_limits<uint>::max()) upd(a.secondIdx);
			if (b.secondIdx != std::numeric_limits<uint>::max()) upd(b.secondIdx);
			out.secondIdx = second;
			return out;
		});

	resSize = new_end.first - uniqueKeys.begin();

	// allocate outputs
	cudaMalloc(&resTri, sizeof(uint3) * resSize);
	cudaMalloc(&lut, sizeof(uint2) * resSize);

	device_ptr<uint3> outTri(resTri);
	device_ptr<uint2> outLut(lut);

	// write results: unique triangle with first occurrence orientation, LUT of first/second indices (1-based)
	transform(device, uniqueVals.begin(), uniqueVals.begin() + resSize, outTri,
		[] __host__ __device__ (const unique_val& v) { return v.tri; });

	transform(device, uniqueVals.begin(), uniqueVals.begin() + resSize, outLut,
		[] __host__ __device__ (const unique_val& v) {
			uint2 r;
			r.x = v.firstIdx + 1; // 1-based index of first appearance
			r.y = (v.secondIdx == std::numeric_limits<uint>::max()) ? 0u : v.secondIdx + 1; // second if exists
			return r;
		});
}

//output size is size+1, with output[0] = 0
void prefixSum(uint* input, uint* output, size_t size) {
    thrust::device_ptr<uint> in(input);
    thrust::device_ptr<uint> out(output);
    
    // inclusive scan: output[i] = input[0] + ... + input[i-1], with output[0] = 0
	thrust::inclusive_scan(thrust::device, input, input + size, output+1); 
}

float sigma_calculate(float3* vlist, uint3* triList, size_t triangleSize) {
	if (triangleSize == 0) {
		return 1.0f;
	}

	std::vector<float3> hVertices;
	std::vector<uint3> hTriangles(triangleSize);

	uint maxVertexIndex = 0;
	cudaMemcpy(hTriangles.data(), triList, sizeof(uint3) * triangleSize, cudaMemcpyDeviceToHost);
	for (const auto& tri : hTriangles) {
		maxVertexIndex = std::max(maxVertexIndex, std::max(tri.x, std::max(tri.y, tri.z)));
	}

	hVertices.resize(static_cast<size_t>(maxVertexIndex) + 1);
	cudaMemcpy(hVertices.data(), vlist, sizeof(float3) * hVertices.size(), cudaMemcpyDeviceToHost);

	double totalEdgeLength = 0.0;
	size_t edgeCount = 0;
	for (const auto& tri : hTriangles) {
		const float3& a = hVertices[tri.x];
		const float3& b = hVertices[tri.y];
		const float3& c = hVertices[tri.z];
		totalEdgeLength += length(a - b);
		totalEdgeLength += length(b - c);
		totalEdgeLength += length(c - a);
		edgeCount += 3;
	}

	if (edgeCount == 0) {
		return 1.0f;
	}

	const float meanEdgeLength = static_cast<float>(totalEdgeLength / static_cast<double>(edgeCount));
	return std::max(meanEdgeLength * 0.5f, 1e-4f);
}
