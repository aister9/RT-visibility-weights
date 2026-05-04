#pragma once
#include "3rdParty/optix7support.h"
#include "geometry/tetrahedron.h"

struct FaceMeshSBTData {
	float3* vert;
	uint3* indice;
	uint2* faceCellInfo; // (cell ID, faceID)
	uint2* triFaceInfo; // (front, back)
};

class LaunchParams {
public:
	OptixTraversableHandle traversable;
};

class RTVisParams : public LaunchParams {
public:
	float3* vertex; uint vSize;
	float3* cam; uint cSize;
	uint2* segment; uint sSize;

	Tetrahedron* dTet;
	cell_info_t* dWeight;

	float sigma;
	float inv2SigmaSq;
};

class RTVisParamsNChunks : public LaunchParams {
public:
	float3* vertex; uint vSize;
	float3* cam; uint cSize;
	uint2* segment; uint sSize;
    
    uint2* segMinMaxIdx;
    float2* segMinMaxDistance;

	Tetrahedron* dTet;
	cell_info_t* dWeight;

	float sigma;
	float inv2SigmaSq;
};