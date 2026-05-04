#include "OptiXGlobalHelper.h"
#include "geometry/tetrahedron.h"
#include "params.h"

extern "C" __constant__ RTVisParams optixLaunchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

struct Payload_t {
    float3 rayTarget;
    float rayTmax;
    float rayTmin;
    int lastHit;
    int firstHit;
};

extern "C" __global__ void __raygen__program__() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int idx = iy * optixGetLaunchDimensions().x + ix;

    if (idx >= optixLaunchParams.sSize) return;

    const uint2 pairIndex = optixLaunchParams.segment[idx];

    float3 rayTarget = optixLaunchParams.vertex[pairIndex.x];
    float3 rayOrigin = optixLaunchParams.cam[pairIndex.y];

    float3 rayDir = normalize(rayTarget - rayOrigin);
    float tMax = length(rayTarget - rayOrigin) + optixLaunchParams.sigma * 3;

    Payload_t payload;
    payload.rayTarget = rayTarget;
    payload.rayTmax = 0;
    payload.rayTmin = 1e8f;
    payload.lastHit = -1;
    payload.firstHit = -1;

    uint32_t u0, u1;
    packPointer(&payload, u0, u1);

    optixTrace(optixLaunchParams.traversable,
        rayOrigin,
        rayDir,
        0.0f,    // tmin
        tMax,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        //OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, //OPTIX_RAY_FLAG_NONE,
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1);

    if (payload.lastHit > 0)
        atomicAdd(&optixLaunchParams.dWeight[payload.lastHit].t, 1.0f);

    if (payload.firstHit > 0)
        optixLaunchParams.dWeight[payload.firstHit].s = (float)(INT_MAX / 8);
}

extern "C" __global__ void __miss__radiance() {
}

extern "C" __global__ void __anyhit__radiance() {
    const FaceMeshSBTData& sbtData
        = *(const FaceMeshSBTData*)optixGetSbtDataPointer();

    Payload_t& prd = *(Payload_t*)getPRD<Payload_t>();

    const int primID = optixGetPrimitiveIndex();
    const float rayT = optixGetRayTmax();

    const float3 rayDir = optixGetWorldRayDirection();
    const float3 rayOrigin = optixGetWorldRayOrigin();

    const float3 surfPos
        = rayOrigin + rayT * rayDir;

    const float3 diff = prd.rayTarget - surfPos;
    const float t = length(diff);

    // float distance = fabsf(length(rayDir - rayOrigin) - rayT);

    const float __w = weightFunc(t, optixLaunchParams.inv2SigmaSq);

    const uint2 triFaceInfo = sbtData.triFaceInfo[primID]-1;

    uint2 cellID;
    auto hitKind = optixGetHitKind();
    if(hitKind == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE)
        cellID = sbtData.faceCellInfo[triFaceInfo.x];
    else
        cellID = sbtData.faceCellInfo[triFaceInfo.y];

    atomicAdd(&optixLaunchParams.dWeight[cellID.x].f[cellID.y], __w);

    if (prd.rayTmin > rayT) {
        prd.rayTmin = rayT;
        prd.firstHit = cellID.x;
    }

    if (prd.rayTmax <= rayT) {
        prd.rayTmax = rayT;
        prd.lastHit = cellID.x;
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __intersection__radiance() {
}

extern "C" __global__ void __closesthit__radiance() {
}