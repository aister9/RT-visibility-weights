#pragma once
#include <optix_device.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "3rdParty/helper_math.h"

#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif

//unpacking
static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

//packing
static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

__forceinline__ __device__ float3 reflect(const float3& i, const float3& n) {
    return i - 2.0f * n * dot(n, i);
}

__forceinline__ __device__ float3 faceforward(const float3& n, const float3& i, const float3& nref)
{
    return n * copysignf(1.0f, dot(i, nref));
}

__forceinline__ __device__ float3 toSRGB(const float3& c)
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)

{
    x = clamp(x, 0.0f, 1.0f);
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color(const float3& c)
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB(clamp(c, 0.0f, 1.0f));
    return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}
__forceinline__ __device__ uchar4 make_color(const float4& c)
{
    return make_color(make_float3(c.x, c.y, c.z));
}

struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

__forceinline__ __device__ float luminance(const float3& rgb)
{
    const float3 ntsc_luminance = { 0.30f, 0.59f, 0.11f };
    return dot(rgb, ntsc_luminance);
}


__forceinline__ __device__ float fresnel_schlick(const float cos_theta, const float exponent = 5.0f,
    const float minimum = 0.0f, const float maximum = 1.0f)
{
    /**
      Clamp the result of the arithmetic due to floating point precision:
      the result should lie strictly within [minimum, maximum]
      return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta, exponent),
                   minimum, maximum);
    */

    /** The max doesn't seem like it should be necessary, but without it you get
        annoying broken pixels at the center of reflective spheres where cos_theta ~ 1.
    */
    return clamp(minimum + (maximum - minimum) * powf(fmaxf(0.0f, 1.0f - cos_theta), exponent),
        minimum, maximum);
}

//NVIDIA SMAPLE Fresnel
__forceinline__ __device__ float3 fresnel_schlick(const float cos_theta, const float exponent,
    const float3& minimum, const float3& maximum)
{
    return make_float3(fresnel_schlick(cos_theta, exponent, minimum.x, maximum.x),
        fresnel_schlick(cos_theta, exponent, minimum.y, maximum.y),
        fresnel_schlick(cos_theta, exponent, minimum.z, maximum.z));
}

__forceinline__ __device__ bool refract(float3& r, const float3& i, const float3& n, const float ior) {
    float3 nn = n;
    float negNdotV = dot(i, nn);
    float eta;

    if (negNdotV > 0.0f)
    {
        eta = ior;
        nn = -1*n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

    if (k < 0.0f)
    {
        // Initialize this value, so that r always leaves this function initialized.
        r = make_float3(0.f);
        return false;
    }
    else
    {
        r = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
        return true;
    }
}

__forceinline__ __device__ float3 exp(const float3& x)
{
    return make_float3(exp(x.x), exp(x.y), exp(x.z));
}