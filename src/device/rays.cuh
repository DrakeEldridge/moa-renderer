// MOA Renderer - rays: what a ray hit, and the two ray types (camera/bounce rays and shadow rays).
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

namespace {

constexpr float kRayEpsilon = 1e-3f;  // offset to avoid a surface re-hitting itself
constexpr float kRayTMax = 1e16f;

// Per-ray result written by the closest-hit / miss programs.
struct HitInfo
{
    float3 position;
    float3 normal;         // geometric normal, as wound in the mesh (not yet flipped to face the ray)
    float3 shadingNormal;  // smooth interpolated normal, on the same side as `normal`
    float4 tangent;        // interpolated tangent (xyz) and bitangent sign (w)
    float2 uv;
    unsigned int materialId;
    unsigned int hit;
};

// The radiance ray carries a pointer to a HitInfo split across two 32-bit payload slots.
__forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long p = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(p >> 32);
    i1 = static_cast<unsigned int>(p & 0xffffffffull);
}

__forceinline__ __device__ HitInfo* getHitInfo()
{
    const unsigned long long p =
        (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
    return reinterpret_cast<HitInfo*>(p);
}

__forceinline__ __device__ void traceRadiance(float3 origin, float3 direction, HitInfo& hit, float tmax = kRayTMax)
{
    unsigned int p0, p1;
    packPointer(&hit, p0, p1);
    optixTrace(params.handle, origin, direction,
               0.0f, tmax, 0.0f,  // tmin, tmax, ray time
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,  // SBT offset, stride, miss index
               p0, p1);
}

// Returns true if nothing blocks the segment from origin along direction up to distance.
__forceinline__ __device__ bool traceShadow(float3 origin, float3 direction, float distance)
{
    unsigned int visible = 0u;  // the shadow miss program sets this to 1
    optixTrace(params.handle, origin, direction,
               0.0f, distance, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW,
               visible);
    return visible != 0u;
}

}  // namespace
