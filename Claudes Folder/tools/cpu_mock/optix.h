// CPU mock of the OptiX device API: lets the real pathtracer.cu run on the CPU for testing.
#pragma once
#include "optix_types.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <texture_types.h>
#include <cmath>
#include <vector>
#define __global__
#define __device__
#define __constant__
#define __forceinline__ inline
typedef unsigned int OptixVisibilityMask;
enum { OPTIX_RAY_FLAG_NONE=0, OPTIX_RAY_FLAG_DISABLE_ANYHIT=1u<<0, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT=1u<<2, OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT=1u<<3 };
struct MockState { uint3 launchIdx; unsigned int* payload; unsigned prim; float tmax; float3 o, d; float2 bary; };
extern thread_local MockState g_mock;
inline uint3 optixGetLaunchIndex(){ return g_mock.launchIdx; }
inline unsigned int optixGetPayload_0(){ return g_mock.payload[0]; }
inline unsigned int optixGetPayload_1(){ return g_mock.payload[1]; }
inline void optixSetPayload_0(unsigned v){ g_mock.payload[0]=v; }
inline unsigned int optixGetPrimitiveIndex(){ return g_mock.prim; }
inline float optixGetRayTmax(){ return g_mock.tmax; }
inline float2 optixGetTriangleBarycentrics(){ return g_mock.bary; }
inline float3 optixGetWorldRayOrigin(){ return g_mock.o; }
inline float3 optixGetWorldRayDirection(){ return g_mock.d; }
// Textures: cudaTextureObject_t is an index into this list (filled by the harness).
struct MockTexture { int w, h; const unsigned char* rgba; bool srgb; const float* rgbaf = nullptr; };
extern std::vector<MockTexture> g_mockTextures;
template <class T> T tex2D(cudaTextureObject_t, float, float);
template <> inline float4 tex2D<float4>(cudaTextureObject_t t, float u, float v)
{
    const MockTexture& m = g_mockTextures[t];
    u -= std::floor(u); v -= std::floor(v);  // wrap
    int x = (int)(u * m.w); if (x >= m.w) x = m.w - 1;
    int y = (int)(v * m.h); if (y >= m.h) y = m.h - 1;
    if (m.rgbaf) { const float* f = m.rgbaf + ((size_t)y * m.w + x) * 4; return make_float4(f[0], f[1], f[2], f[3]); }
    const unsigned char* p = m.rgba + ((size_t)y * m.w + x) * 4;
    float c[4];
    for (int i = 0; i < 4; ++i) {
        float f = p[i] / 255.0f;
        if (m.srgb && i < 3) f = f <= 0.04045f ? f / 12.92f : std::pow((f + 0.055f) / 1.055f, 2.4f);
        c[i] = f;
    }
    return make_float4(c[0], c[1], c[2], c[3]);
}

// Implemented in the test harness after pathtracer.cu is included.
void mockTrace(OptixTraversableHandle, float3, float3, float, float, unsigned, unsigned, unsigned, unsigned*);
inline void optixTrace(OptixTraversableHandle h, float3 o, float3 d, float tmin, float tmax, float, OptixVisibilityMask, unsigned flags, unsigned sbtOff, unsigned, unsigned missIdx, unsigned& p0, unsigned& p1){ unsigned p[2]={p0,p1}; mockTrace(h,o,d,tmin,tmax,flags,sbtOff,missIdx,p); p0=p[0]; p1=p[1]; }
inline void optixTrace(OptixTraversableHandle h, float3 o, float3 d, float tmin, float tmax, float, OptixVisibilityMask, unsigned flags, unsigned sbtOff, unsigned, unsigned missIdx, unsigned& p0){ unsigned p[2]={p0,0}; mockTrace(h,o,d,tmin,tmax,flags,sbtOff,missIdx,p); p0=p[0]; }
