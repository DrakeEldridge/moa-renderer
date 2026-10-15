// CPU mock of the OptiX device API: lets the real pathtracer.cu run on the CPU for testing.
#pragma once
#include "optix_types.h"
#include <vector_types.h>
#include <vector_functions.h>
#include <cmath>
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
// Implemented in the test harness after pathtracer.cu is included.
void mockTrace(OptixTraversableHandle, float3, float3, float, float, unsigned, unsigned, unsigned, unsigned*);
inline void optixTrace(OptixTraversableHandle h, float3 o, float3 d, float tmin, float tmax, float, OptixVisibilityMask, unsigned flags, unsigned sbtOff, unsigned, unsigned missIdx, unsigned& p0, unsigned& p1){ unsigned p[2]={p0,p1}; mockTrace(h,o,d,tmin,tmax,flags,sbtOff,missIdx,p); p0=p[0]; p1=p[1]; }
inline void optixTrace(OptixTraversableHandle h, float3 o, float3 d, float tmin, float tmax, float, OptixVisibilityMask, unsigned flags, unsigned sbtOff, unsigned, unsigned missIdx, unsigned& p0){ unsigned p[2]={p0,0}; mockTrace(h,o,d,tmin,tmax,flags,sbtOff,missIdx,p); p0=p[0]; }
