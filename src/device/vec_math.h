// MOA Renderer - small float3 maths library shared by host (C++) and device (CUDA) code.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector_types.h>
#include <vector_functions.h>
#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

#if defined(__CUDACC__)
#define MOA_HD __host__ __device__ __forceinline__
#else
#define MOA_HD inline
#endif

namespace moa {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 0.31830988618379067154f;

MOA_HD float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
MOA_HD float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
MOA_HD float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
MOA_HD float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
MOA_HD float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
MOA_HD float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
MOA_HD float3 operator/(float3 a, float s) { const float inv = 1.0f / s; return a * inv; }
MOA_HD float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
MOA_HD float3& operator*=(float3& a, float3 b) { a = a * b; return a; }
MOA_HD float3& operator*=(float3& a, float s) { a = a * s; return a; }
MOA_HD float3& operator/=(float3& a, float s) { a = a / s; return a; }

MOA_HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
MOA_HD float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
MOA_HD float length(float3 a) { return sqrtf(dot(a, a)); }
MOA_HD float3 normalize(float3 a) { return a * (1.0f / sqrtf(dot(a, a))); }
MOA_HD float maxComponent(float3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }
MOA_HD float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
MOA_HD float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }

// Mirror reflection of incoming direction i about normal n (both normalised, n facing i's origin side).
MOA_HD float3 reflect(float3 i, float3 n) { return i - 2.0f * dot(n, i) * n; }

MOA_HD bool isFinite(float3 a)
{
#if defined(__CUDACC__)
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
#else
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
#endif
}

}  // namespace moa
