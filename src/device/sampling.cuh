// MOA Renderer - sampling helpers: directions, local frames, MIS weights, CDF search.
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

namespace {

// Builds an orthonormal basis around n and returns a cosine-weighted direction.
__forceinline__ __device__ float3 sampleCosineHemisphere(float3 n, float u1, float u2)
{
    const float r = sqrtf(u1);
    const float phi = 2.0f * kPi * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));

    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    const float3 t = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    const float3 bt = make_float3(b, sign + n.y * n.y * a, -n.y);
    return normalize(t * x + bt * y + n * z);
}

struct Frame
{
    float3 t, b, n;
    __forceinline__ __device__ float3 toLocal(float3 v) const { return make_float3(dot(v, t), dot(v, b), dot(v, n)); }
    __forceinline__ __device__ float3 toWorld(float3 v) const { return t * v.x + b * v.y + n * v.z; }
};

__forceinline__ __device__ Frame makeFrame(float3 n)
{
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    Frame f;
    f.t = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    f.b = make_float3(b, sign + n.y * n.y * a, -n.y);
    f.n = n;
    return f;
}

__forceinline__ __device__ float luminance(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

// Power heuristic for combining two sampling strategies (multiple importance sampling).
__forceinline__ __device__ float misWeight(float pdfA, float pdfB)
{
    const float a = pdfA * pdfA, b = pdfB * pdfB;
    return a + b > 0.0f ? a / (a + b) : 0.0f;
}

// Finds the last index i in cdf[0..count] with cdf[i] <= value (cdf is increasing, cdf[0] = 0).
__forceinline__ __device__ unsigned int findInterval(const float* cdf, unsigned int count, float value)
{
    unsigned int lo = 0, hi = count;  // answer in [lo, hi)
    while (hi - lo > 1)
    {
        const unsigned int mid = (lo + hi) / 2;
        if (cdf[mid] <= value)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

}  // namespace
