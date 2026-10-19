// MOA Renderer - environment (HDRI dome) lighting: lookup and importance sampling.
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

#include "sampling.cuh"

namespace {

// ---- Environment (HDRI) lighting -------------------------------------------------------------
// Equirectangular mapping: u wraps around the horizon, v runs from straight up (0) to down (1).

__forceinline__ __device__ float2 envDirectionToUv(float3 d)
{
    float u = 0.5f + (atan2f(d.x, -d.z) - params.envRotation) * (0.5f * kInvPi);
    u -= floorf(u);
    const float v = acosf(clampf(d.y, -1.0f, 1.0f)) * kInvPi;
    return make_float2(u, v);
}

__forceinline__ __device__ float3 envUvToDirection(float u, float v)
{
    const float phi = (u - 0.5f) * 2.0f * kPi + params.envRotation;
    const float theta = v * kPi;
    const float st = sinf(theta);
    return make_float3(st * sinf(phi), cosf(theta), -st * cosf(phi));
}

__forceinline__ __device__ float3 envRadiance(float3 d)
{
    const float2 uv = envDirectionToUv(d);
    const float4 c = tex2D<float4>(params.envTexture, uv.x, uv.y);
    return make_float3(c.x, c.y, c.z) * params.envIntensity;
}

// Probability (per unit solid angle) that sampleEnvironment() picks direction d.
__forceinline__ __device__ float envPdf(float3 d)
{
    const float2 uv = envDirectionToUv(d);
    const unsigned int xi = static_cast<unsigned int>(uv.x * params.envWidth);
    const unsigned int yi = static_cast<unsigned int>(uv.y * params.envHeight);
    const unsigned int x = xi < params.envWidth ? xi : params.envWidth - 1;
    const unsigned int y = yi < params.envHeight ? yi : params.envHeight - 1;
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - d.y * d.y));
    if (sinTheta <= 1e-6f || params.envFuncIntegral <= 0.0f)
        return 0.0f;
    const float pdfUv = params.envFunc[y * params.envWidth + x] / params.envFuncIntegral;
    return pdfUv / (2.0f * kPi * kPi * sinTheta);
}

// Picks a direction towards the environment, favouring bright areas (sun, windows, softboxes).
__forceinline__ __device__ float3 sampleEnvironment(float r1, float r2, float& pdf)
{
    const unsigned int W = params.envWidth, H = params.envHeight;
    const unsigned int y = findInterval(params.envMarginalCdf, H, r1);
    const float m0 = params.envMarginalCdf[y], m1 = params.envMarginalCdf[y + 1];
    const float dv = m1 > m0 ? (r1 - m0) / (m1 - m0) : 0.5f;
    const float* row = params.envConditionalCdf + static_cast<size_t>(y) * (W + 1);
    const unsigned int x = findInterval(row, W, r2);
    const float c0 = row[x], c1 = row[x + 1];
    const float du = c1 > c0 ? (r2 - c0) / (c1 - c0) : 0.5f;
    const float u = (x + du) / W, v = (y + dv) / H;
    const float3 d = envUvToDirection(u, v);
    const float sinTheta = sinf(v * kPi);
    pdf = sinTheta > 1e-6f ? (params.envFunc[y * W + x] / params.envFuncIntegral) / (2.0f * kPi * kPi * sinTheta) : 0.0f;
    return d;
}

}  // namespace
