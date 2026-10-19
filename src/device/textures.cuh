// MOA Renderer - textures: colour, normal, roughness and metalness maps at a hit point.
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

#include "rays.cuh"

namespace {

__forceinline__ __device__ float4 sampleTexture(int index, float2 uv, unsigned int flags)
{
    const float v = (flags & MATFLAG_FLIP_V) ? uv.y : 1.0f - uv.y;  // image row 0 is the top
    return tex2D<float4>(params.textures[index], uv.x, v);
}

// Bends the smooth normal using the normal map (tangent space -> world space).
__forceinline__ __device__ float3 applyNormalMap(const Material& m, const HitInfo& hit)
{
    const float3 N = hit.shadingNormal;
    const float4 t = sampleTexture(m.normalTexture, hit.uv, m.flags);
    float3 tn = make_float3(t.x * 2.0f - 1.0f, t.y * 2.0f - 1.0f, t.z * 2.0f - 1.0f);
    if (m.flags & MATFLAG_FLIP_NORMAL_GREEN)
        tn.y = -tn.y;
    tn.x *= m.normalStrength;
    tn.y *= m.normalStrength;
    float3 T = make_float3(hit.tangent.x, hit.tangent.y, hit.tangent.z);
    T = T - N * dot(N, T);
    const float tl = length(T);
    if (tl < 1e-6f)
        return N;
    T = T / tl;
    const float3 B = cross(N, T) * hit.tangent.w;
    const float3 mapped = normalize(T * tn.x + B * tn.y + N * fmaxf(tn.z, 1e-3f));
    // Never let the bent normal face away from the real surface (it causes black spots).
    return dot(mapped, hit.normal) > 0.05f ? mapped : N;
}

}  // namespace
