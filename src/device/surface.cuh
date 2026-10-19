// MOA Renderer - surface: the material at a hit point with all its texture maps applied, plus the
// normals the integrator needs. New map types (displacement-normals, AO, emission maps...) go here.
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

#include "rays.cuh"
#include "textures.cuh"

namespace {

struct SurfacePoint
{
    Material m;      // material with roughness / metalness maps applied
    float3 albedo;   // base colour with the colour map applied
    float3 ng;       // true surface side, facing the incoming ray (offsets new rays off the surface)
    float3 n;        // shading normal (smooth + normal map), on the same side as ng
    bool frontFace;  // the ray hit the side the mesh normal points to
};

__forceinline__ __device__ SurfacePoint resolveSurface(const HitInfo& hit, float3 direction)
{
    SurfacePoint sp;
    sp.m = params.materials[hit.materialId];
    Material& m = sp.m;
    // Roughness / metalness maps replace the slider values (as a connected input does in
    // MaterialX). Greyscale, linear: the red channel is used.
    if (m.roughnessTexture >= 0)
        m.roughness = sampleTexture(m.roughnessTexture, hit.uv, m.flags).x;
    if (m.metallicTexture >= 0)
        m.metallic = sampleTexture(m.metallicTexture, hit.uv, m.flags).x;
    sp.frontFace = dot(direction, hit.normal) < 0.0f;
    sp.ng = sp.frontFace ? hit.normal : -hit.normal;
    const float3 mappedNormal = m.normalTexture >= 0 ? applyNormalMap(m, hit) : hit.shadingNormal;
    sp.n = sp.frontFace ? mappedNormal : -mappedNormal;

    sp.albedo = m.albedo;
    if (m.colorTexture >= 0)
    {
        const float4 c = sampleTexture(m.colorTexture, hit.uv, m.flags);
        sp.albedo = sp.albedo * make_float3(c.x, c.y, c.z);
    }
    return sp;
}

}  // namespace
