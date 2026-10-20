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

// Coverage / cut-outs: does this ray go through the surface here (a hole in a mesh guard, the
// see-through part of a visor)? Soft coverage lets a random share of rays through.
__forceinline__ __device__ bool passesThrough(const HitInfo& hit, unsigned int& seed)
{
    const Material& m = params.materials[hit.materialId];
    float opacity = 1.0f - m.transparency;
    if (m.opacityTexture >= 0)
    {
        const float4 t = sampleTexture(m.opacityTexture, hit.uv, m.flags);
        opacity *= m.opacityChannel == 3 ? t.w : t.x;
    }
    if (opacity >= 1.0f)
        return false;
    if (m.opacityThreshold > 0.0f)
        return opacity < m.opacityThreshold;
    return rnd(seed) >= opacity;
}

// Shadow rays: like passesThrough, and thin glass sheets (a visor, a window) let a random share
// through, as much as they transmit. (The shadow isn't tinted by the glass colour yet.)
__forceinline__ __device__ bool shadowPassesThrough(const HitInfo& hit, float3 direction, unsigned int& seed)
{
    if (passesThrough(hit, seed))
        return true;
    const Material& m = params.materials[hit.materialId];
    if (!(m.flags & MATFLAG_THIN_WALLED) || m.type != MAT_STANDARD || m.transmission <= 0.5f)
        return false;
    float3 tint = m.albedo;
    if (m.colorTexture >= 0)
    {
        const float4 c = sampleTexture(m.colorTexture, hit.uv, m.flags);
        tint = tint * make_float3(c.x, c.y, c.z);
    }
    const float cosI = fabsf(dot(direction, hit.normal));
    const float r = fresnelDielectric(cosI, 1.0f / fmaxf(m.ior, 1.0001f));
    const float reflect = r < 1.0f ? 2.0f * r / (1.0f + r) : 1.0f;
    return rnd(seed) < (1.0f - reflect) * clampf(luminance(tint), 0.0f, 1.0f);
}

// Traces a camera / bounce ray, stepping through holes and see-through parts.
__forceinline__ __device__ void traceThroughCutouts(float3 origin, float3 direction, HitInfo& hit, unsigned int& seed)
{
    traceRadiance(origin, direction, hit);
    if (!params.hasCutouts)
        return;
    for (int i = 0; i < 64 && hit.hit && passesThrough(hit, seed); ++i)
    {
        const float3 facing = dot(direction, hit.normal) < 0.0f ? hit.normal : -hit.normal;
        origin = hit.position - facing * kRayEpsilon;
        hit.hit = 0u;
        traceRadiance(origin, direction, hit);
    }
}

// Shadow test that lets light through holes and see-through parts.
__forceinline__ __device__ bool visibleThroughCutouts(float3 origin, float3 direction, float distance,
                                                      unsigned int& seed)
{
    if (!params.hasCutouts)
        return traceShadow(origin, direction, distance);
    for (int i = 0; i < 64; ++i)
    {
        HitInfo h;
        h.hit = 0u;
        traceRadiance(origin, direction, h, distance);
        if (!h.hit)
            return true;
        if (!shadowPassesThrough(h, direction, seed))
            return false;
        const float travelled = length(h.position - origin);
        const float3 facing = dot(direction, h.normal) < 0.0f ? h.normal : -h.normal;
        origin = h.position - facing * kRayEpsilon;
        distance -= travelled + kRayEpsilon;
        if (distance <= 0.0f)
            return true;
    }
    return false;
}

// Shadow test from `origin` to a point on a light.
__forceinline__ __device__ bool visibleToPoint(float3 origin, float3 target, unsigned int& seed)
{
    const float3 d = target - origin;
    const float len = length(d);
    if (len <= 2.0f * kRayEpsilon)
        return true;
    return visibleThroughCutouts(origin, d / len, len - 2.0f * kRayEpsilon, seed);
}

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
