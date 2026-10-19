// MOA Renderer - lights: picking points on light sources for direct lighting.
// (More light types - sphere, disk, distant, many-light sampling - go here.)
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

#include "sampling.cuh"

namespace {

// A uniformly random point on a rectangular area light (pdf = 1 / area). Uses two random numbers.
__forceinline__ __device__ float3 sampleQuadLightPoint(const QuadLight& light, unsigned int& seed)
{
    return light.corner + light.edge1 * rnd(seed) + light.edge2 * rnd(seed);
}

// Direct lighting from one extra light (LightSource) as seen from point p.
// Returns false if the light can't reach p. Otherwise:
//   wi    = direction towards the light,  dist = shadow ray length,
//   value = what to multiply by (BSDF x cos(surface)) to get the light's contribution
//           (emitted light x geometry / sampling pdf).
__forceinline__ __device__ bool sampleLightSource(const LightSource& L, float3 p, unsigned int& seed, float3& wi,
                                                  float& dist, float3& value)
{
    if (L.type == LIGHT_QUAD || L.type == LIGHT_DISK)
    {
        float3 q;
        const float r1 = rnd(seed), r2 = rnd(seed);
        if (L.type == LIGHT_QUAD)
            q = L.position + L.axisU * r1 + L.axisV * r2;
        else
        {
            const float r = sqrtf(r1), phi = 2.0f * kPi * r2;
            q = L.position + L.axisU * (r * cosf(phi)) + L.axisV * (r * sinf(phi));
        }
        const float3 d = q - p;
        const float dist2 = dot(d, d);
        if (dist2 <= 1e-12f)
            return false;
        dist = sqrtf(dist2);
        wi = d / dist;
        const float cosLight = -dot(L.normal, wi);
        if (cosLight <= 0.0f)
            return false;
        value = L.emission * (cosLight * L.area / dist2);
        return true;
    }
    if (L.type == LIGHT_SPHERE)
    {
        // Sample the cone of directions the sphere covers (uniform in solid angle).
        const float3 d = L.position - p;
        const float dist2 = dot(d, d);
        const float r2 = L.radius * L.radius;
        if (dist2 <= r2)
            return false;  // inside the light
        const float cosMax = sqrtf(fmaxf(0.0f, 1.0f - r2 / dist2));
        const float cosT = 1.0f - rnd(seed) * (1.0f - cosMax);
        const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
        const float phi = 2.0f * kPi * rnd(seed);
        const Frame f = makeFrame(d / sqrtf(dist2));
        wi = f.toWorld(make_float3(sinT * cosf(phi), sinT * sinf(phi), cosT));
        const float b = dot(wi, d);
        dist = b - sqrtf(fmaxf(0.0f, b * b - (dist2 - r2)));  // distance to the sphere's surface
        value = L.emission * (2.0f * kPi * (1.0f - cosMax));
        return dist > 0.0f;
    }
    if (L.type == LIGHT_POINT)
    {
        const float3 d = L.position - p;
        const float dist2 = dot(d, d);
        if (dist2 <= 1e-12f)
            return false;
        dist = sqrtf(dist2);
        wi = d / dist;
        value = L.emission * (1.0f / dist2);
        return true;
    }
    // LIGHT_DISTANT: a sun. Soft shadows come from its angular size.
    if (L.cosAngle >= 0.9999999f)
        wi = L.normal;
    else
    {
        const float cosT = 1.0f - rnd(seed) * (1.0f - L.cosAngle);
        const float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT * cosT));
        const float phi = 2.0f * kPi * rnd(seed);
        wi = makeFrame(L.normal).toWorld(make_float3(sinT * cosf(phi), sinT * sinf(phi), cosT));
    }
    dist = 1e16f;
    value = L.emission;
    return true;
}

}  // namespace
