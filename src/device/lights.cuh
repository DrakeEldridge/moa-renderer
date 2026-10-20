// MOA Renderer - lights: picking points on light sources for direct lighting, spot cones, and
// rays that hit a light's shape (lights seen in mirrors, glass and by the camera).
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

// Spot cone: how much of the light leaves in direction `out` (unit, from the light). 1 = no cone.
__forceinline__ __device__ float spotFactor(const LightSource& L, float3 out)
{
    if (!(L.flags & LIGHTFLAG_SPOT))
        return 1.0f;
    const float c = dot(L.spotAxis, out);
    if (c <= L.spotCosOuter)
        return 0.0f;
    float f = 1.0f;
    if (c < L.spotCosInner)
    {
        const float t = (c - L.spotCosOuter) / fmaxf(L.spotCosInner - L.spotCosOuter, 1e-6f);
        f = t * t * (3.0f - 2.0f * t);  // smoothstep: a soft edge
    }
    if (L.spotFocus > 0.0f)
        f *= powf(fmaxf(c, 0.0f), L.spotFocus);
    return f;
}

// Direct lighting from one extra light (LightSource) as seen from point p.
// Returns false if the light can't reach p. Otherwise:
//   wi    = direction towards the light,  dist = shadow ray length,
//   value = what to multiply by (BSDF x cos(surface)) to get the light's contribution
//           (emitted light x geometry / sampling pdf).
__forceinline__ __device__ bool sampleLightShape(const LightSource& L, float3 p, unsigned int& seed, float3& wi,
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
    if (L.type == LIGHT_CYLINDER)
    {
        // A uniformly random point on the tube's side (the ends don't emit).
        const float t = 2.0f * rnd(seed) - 1.0f;
        const float phi = 2.0f * kPi * rnd(seed);
        const float len = length(L.axisU);
        if (len <= 0.0f)
            return false;
        const float3 axis = L.axisU / len;
        const float3 e1 = L.axisV;
        const float3 e2 = cross(axis, e1);
        const float3 outward = e1 * cosf(phi) + e2 * sinf(phi);
        const float3 q = L.position + L.axisU * t + outward * L.radius;
        const float3 d = q - p;
        const float dist2 = dot(d, d);
        if (dist2 <= 1e-12f)
            return false;
        dist = sqrtf(dist2);
        wi = d / dist;
        const float cosLight = -dot(outward, wi);
        if (cosLight <= 0.0f)
            return false;  // that point faces away from us
        value = L.emission * (cosLight * L.area / dist2);
        return true;
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

__forceinline__ __device__ bool sampleLightSource(const LightSource& L, float3 p, unsigned int& seed, float3& wi,
                                                  float& dist, float3& value)
{
    if (!sampleLightShape(L, p, seed, wi, dist, value))
        return false;
    if (L.flags & LIGHTFLAG_SPOT)
    {
        const float f = spotFactor(L, -wi);
        if (f <= 0.0f)
            return false;
        value = value * f;
    }
    return true;
}

// Does a ray hit the shape of an extra light before `tmax`? (Lights seen directly, in mirrors and
// through glass.) cameraRay: only lights marked visible to the camera count. On a hit, `le` is the
// radiance seen along the ray. Point lights have no shape; suns are found when tmax is infinite and
// allowSun is set (only after sharp bounces: a tiny, very bright disc would be a firefly otherwise).
__forceinline__ __device__ bool hitVisibleLight(float3 o, float3 d, float tmax, bool cameraRay, bool allowSun,
                                                float3& le)
{
    const unsigned int need = cameraRay ? LIGHTFLAG_VISIBLE_CAMERA : LIGHTFLAG_VISIBLE_REFLECT;
    float best = tmax;
    int bestIndex = -1;
    for (unsigned int i = 0; i < params.lightCount; ++i)
    {
        const LightSource& L = params.lights[i];
        if (!(L.flags & need))
            continue;
        float t = -1.0f;
        if (L.type == LIGHT_QUAD || L.type == LIGHT_DISK)
        {
            const float denom = dot(d, L.normal);
            if (denom >= -1e-8f)
                continue;  // they glow from the front only
            t = dot(L.position - o, L.normal) / denom;  // position lies on the light's plane
            if (t <= kRayEpsilon || t >= best)
                continue;
            const float3 q = o + d * t - L.position;
            const float u = dot(q, L.axisU) / fmaxf(dot(L.axisU, L.axisU), 1e-20f);
            const float v = dot(q, L.axisV) / fmaxf(dot(L.axisV, L.axisV), 1e-20f);
            const bool inside = L.type == LIGHT_QUAD ? (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f)
                                                     : (u * u + v * v <= 1.0f);
            if (!inside)
                continue;
        }
        else if (L.type == LIGHT_SPHERE)
        {
            const float3 oc = o - L.position;
            const float b = dot(oc, d);
            const float c = dot(oc, oc) - L.radius * L.radius;
            const float disc = b * b - c;
            if (disc < 0.0f)
                continue;
            const float sq = sqrtf(disc);
            t = -b - sq;
            if (t <= kRayEpsilon)
                continue;  // starts inside (or behind): not seen from outside
        }
        else if (L.type == LIGHT_CYLINDER)
        {
            const float len = length(L.axisU);
            if (len <= 0.0f)
                continue;
            const float3 a = L.axisU / len;
            const float3 oc = o - L.position;
            const float3 dp = d - a * dot(d, a);
            const float3 op = oc - a * dot(oc, a);
            const float A = dot(dp, dp);
            if (A < 1e-12f)
                continue;
            const float B = dot(dp, op);
            const float C = dot(op, op) - L.radius * L.radius;
            const float disc = B * B - A * C;
            if (disc < 0.0f)
                continue;
            t = (-B - sqrtf(disc)) / A;  // outside of the tube only (it glows outwards)
            if (t <= kRayEpsilon)
                continue;
            if (fabsf(dot(oc + d * t, a)) > len)
                continue;  // past the ends
        }
        else
            continue;  // points and suns: below / no shape
        if (t > kRayEpsilon && t < best)
        {
            best = t;
            bestIndex = static_cast<int>(i);
        }
    }
    if (bestIndex >= 0)
    {
        const LightSource& L = params.lights[bestIndex];
        le = L.emission * spotFactor(L, -d);
        return true;
    }
    if (tmax < 1e15f || !allowSun)
        return false;
    // Nothing else in the way: a sun disc?
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    bool any = false;
    for (unsigned int i = 0; i < params.lightCount; ++i)
    {
        const LightSource& L = params.lights[i];
        if (L.type != LIGHT_DISTANT || !(L.flags & need) || L.cosAngle >= 0.9999999f)
            continue;
        if (dot(d, L.normal) >= L.cosAngle)
        {
            // Irradiance spread over the disc's solid angle = the radiance of the sun's face.
            sum = sum + L.emission * (1.0f / (2.0f * kPi * (1.0f - L.cosAngle)));
            any = true;
        }
    }
    le = sum;
    return any;
}

}  // namespace
