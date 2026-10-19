// MOA Renderer - materials (BSDFs): how light reflects off / passes through a surface.
// Standard material = Lambert diffuse under a GGX microfacet reflection layer, with metalness and
// a (rough) dielectric glass mode. To change how a material looks, this is the file.
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

#include "sampling.cuh"

namespace {

// Unpolarised Fresnel reflectance for a smooth dielectric. eta = n_incident / n_transmitted.
// Returns 1 for total internal reflection.
__forceinline__ __device__ float fresnelDielectric(float cosI, float eta)
{
    const float sin2T = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    if (sin2T >= 1.0f)
        return 1.0f;
    const float cosT = sqrtf(1.0f - sin2T);
    const float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
    const float rp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5f * (rs * rs + rp * rp);
}

__forceinline__ __device__ float3 refractDir(float3 i, float3 n, float eta, float cosI)
{
    const float sin2T = eta * eta * fmaxf(0.0f, 1.0f - cosI * cosI);
    const float cosT = sqrtf(fmaxf(0.0f, 1.0f - sin2T));
    return normalize(i * eta + n * (eta * cosI - cosT));
}

// ---------------------------------------------------------------------------------------
// Standard material: Lambert diffuse under a GGX microfacet reflection layer, with
// metalness (the base colour tints the reflection and there is no diffuse) and an optional
// rough dielectric (glass) mode. Microfacet maths: Walter et al. 2007 (GGX), Heitz 2014
// (Smith masking), Heitz 2018 (sampling the visible normals).
// ---------------------------------------------------------------------------------------

__forceinline__ __device__ float ggxD(float a2, float cosH)
{
    const float d = cosH * cosH * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d);
}

__forceinline__ __device__ float smithLambda(float a2, float cosV)
{
    const float c2 = fmaxf(cosV * cosV, 1e-8f);
    return 0.5f * (-1.0f + sqrtf(1.0f + a2 * (1.0f - c2) / c2));
}

// Samples a microfacet normal among those visible from wo (local frame, wo.z > 0).
__forceinline__ __device__ float3 sampleGgxVisibleNormal(float3 wo, float alpha, float u1, float u2)
{
    const float3 vh = normalize(make_float3(alpha * wo.x, alpha * wo.y, wo.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 = lensq > 0.0f ? make_float3(-vh.y, vh.x, 0.0f) * (1.0f / sqrtf(lensq)) : make_float3(1.0f, 0.0f, 0.0f);
    const float3 t2 = cross(vh, t1);
    const float r = sqrtf(u1);
    const float phi = 2.0f * kPi * u2;
    const float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1)) + s * p2;
    const float3 nh = t1 * p1 + t2 * p2 + vh * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2));
    return normalize(make_float3(alpha * nh.x, alpha * nh.y, fmaxf(1e-6f, nh.z)));
}

__forceinline__ __device__ float3 schlick(float3 f0, float cosTheta)
{
    const float m = clampf(1.0f - cosTheta, 0.0f, 1.0f);
    const float m5 = m * m * m * m * m;
    return f0 + (make_float3(1.0f, 1.0f, 1.0f) - f0) * m5;
}

// Everything the opaque standard material needs at one hit point.
struct StandardBsdf
{
    float3 base;
    float metallic;
    float specular;
    float ior;
    float alpha;       // GGX roughness (roughness squared)
    bool deltaSpecular;  // roughness ~ 0: a perfect mirror layer
    float specProb;    // probability of sampling the reflection layer (from wo)

    __forceinline__ __device__ float dielectricF(float cosTheta) const
    {
        return specular * fresnelDielectric(cosTheta, 1.0f / ior);
    }
    __forceinline__ __device__ float3 reflectionColor(float cosTheta) const
    {
        const float fd = dielectricF(cosTheta);
        return make_float3(fd, fd, fd) * (1.0f - metallic) + schlick(base, cosTheta) * metallic;
    }
    __forceinline__ __device__ float3 diffuseColor(float cosWo) const
    {
        // Light reflected by the top layer doesn't reach the diffuse base.
        return base * ((1.0f - metallic) * (1.0f - dielectricF(cosWo)) * kInvPi);
    }
};

__forceinline__ __device__ StandardBsdf makeStandardBsdf(const Material& m, float3 albedo, float cosWo)
{
    StandardBsdf s;
    s.base = albedo;
    s.metallic = clampf(m.metallic, 0.0f, 1.0f);
    s.specular = fmaxf(m.specular, 0.0f);
    s.ior = fmaxf(m.ior, 1.0001f);
    const float rough = clampf(m.roughness, 0.0f, 1.0f);
    s.deltaSpecular = rough < 0.02f;
    s.alpha = fmaxf(rough * rough, 1e-4f);
    // Pick the reflection layer roughly in proportion to how much light it carries, but often
    // enough that glossy reflections clean up.
    const float wSpec = luminance(s.reflectionColor(cosWo));
    const float wDiff = luminance(s.diffuseColor(cosWo)) * kPi;
    const float total = wSpec + wDiff;
    s.specProb = total > 0.0f ? clampf(wSpec / total, 0.0f, 1.0f) : 0.0f;
    if (wSpec > 0.0f && wDiff > 0.0f)
        s.specProb = clampf(s.specProb, 0.25f, 0.9f);
    return s;
}

// BSDF value (without the cosine) and the pdf of choosing wi by sampleStandard(). Local frame,
// wo.z > 0. The perfect-mirror layer (deltaSpecular) can't be evaluated and is left out.
__forceinline__ __device__ float3 evalStandard(const StandardBsdf& s, float3 wo, float3 wi, float& pdf)
{
    pdf = 0.0f;
    if (wi.z <= 0.0f || wo.z <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    float3 f = s.diffuseColor(wo.z);
    pdf = (1.0f - s.specProb) * wi.z * kInvPi;
    if (!s.deltaSpecular && s.specProb > 0.0f)
    {
        const float3 h = normalize(wo + wi);
        const float a2 = s.alpha * s.alpha;
        const float d = ggxD(a2, h.z);
        const float lo = smithLambda(a2, wo.z);
        const float g2 = 1.0f / (1.0f + lo + smithLambda(a2, wi.z));
        const float voh = fmaxf(dot(wo, h), 1e-6f);
        f += s.reflectionColor(voh) * (d * g2 / (4.0f * wo.z * wi.z));
        // Visible-normal sampling: pdf(wi) = G1(wo) D / (4 wo.z).
        pdf += s.specProb * d / ((1.0f + lo) * 4.0f * wo.z);
    }
    return f;
}

// Samples a bounce. Returns the weight f*cos/pdf; delta = a perfect-mirror bounce.
__forceinline__ __device__ float3 sampleStandard(const StandardBsdf& s, float3 wo, unsigned int& seed, float3& wi,
                                                 float& pdf, bool& delta)
{
    delta = false;
    if (rnd(seed) < s.specProb)
    {
        if (s.deltaSpecular)
        {
            wi = make_float3(-wo.x, -wo.y, wo.z);
            delta = true;
            pdf = 0.0f;
            return s.reflectionColor(wo.z) * (1.0f / s.specProb);
        }
        const float3 h = sampleGgxVisibleNormal(wo, s.alpha, rnd(seed), rnd(seed));
        wi = reflect(-wo, h);
    }
    else
    {
        const float r = sqrtf(rnd(seed));
        const float phi = 2.0f * kPi * rnd(seed);
        wi = make_float3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - r * r)));
    }
    if (wi.z <= 0.0f)
        return make_float3(0.0f, 0.0f, 0.0f);
    const float3 f = evalStandard(s, wo, wi, pdf);
    return pdf > 0.0f ? f * (wi.z / pdf) : make_float3(0.0f, 0.0f, 0.0f);
}

}  // namespace
