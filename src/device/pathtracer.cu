// MOA Renderer - GPU programs for the path tracer (compiled to PTX and loaded by OptiX).
//
// Structure: the ray-generation program runs the whole path loop for one pixel. It
// traces a ray, the closest-hit program reports what was hit, and the loop decides how
// the light path continues. Direct lighting from the area light is sampled explicitly
// at every diffuse hit (next-event estimation), which is what makes the image clean up
// quickly instead of waiting for random bounces to find the light.
//
// This file is the integrator (the bounce loop). The building blocks live in their own files so
// one can be changed without touching the others:
//   camera.cuh       pixel -> camera ray                 bsdf.cuh      materials (how light bounces)
//   surface.cuh      material + texture maps at a hit    textures.cuh  texture/normal map lookups
//   lights.cuh       area light sampling                 environment.cuh  HDRI dome lighting
//   rays.cuh         ray types, hit information          sampling.cuh  random directions, MIS
// "Claudes Folder/tools/regress.sh" (private workspace) checks that a change didn't alter renders.
// SPDX-License-Identifier: Apache-2.0

#include <optix.h>

#include "LaunchParams.h"
#include "random.h"
#include "vec_math.h"

using namespace moa;

extern "C" {
__constant__ LaunchParams params;
}

// The building blocks, one file per job (see each file for details).
#include "rays.cuh"         // ray types and hit information
#include "sampling.cuh"     // random directions, frames, MIS
#include "bsdf.cuh"         // materials
#include "textures.cuh"     // texture maps
#include "environment.cuh"  // HDRI lighting
#include "lights.cuh"       // area lights
#include "camera.cuh"       // camera rays
#include "surface.cuh"      // material + maps at a hit point

namespace {

// Firefly control: limits how bright a caustic contribution can be (light that went
// diffuse -> mirror/glass -> light). Those paths are found only rarely by random bounces and
// show up as bright speckles. Clamping is slightly biased (like Redshift's "max subsample
// intensity") but removes them. Direct light sampling is smooth and is never clamped.
__forceinline__ __device__ float3 clampCaustic(float3 c)
{
    if (params.maxSampleValue <= 0.0f)
        return c;
    const float m = maxComponent(c);
    return m > params.maxSampleValue ? c * (params.maxSampleValue / m) : c;
}

__forceinline__ __device__ bool isEmissive(const Material& m)
{
    return m.emission.x > 0.0f || m.emission.y > 0.0f || m.emission.z > 0.0f;
}

}  // namespace

extern "C" __global__ void __raygen__render()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int pixel = idx.y * params.width + idx.x;
    unsigned int seed = tea<4>(pixel, params.subframeIndex);

    const Camera& cam = params.camera;
    const QuadLight& light = params.light;

    float3 sumRadiance = make_float3(0.0f, 0.0f, 0.0f);
    float3 sumAlbedo = make_float3(0.0f, 0.0f, 0.0f);
    float3 sumNormal = make_float3(0.0f, 0.0f, 0.0f);
    float firstDepth = 0.0f;  // depth pass: 0 = the camera ray hit nothing

    for (unsigned int s = 0; s < params.samplesPerLaunch; ++s)
    {
        float3 origin, direction;
        generateCameraRay(cam, idx, seed, origin, direction);

        float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
        float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
        float3 albedoAov = make_float3(0.0f, 0.0f, 0.0f);
        float3 normalAov = make_float3(0.0f, 0.0f, 0.0f);
        bool countEmission = true;  // camera rays and specular bounces see lights directly
        bool afterDiffuse = false;  // has the path bounced off a diffuse surface yet?
        float lastBsdfPdf = 0.0f;   // pdf of the last diffuse bounce direction (for MIS with the HDRI)
        bool lastWasDiffuse = false;

        for (unsigned int depth = 0; depth < params.maxDepth; ++depth)
        {
            HitInfo hit;
            hit.hit = 0u;
            traceRadiance(origin, direction, hit);
            if (!hit.hit)
            {
                if (params.envWidth > 0)
                {
                    const float3 le = envRadiance(direction);
                    if (depth == 0)
                    {
                        if (params.envVisible)
                            radiance += le;
                    }
                    else if (lastWasDiffuse)
                    {
                        // This direction could also have been picked by HDRI sampling: weight it.
                        radiance += throughput * le * misWeight(lastBsdfPdf, envPdf(direction));
                    }
                    else
                        radiance += throughput * le;  // seen through a mirror or glass
                }
                else
                {
                    // Uniform background light. It isn't sampled by NEE, so every path that escapes
                    // picks it up here (diffuse bounces included) and the estimate stays correct.
                    radiance += throughput * params.background;
                }
                break;
            }

            // The material at this point, with its texture maps applied.
            const SurfacePoint sp = resolveSurface(hit, direction);
            const Material& m = sp.m;
            const bool frontFace = sp.frontFace;
            const float3 ng = sp.ng;
            const float3 n = sp.n;
            const float3 albedo = sp.albedo;

            if (depth == 0)
            {
                albedoAov = albedo;
                normalAov = n;
                if (s == 0)
                    firstDepth = dot(hit.position - cam.eye, normalize(cam.w));
            }

            if (isEmissive(m))
            {
                // Lights only emit from their front side and don't reflect in this version.
                if (countEmission && frontFace)
                {
                    const float3 e = throughput * m.emission;
                    radiance += afterDiffuse ? clampCaustic(e) : e;
                }
                break;
            }

            lastWasDiffuse = false;
            const bool isStandard = m.type == MAT_STANDARD;
            const bool standardGlass = isStandard && m.transmission > 0.5f;
            const Frame frame = makeFrame(n);
            float3 woLocal = frame.toLocal(-direction);
            woLocal.z = fmaxf(woLocal.z, 1e-4f);  // normal maps can tilt n slightly past the eye
            StandardBsdf sb = {};
            if (isStandard && !standardGlass)
                sb = makeStandardBsdf(m, albedo, woLocal.z);

            // Light sampling (next-event estimation) for matte and opaque standard surfaces.
            const bool useNee = m.type == MAT_DIFFUSE || (isStandard && !standardGlass);
            auto evalBsdf = [&](float3 l, float& pdfBsdf) -> float3 {
                if (m.type == MAT_DIFFUSE)
                {
                    pdfBsdf = fmaxf(dot(n, l), 0.0f) * kInvPi;
                    return albedo * kInvPi;
                }
                return evalStandard(sb, woLocal, frame.toLocal(l), pdfBsdf);
            };

            if (useNee && params.envWidth > 0)
            {
                // Towards the HDRI, combined with bounce sampling via MIS.
                float pdfLight = 0.0f;
                const float3 l = sampleEnvironment(rnd(seed), rnd(seed), pdfLight);
                const float cosSurface = dot(n, l);
                if (pdfLight > 0.0f && cosSurface > 0.0f && dot(ng, l) > 0.0f)
                {
                    float pdfBsdf = 0.0f;
                    const float3 f = evalBsdf(l, pdfBsdf);
                    if (maxComponent(f) > 0.0f && traceShadow(hit.position + ng * kRayEpsilon, l, kRayTMax))
                    {
                        const float3 le = envRadiance(l);
                        const float3 c = throughput * f * le * (cosSurface / pdfLight * misWeight(pdfLight, pdfBsdf));
                        // Glossy surfaces seen after a diffuse bounce make caustics: clamp those.
                        radiance += (afterDiffuse && isStandard) ? clampCaustic(c) : c;
                    }
                }
            }
            if (useNee && params.hasQuadLight)
            {
                // Pick a point on the area light and test if it is visible.
                const float3 lightPoint = sampleQuadLightPoint(light, seed);
                const float3 toLight = lightPoint - hit.position;
                const float dist2 = dot(toLight, toLight);
                const float dist = sqrtf(dist2);
                const float3 l = toLight / dist;
                const float cosSurface = dot(n, l);
                const float cosLight = -dot(light.normal, l);
                if (cosSurface > 0.0f && cosLight > 0.0f && dot(ng, l) > 0.0f)
                {
                    float pdfBsdf = 0.0f;
                    const float3 f = evalBsdf(l, pdfBsdf);
                    const float3 shadowOrigin = hit.position + ng * kRayEpsilon;
                    if (maxComponent(f) > 0.0f && traceShadow(shadowOrigin, l, dist - 2.0f * kRayEpsilon))
                    {
                        // BSDF * Le * geometry term / pdf(area), with pdf = 1/area.
                        const float g = cosSurface * cosLight / dist2;
                        const float3 c = throughput * f * light.emission * (g * light.area);
                        radiance += (afterDiffuse && isStandard) ? clampCaustic(c) : c;
                    }
                }
            }

            if (useNee && params.lightCount > 0)
            {
                // One of the extra lights, picked at random (its contribution x the light count).
                unsigned int li = static_cast<unsigned int>(rnd(seed) * static_cast<float>(params.lightCount));
                if (li >= params.lightCount)
                    li = params.lightCount - 1;
                float3 l, value;
                float dist = 0.0f;
                if (sampleLightSource(params.lights[li], hit.position, seed, l, dist, value))
                {
                    const float cosSurface = dot(n, l);
                    if (cosSurface > 0.0f && dot(ng, l) > 0.0f)
                    {
                        float pdfBsdf = 0.0f;
                        const float3 f = evalBsdf(l, pdfBsdf);
                        if (maxComponent(f) > 0.0f &&
                            traceShadow(hit.position + ng * kRayEpsilon, l, dist - 2.0f * kRayEpsilon))
                        {
                            const float3 c = throughput * f * value * (cosSurface * static_cast<float>(params.lightCount));
                            radiance += (afterDiffuse && isStandard) ? clampCaustic(c) : c;
                        }
                    }
                }
            }

            if (m.type == MAT_DIFFUSE)
            {
                // Continue the path with a cosine-weighted bounce. The cosine and pi cancel
                // against the pdf, leaving just the albedo.
                direction = sampleCosineHemisphere(n, rnd(seed), rnd(seed));
                origin = hit.position + ng * kRayEpsilon;
                throughput *= albedo;
                countEmission = false;  // direct light was already counted by NEE
                afterDiffuse = true;
                lastWasDiffuse = true;
                lastBsdfPdf = fmaxf(dot(n, direction), 0.0f) * kInvPi;
                if (dot(ng, direction) <= 0.0f)
                    break;  // bent (normal-mapped) normal sent the bounce into the surface
            }
            else if (isStandard && !standardGlass)
            {
                float3 wiLocal;
                float pdf = 0.0f;
                bool delta = false;
                const float3 weight = sampleStandard(sb, woLocal, seed, wiLocal, pdf, delta);
                if (maxComponent(weight) <= 0.0f)
                    break;
                direction = frame.toWorld(wiLocal);
                origin = hit.position + ng * kRayEpsilon;
                throughput *= weight;
                if (delta)
                    countEmission = true;  // mirror-sharp reflection: lights are seen directly
                else
                {
                    countEmission = false;
                    afterDiffuse = true;
                    lastWasDiffuse = true;
                    lastBsdfPdf = pdf;
                }
                if (dot(ng, direction) <= 0.0f)
                    break;
            }
            else if (m.type == MAT_MIRROR)
            {
                direction = reflect(direction, n);
                origin = hit.position + ng * kRayEpsilon;
                throughput *= albedo;
                countEmission = true;
            }
            else if (standardGlass && m.roughness >= 0.02f)
            {
                // Rough glass: pick a visible microfacet, then reflect or refract through it by
                // its Fresnel reflectance. The weight left over is the masking of the new direction.
                const float eta = frontFace ? 1.0f / m.ior : m.ior;
                const float rough = clampf(m.roughness, 0.0f, 1.0f);
                const float alpha = fmaxf(rough * rough, 1e-4f);
                const float3 h = sampleGgxVisibleNormal(woLocal, alpha, rnd(seed), rnd(seed));
                const float voh = fmaxf(dot(woLocal, h), 1e-6f);
                const bool reflected = rnd(seed) < fresnelDielectric(voh, eta);
                const float3 wiLocal = reflected ? reflect(-woLocal, h) : refractDir(-woLocal, h, eta, voh);
                if (reflected ? wiLocal.z <= 0.0f : wiLocal.z >= 0.0f)
                    break;
                const float g1 = 1.0f / (1.0f + smithLambda(alpha * alpha, fabsf(wiLocal.z)));
                direction = frame.toWorld(wiLocal);
                origin = hit.position + (reflected ? ng : -ng) * kRayEpsilon;
                throughput *= albedo * g1;
                countEmission = true;
            }
            else  // MAT_GLASS, or smooth standard glass
            {
                const float eta = frontFace ? 1.0f / m.ior : m.ior;
                const float cosI = -dot(direction, n);
                const float reflectance = fresnelDielectric(cosI, eta);
                if (rnd(seed) < reflectance)
                {
                    direction = reflect(direction, n);
                    origin = hit.position + ng * kRayEpsilon;
                }
                else
                {
                    direction = refractDir(direction, n, eta, cosI);
                    origin = hit.position - ng * kRayEpsilon;
                }
                throughput *= albedo;
                countEmission = true;
            }

            // Russian roulette: randomly end weak paths, boosting the survivors to stay unbiased.
            if (depth >= 3)
            {
                const float survive = clampf(maxComponent(throughput), 0.05f, 0.95f);
                if (rnd(seed) >= survive)
                    break;
                throughput /= survive;
            }
        }

        if (!isFinite(radiance))
            radiance = make_float3(0.0f, 0.0f, 0.0f);  // never let one bad sample poison the image

        sumRadiance += radiance;
        sumAlbedo += albedoAov;
        sumNormal += normalAov;
    }

    // Progressive accumulation: keep a running average across launches.
    const float invSamples = 1.0f / static_cast<float>(params.samplesPerLaunch);
    const float3 newRadiance = sumRadiance * invSamples;
    const float3 newAlbedo = sumAlbedo * invSamples;
    const float3 newNormal = sumNormal * invSamples;

    if (params.subframeIndex == 0)
    {
        if (params.depth)
            params.depth[pixel] = firstDepth;
        params.beauty[pixel] = make_float4(newRadiance.x, newRadiance.y, newRadiance.z, 1.0f);
        params.albedo[pixel] = make_float4(newAlbedo.x, newAlbedo.y, newAlbedo.z, 1.0f);
        params.normal[pixel] = make_float4(newNormal.x, newNormal.y, newNormal.z, 0.0f);
    }
    else
    {
        const float a = 1.0f / static_cast<float>(params.subframeIndex + 1);
        const float4 pb = params.beauty[pixel];
        const float4 pa = params.albedo[pixel];
        const float4 pn = params.normal[pixel];
        const float3 b = lerp(make_float3(pb.x, pb.y, pb.z), newRadiance, a);
        const float3 al = lerp(make_float3(pa.x, pa.y, pa.z), newAlbedo, a);
        const float3 nn = lerp(make_float3(pn.x, pn.y, pn.z), newNormal, a);
        params.beauty[pixel] = make_float4(b.x, b.y, b.z, 1.0f);
        params.albedo[pixel] = make_float4(al.x, al.y, al.z, 1.0f);
        params.normal[pixel] = make_float4(nn.x, nn.y, nn.z, 0.0f);
    }
}

extern "C" __global__ void __closesthit__radiance()
{
    HitInfo* hit = getHitInfo();
    const unsigned int prim = optixGetPrimitiveIndex();
    const uint3 tri = params.indices[prim];
    const float3 v0 = params.positions[tri.x];
    const float3 v1 = params.positions[tri.y];
    const float3 v2 = params.positions[tri.z];

    const float t = optixGetRayTmax();
    hit->position = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * t;
    const float3 ng = normalize(cross(v1 - v0, v2 - v0));
    hit->normal = ng;

    // Interpolate the vertex normals with the hit's barycentric coordinates.
    const float2 bary = optixGetTriangleBarycentrics();
    float3 ns = params.normals[tri.x] * (1.0f - bary.x - bary.y) + params.normals[tri.y] * bary.x +
                params.normals[tri.z] * bary.y;
    const float len = length(ns);
    ns = len > 0.0f ? ns / len : ng;
    hit->shadingNormal = dot(ns, ng) < 0.0f ? -ns : ns;  // keep it on the geometric side

    const float b0 = 1.0f - bary.x - bary.y;
    const float2 t0 = params.texcoords[tri.x], t1 = params.texcoords[tri.y], t2 = params.texcoords[tri.z];
    hit->uv = make_float2(t0.x * b0 + t1.x * bary.x + t2.x * bary.y, t0.y * b0 + t1.y * bary.x + t2.y * bary.y);
    const float4 g0 = params.tangents[tri.x], g1 = params.tangents[tri.y], g2 = params.tangents[tri.z];
    hit->tangent = make_float4(g0.x * b0 + g1.x * bary.x + g2.x * bary.y, g0.y * b0 + g1.y * bary.x + g2.y * bary.y,
                               g0.z * b0 + g1.z * bary.x + g2.z * bary.y, g0.w);
    hit->materialId = params.triangleMaterial[prim];
    hit->hit = 1u;
}

extern "C" __global__ void __miss__radiance()
{
    getHitInfo()->hit = 0u;
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(1u);  // nothing was in the way
}
