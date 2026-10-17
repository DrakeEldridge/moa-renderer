// MOA Renderer - GPU programs for the path tracer (compiled to PTX and loaded by OptiX).
//
// Structure: the ray-generation program runs the whole path loop for one pixel. It
// traces a ray, the closest-hit program reports what was hit, and the loop decides how
// the light path continues. Direct lighting from the area light is sampled explicitly
// at every diffuse hit (next-event estimation), which is what makes the image clean up
// quickly instead of waiting for random bounces to find the light.
// SPDX-License-Identifier: Apache-2.0

#include <optix.h>

#include "LaunchParams.h"
#include "random.h"
#include "vec_math.h"

using namespace moa;

extern "C" {
__constant__ LaunchParams params;
}

namespace {

constexpr float kRayEpsilon = 1e-3f;  // offset to avoid a surface re-hitting itself
constexpr float kRayTMax = 1e16f;

// Per-ray result written by the closest-hit / miss programs.
struct HitInfo
{
    float3 position;
    float3 normal;         // geometric normal, as wound in the mesh (not yet flipped to face the ray)
    float3 shadingNormal;  // smooth interpolated normal, on the same side as `normal`
    float4 tangent;        // interpolated tangent (xyz) and bitangent sign (w)
    float2 uv;
    unsigned int materialId;
    unsigned int hit;
};

// The radiance ray carries a pointer to a HitInfo split across two 32-bit payload slots.
__forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long p = reinterpret_cast<unsigned long long>(ptr);
    i0 = static_cast<unsigned int>(p >> 32);
    i1 = static_cast<unsigned int>(p & 0xffffffffull);
}

__forceinline__ __device__ HitInfo* getHitInfo()
{
    const unsigned long long p =
        (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
    return reinterpret_cast<HitInfo*>(p);
}

__forceinline__ __device__ void traceRadiance(float3 origin, float3 direction, HitInfo& hit)
{
    unsigned int p0, p1;
    packPointer(&hit, p0, p1);
    optixTrace(params.handle, origin, direction,
               0.0f, kRayTMax, 0.0f,  // tmin, tmax, ray time
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,  // SBT offset, stride, miss index
               p0, p1);
}

// Returns true if nothing blocks the segment from origin along direction up to distance.
__forceinline__ __device__ bool traceShadow(float3 origin, float3 direction, float distance)
{
    unsigned int visible = 0u;  // the shadow miss program sets this to 1
    optixTrace(params.handle, origin, direction,
               0.0f, distance, 0.0f,
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT |
                   OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               RAY_TYPE_SHADOW, RAY_TYPE_COUNT, RAY_TYPE_SHADOW,
               visible);
    return visible != 0u;
}

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

// Power heuristic for combining two sampling strategies (multiple importance sampling).
__forceinline__ __device__ float misWeight(float pdfA, float pdfB)
{
    const float a = pdfA * pdfA, b = pdfB * pdfB;
    return a + b > 0.0f ? a / (a + b) : 0.0f;
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
        // Jittered position inside the pixel gives free anti-aliasing.
        const float sx = (static_cast<float>(idx.x) + rnd(seed)) / static_cast<float>(params.width);
        const float sy = (static_cast<float>(idx.y) + rnd(seed)) / static_cast<float>(params.height);
        float3 origin = cam.eye;
        float3 direction = normalize(cam.u * (2.0f * sx - 1.0f) + cam.v * (2.0f * sy - 1.0f) + cam.w);

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

            const Material m = params.materials[hit.materialId];
            const bool frontFace = dot(direction, hit.normal) < 0.0f;
            // ng: true surface side (used to offset new rays so they don't re-hit the surface).
            // n: smooth shading normal on the same side, used for lighting and reflection.
            const float3 ng = frontFace ? hit.normal : -hit.normal;
            const float3 mappedNormal = m.normalTexture >= 0 ? applyNormalMap(m, hit) : hit.shadingNormal;
            const float3 n = frontFace ? mappedNormal : -mappedNormal;

            float3 albedo = m.albedo;
            if (m.colorTexture >= 0)
            {
                const float4 c = sampleTexture(m.colorTexture, hit.uv, m.flags);
                albedo = albedo * make_float3(c.x, c.y, c.z);
            }

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
            if (m.type == MAT_DIFFUSE)
            {
                // Next-event estimation towards the HDRI, combined with bounce sampling via MIS.
                if (params.envWidth > 0)
                {
                    float pdfLight = 0.0f;
                    const float3 l = sampleEnvironment(rnd(seed), rnd(seed), pdfLight);
                    const float cosSurface = dot(n, l);
                    if (pdfLight > 0.0f && cosSurface > 0.0f && dot(ng, l) > 0.0f)
                    {
                        if (traceShadow(hit.position + ng * kRayEpsilon, l, kRayTMax))
                        {
                            const float pdfBsdf = cosSurface * kInvPi;
                            const float3 le = envRadiance(l);
                            radiance += throughput * albedo * le *
                                        (kInvPi * cosSurface / pdfLight * misWeight(pdfLight, pdfBsdf));
                        }
                    }
                }
            }
            if (m.type == MAT_DIFFUSE && params.hasQuadLight)
            {
                // Next-event estimation: pick a point on the light and test if it is visible.
                const float3 lightPoint = light.corner + light.edge1 * rnd(seed) + light.edge2 * rnd(seed);
                const float3 toLight = lightPoint - hit.position;
                const float dist2 = dot(toLight, toLight);
                const float dist = sqrtf(dist2);
                const float3 l = toLight / dist;
                const float cosSurface = dot(n, l);
                const float cosLight = -dot(light.normal, l);
                if (cosSurface > 0.0f && cosLight > 0.0f)
                {
                    const float3 shadowOrigin = hit.position + ng * kRayEpsilon;
                    if (traceShadow(shadowOrigin, l, dist - 2.0f * kRayEpsilon))
                    {
                        // BSDF (albedo/pi) * Le * geometry term / pdf(area), with pdf = 1/area.
                        const float g = cosSurface * cosLight / dist2;
                        radiance += throughput * albedo * light.emission * (kInvPi * g * light.area);
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
            else if (m.type == MAT_MIRROR)
            {
                direction = reflect(direction, n);
                origin = hit.position + ng * kRayEpsilon;
                throughput *= albedo;
                countEmission = true;
            }
            else  // MAT_GLASS
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
