// MOA Renderer - data shared between the host renderer and the GPU programs.
// Everything in here must have the same memory layout on the CPU and the GPU,
// so keep it to plain structs of CUDA vector types, pointers and integers.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optix_types.h>
#include <vector_types.h>

namespace moa {

enum RayType : unsigned int
{
    RAY_TYPE_RADIANCE = 0,  // regular path-tracing rays
    RAY_TYPE_SHADOW = 1,    // visibility tests towards lights
    RAY_TYPE_COUNT
};

enum MaterialType : unsigned int
{
    MAT_DIFFUSE = 0,  // Lambertian (matte) surface
    MAT_MIRROR = 1,   // perfect mirror, tinted by albedo
    MAT_GLASS = 2,    // smooth dielectric with Fresnel, tinted by albedo
};

struct Material
{
    float3 albedo;      // base colour / tint
    float3 emission;    // emitted radiance (non-zero for lights)
    unsigned int type;  // MaterialType
    float ior;          // index of refraction (glass only)
};

// Pinhole camera. Primary ray direction = normalize(x * u + y * v + w) for x, y in [-1, 1].
struct Camera
{
    float3 eye;
    float3 u;  // right, scaled by tan(fov/2) * aspect
    float3 v;  // up, scaled by tan(fov/2)
    float3 w;  // forward, unit length
};

// A single rectangular area light, sampled directly for next-event estimation.
struct QuadLight
{
    float3 corner;
    float3 edge1;
    float3 edge2;
    float3 normal;    // direction the light emits towards (unit length)
    float3 emission;  // radiance
    float area;
};

struct LaunchParams
{
    unsigned int width;
    unsigned int height;
    unsigned int subframeIndex;     // 0 on the first launch, increments every launch
    unsigned int samplesPerLaunch;  // samples per pixel traced in one launch
    unsigned int maxDepth;          // maximum number of bounces
    float maxSampleValue;           // clamp for caustic contributions (0 = off)
    float3 background;              // radiance seen by rays that leave the scene

    // Running averages, one float4 per pixel (w unused for albedo/normal).
    float4* beauty;
    float4* albedo;
    float4* normal;

    Camera camera;
    QuadLight light;

    // Scene geometry: indexed triangles with a shading normal per vertex.
    const float3* positions;
    const float3* normals;
    const uint3* indices;
    const unsigned int* triangleMaterial;  // material index per triangle
    const Material* materials;

    OptixTraversableHandle handle;
};

}  // namespace moa
