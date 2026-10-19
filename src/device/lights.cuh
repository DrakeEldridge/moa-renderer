// MOA Renderer - lights: picking points on light sources for direct lighting.
// (More light types - sphere, disk, distant, many-light sampling - go here.)
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

namespace {

// A uniformly random point on a rectangular area light (pdf = 1 / area). Uses two random numbers.
__forceinline__ __device__ float3 sampleQuadLightPoint(const QuadLight& light, unsigned int& seed)
{
    return light.corner + light.edge1 * rnd(seed) + light.edge2 * rnd(seed);
}

}  // namespace
