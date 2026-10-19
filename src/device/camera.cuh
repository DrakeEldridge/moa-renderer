// MOA Renderer - camera: turns a pixel into a ray. (Depth of field / bokeh / lens shapes go here.)
// SPDX-License-Identifier: Apache-2.0
// Part of the GPU path tracer; included by pathtracer.cu after `params` is declared.
#pragma once

namespace {

// Pinhole camera with a random position inside the pixel (free anti-aliasing).
// Uses two random numbers from `seed`.
__forceinline__ __device__ void generateCameraRay(const Camera& cam, uint3 idx, unsigned int& seed, float3& origin,
                                                  float3& direction)
{
    // Jittered position inside the pixel gives free anti-aliasing.
    const float sx = (static_cast<float>(idx.x) + rnd(seed)) / static_cast<float>(params.width);
    const float sy = (static_cast<float>(idx.y) + rnd(seed)) / static_cast<float>(params.height);
    origin = cam.eye;
    direction = normalize(cam.u * (2.0f * sx - 1.0f) + cam.v * (2.0f * sy - 1.0f) + cam.w);
}

}  // namespace
