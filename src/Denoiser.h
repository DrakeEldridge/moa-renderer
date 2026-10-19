// MOA Renderer - the denoiser slot. The renderer only knows "denoise this image"; which denoiser
// does the work (NVIDIA's OptiX AI denoiser today; Intel Open Image Denoise for non-NVIDIA farm
// machines or MOA's own AI denoiser later) is a swappable module behind this interface.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optix.h>
#include <vector_types.h>

#include <memory>

namespace moa {

// The render's passes, on the GPU (width * height float4 pixels each).
struct DenoiseInputs
{
    const float4* beauty = nullptr;  // noisy image (linear HDR)
    const float4* albedo = nullptr;  // first-hit surface colour (guide)
    const float4* normal = nullptr;  // first-hit world-space normal (guide)
    unsigned int width = 0;
    unsigned int height = 0;
};

class Denoiser
{
public:
    virtual ~Denoiser() = default;
    // Short name for logs and UI, e.g. "OptiX AI".
    virtual const char* name() const = 0;
    // Writes the denoised beauty to `output` (GPU, width * height float4). Blocks until done.
    virtual void denoise(const DenoiseInputs& inputs, float4* output) = 0;
};

// NVIDIA OptiX AI denoiser (AOV model, guided by albedo + normals). Needs an NVIDIA RTX-capable GPU.
std::unique_ptr<Denoiser> createOptixDenoiser(OptixDeviceContext context);

}  // namespace moa
