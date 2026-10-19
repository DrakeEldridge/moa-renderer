// MOA Renderer - the OptiX AI denoiser module (see Denoiser.h).
// SPDX-License-Identifier: Apache-2.0
#include "Denoiser.h"

#include "Check.h"
#include "CudaBuffer.h"

#include <cuda_runtime.h>
#include <optix_stubs.h>

namespace moa {
namespace {

class OptixAiDenoiser final : public Denoiser
{
public:
    explicit OptixAiDenoiser(OptixDeviceContext context) : m_context(context) {}
    ~OptixAiDenoiser() override
    {
        // Release GPU memory before destroying the denoiser.
        m_state.release();
        m_scratch.release();
        if (m_denoiser)
            optixDenoiserDestroy(m_denoiser);
    }

    const char* name() const override { return "OptiX AI"; }

    void denoise(const DenoiseInputs& in, float4* output) override
    {
        setup(in.width, in.height);

        auto image = [&](const void* data) {
            OptixImage2D img = {};
            img.data = reinterpret_cast<CUdeviceptr>(data);
            img.width = in.width;
            img.height = in.height;
            img.rowStrideInBytes = in.width * static_cast<unsigned int>(sizeof(float4));
            img.pixelStrideInBytes = static_cast<unsigned int>(sizeof(float4));
            img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
            return img;
        };

        OptixDenoiserGuideLayer guide = {};
        guide.albedo = image(in.albedo);
        guide.normal = image(in.normal);  // world-space normals, as the AOV model expects

        OptixDenoiserLayer layer = {};
        layer.input = image(in.beauty);
        layer.output = image(output);
        layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

        OptixDenoiserParams params = {};  // automatic exposure, no blending

        OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, nullptr, &params, m_state.dptr(), m_state.size(), &guide, &layer,
                                        1, 0, 0, m_scratch.dptr(), m_scratch.size()));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

private:
    // Creates the denoiser once and (re)allocates its memory when the image size changes.
    void setup(unsigned int width, unsigned int height)
    {
        if (m_denoiser && m_width == width && m_height == height)
            return;

        if (!m_denoiser)
        {
            OptixDenoiserOptions options = {};
            options.guideAlbedo = 1;
            options.guideNormal = 1;
            options.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
            OPTIX_CHECK(optixDenoiserCreate(m_context, OPTIX_DENOISER_MODEL_KIND_AOV, &options, &m_denoiser));
        }

        OptixDenoiserSizes sizes = {};
        OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, width, height, &sizes));
        m_state.resize(sizes.stateSizeInBytes);
        m_scratch.resize(sizes.withoutOverlapScratchSizeInBytes);
        OPTIX_CHECK(optixDenoiserSetup(m_denoiser, nullptr, width, height, m_state.dptr(), m_state.size(),
                                       m_scratch.dptr(), m_scratch.size()));
        m_width = width;
        m_height = height;
    }

    OptixDeviceContext m_context = nullptr;
    OptixDenoiser m_denoiser = nullptr;
    unsigned int m_width = 0;
    unsigned int m_height = 0;
    CudaBuffer m_state;
    CudaBuffer m_scratch;
};

}  // namespace

std::unique_ptr<Denoiser> createOptixDenoiser(OptixDeviceContext context)
{
    return std::make_unique<OptixAiDenoiser>(context);
}

}  // namespace moa
