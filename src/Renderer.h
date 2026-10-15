// MOA Renderer - the OptiX renderer: owns the GPU context, pipeline, scene buffers and denoiser.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "CudaBuffer.h"
#include "Scene.h"
#include "device/LaunchParams.h"

#include <optix.h>

#include <functional>
#include <string>
#include <vector>

namespace moa {

struct RenderSettings
{
    unsigned int width = 1024;
    unsigned int height = 1024;
    unsigned int samplesPerPixel = 256;
    unsigned int samplesPerLaunch = 4;  // larger = faster, but each GPU launch takes longer
    unsigned int maxDepth = 8;
    float maxSampleValue = 4.0f;  // caustic firefly clamp; 0 disables it
    float background = 0.0f;      // uniform grey background light (rays leaving the scene)
};

class Renderer
{
public:
    // Loads the compiled GPU programs from `ptxPath` and sets up OptiX.
    Renderer(const std::string& ptxPath, bool verbose);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    const std::string& deviceName() const { return m_deviceName; }

    void setScene(const Scene& scene);

    // Progressive render. Calls `progress(samplesDone, samplesTotal)` after each launch.
    // Returns the render time in milliseconds.
    double render(const RenderSettings& settings, const std::function<void(unsigned int, unsigned int)>& progress);

    // Interactive (progressive) rendering: beginProgressive() resets the image, then each
    // renderPass() adds samplesPerLaunch samples per pixel. Used by the live window.
    void beginProgressive(const RenderSettings& settings);
    void renderPass();

    // Camera used by the next beginProgressive()/render(). setScene() resets it to the scene's camera.
    void setCamera(const CameraDesc& camera) { m_cameraDesc = camera; }
    const CameraDesc& camera() const { return m_cameraDesc; }
    unsigned int width() const { return m_width; }
    unsigned int height() const { return m_height; }

    // Runs the OptiX AI denoiser on the current render, guided by the albedo and normal passes.
    // Returns the denoise time in milliseconds.
    double denoise();

    unsigned int samplesRendered() const { return m_samplesRendered; }

    void downloadBeauty(std::vector<float4>& out) const { m_beauty.download(out); }
    void downloadAlbedo(std::vector<float4>& out) const { m_albedo.download(out); }
    void downloadNormal(std::vector<float4>& out) const { m_normal.download(out); }
    void downloadDenoised(std::vector<float4>& out) const { m_denoised.download(out); }

private:
    void createContext(bool verbose);
    void createModule(const std::string& ptxPath);
    void createProgramGroups();
    void createPipeline();
    void createShaderBindingTable();
    void buildAccel(const Scene& scene);
    void resizeFrame(unsigned int width, unsigned int height);
    void setupDenoiser();

    std::string m_deviceName;

    OptixDeviceContext m_context = nullptr;
    OptixModule m_module = nullptr;
    OptixPipelineCompileOptions m_pipelineOptions = {};
    OptixProgramGroup m_raygenPG = nullptr;
    OptixProgramGroup m_missRadiancePG = nullptr;
    OptixProgramGroup m_missShadowPG = nullptr;
    OptixProgramGroup m_hitPG = nullptr;
    OptixPipeline m_pipeline = nullptr;

    OptixShaderBindingTable m_sbt = {};
    CudaBuffer m_raygenRecord;
    CudaBuffer m_missRecords;
    CudaBuffer m_hitRecords;

    // Scene
    CudaBuffer m_positions;
    CudaBuffer m_normals;
    CudaBuffer m_indices;
    CudaBuffer m_triangleMaterial;
    CudaBuffer m_materials;
    CudaBuffer m_gas;
    OptixTraversableHandle m_gasHandle = 0;
    Camera m_camera = {};
    CameraDesc m_cameraDesc = {};
    QuadLight m_light = {};

    // Frame buffers (running averages)
    unsigned int m_width = 0;
    unsigned int m_height = 0;
    unsigned int m_samplesRendered = 0;
    CudaBuffer m_beauty;
    CudaBuffer m_albedo;
    CudaBuffer m_normal;
    CudaBuffer m_launchParams;
    LaunchParams m_launch = {};

    // Denoiser
    OptixDenoiser m_denoiser = nullptr;
    unsigned int m_denoiserWidth = 0;
    unsigned int m_denoiserHeight = 0;
    CudaBuffer m_denoiserState;
    CudaBuffer m_denoiserScratch;
    CudaBuffer m_denoised;
};

}  // namespace moa
