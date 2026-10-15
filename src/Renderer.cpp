// MOA Renderer - the OptiX renderer.
// SPDX-License-Identifier: Apache-2.0
#include "Renderer.h"

#include "device/vec_math.h"

// optix_function_table_definition.h must be included in exactly one source file.
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace moa {

namespace {

// Shader binding table record. The header is filled by OptiX; our programs read all data
// from the launch parameters, so the payload is a placeholder.
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    int unused;
};

void optixLog(unsigned int level, const char* tag, const char* message, void* /*cbdata*/)
{
    std::fprintf(stderr, "[OptiX %u][%s] %s\n", level, tag, message);
}

std::string readTextFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Couldn't open GPU program file: " + path +
                                 "\nIt should sit next to moa.exe (the build copies it there).");
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::chrono::steady_clock::time_point now() { return std::chrono::steady_clock::now(); }

double msSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(now() - start).count();
}

Camera makeCamera(const CameraDesc& desc, unsigned int width, unsigned int height)
{
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float tanHalf = std::tan(0.5f * desc.verticalFovDegrees * kPi / 180.0f);
    Camera c;
    c.eye = desc.eye;
    c.w = normalize(desc.lookAt - desc.eye);
    const float3 right = normalize(cross(c.w, desc.up));
    const float3 up = normalize(cross(right, c.w));
    c.u = right * (tanHalf * aspect);
    c.v = up * tanHalf;
    return c;
}

}  // namespace

Renderer::Renderer(const std::string& ptxPath, bool verbose)
{
    createContext(verbose);
    createModule(ptxPath);
    createProgramGroups();
    createPipeline();
    createShaderBindingTable();
    m_launchParams.resize(sizeof(LaunchParams));
}

Renderer::~Renderer()
{
    // Release GPU memory before tearing down OptiX objects.
    m_denoiserState.release();
    m_denoiserScratch.release();
    m_denoised.release();
    if (m_denoiser)
        optixDenoiserDestroy(m_denoiser);
    if (m_pipeline)
        optixPipelineDestroy(m_pipeline);
    for (OptixProgramGroup pg : {m_raygenPG, m_missRadiancePG, m_missShadowPG, m_hitPG})
        if (pg)
            optixProgramGroupDestroy(pg);
    if (m_module)
        optixModuleDestroy(m_module);
    if (m_context)
        optixDeviceContextDestroy(m_context);
}

void Renderer::createContext(bool verbose)
{
    // Initialise CUDA on the default GPU.
    CUDA_CHECK(cudaFree(nullptr));
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    m_deviceName = props.name;

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLog;
    options.logCallbackLevel = verbose ? 4 : 2;  // 1 fatal, 2 error, 3 warning, 4 info
    OPTIX_CHECK(optixDeviceContextCreate(nullptr /* current CUDA context */, &options, &m_context));
}

void Renderer::createModule(const std::string& ptxPath)
{
    const std::string ptx = readTextFile(ptxPath);

    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    m_pipelineOptions = {};
    m_pipelineOptions.usesMotionBlur = 0;
    m_pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    m_pipelineOptions.numPayloadValues = 2;    // radiance rays: pointer to HitInfo
    m_pipelineOptions.numAttributeValues = 2;  // triangle barycentrics
    m_pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    m_pipelineOptions.pipelineLaunchParamsVariableName = "params";
    m_pipelineOptions.pipelineLaunchParamsSizeInBytes = sizeof(LaunchParams);
    m_pipelineOptions.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    char log[4096];
    size_t logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreate(m_context, &moduleOptions, &m_pipelineOptions, ptx.c_str(), ptx.size(),
                                      log, &logSize, &m_module));
}

void Renderer::createProgramGroups()
{
    char log[4096];
    size_t logSize = sizeof(log);
    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = m_module;
    desc.raygen.entryFunctionName = "__raygen__render";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &desc, 1, &pgOptions, log, &logSize, &m_raygenPG));

    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = m_module;
    desc.miss.entryFunctionName = "__miss__radiance";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &desc, 1, &pgOptions, log, &logSize, &m_missRadiancePG));

    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = m_module;
    desc.miss.entryFunctionName = "__miss__shadow";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &desc, 1, &pgOptions, log, &logSize, &m_missShadowPG));

    // One hit group serves both ray types: shadow rays disable the closest-hit call, so only
    // the radiance rays actually run it.
    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleCH = m_module;
    desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(m_context, &desc, 1, &pgOptions, log, &logSize, &m_hitPG));
}

void Renderer::createPipeline()
{
    const OptixProgramGroup groups[] = {m_raygenPG, m_missRadiancePG, m_missShadowPG, m_hitPG};

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;  // only the ray-generation program traces rays

    char log[4096];
    size_t logSize = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(m_context, &m_pipelineOptions, &linkOptions, groups,
                                        static_cast<unsigned int>(std::size(groups)), log, &logSize, &m_pipeline));

    OptixStackSizes stackSizes = {};
    for (OptixProgramGroup pg : groups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stackSizes, m_pipeline));

    unsigned int dcFromTraversal = 0, dcFromState = 0, continuation = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes, linkOptions.maxTraceDepth, 0, 0, &dcFromTraversal,
                                           &dcFromState, &continuation));
    OPTIX_CHECK(optixPipelineSetStackSize(m_pipeline, dcFromTraversal, dcFromState, continuation,
                                          1 /* single GAS, no instancing yet */));
}

void Renderer::createShaderBindingTable()
{
    EmptyRecord raygen = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenPG, &raygen));
    m_raygenRecord.upload(&raygen, sizeof(raygen));

    EmptyRecord miss[RAY_TYPE_COUNT] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missRadiancePG, &miss[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_missShadowPG, &miss[RAY_TYPE_SHADOW]));
    m_missRecords.upload(miss, sizeof(miss));

    EmptyRecord hit[RAY_TYPE_COUNT] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitPG, &hit[RAY_TYPE_RADIANCE]));
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitPG, &hit[RAY_TYPE_SHADOW]));
    m_hitRecords.upload(hit, sizeof(hit));

    m_sbt = {};
    m_sbt.raygenRecord = m_raygenRecord.dptr();
    m_sbt.missRecordBase = m_missRecords.dptr();
    m_sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    m_sbt.missRecordCount = RAY_TYPE_COUNT;
    m_sbt.hitgroupRecordBase = m_hitRecords.dptr();
    m_sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    m_sbt.hitgroupRecordCount = RAY_TYPE_COUNT;
}

void Renderer::setScene(const Scene& scene)
{
    if (scene.triangleCount() == 0 || scene.indices.size() != scene.triangleCount() ||
        scene.normals.size() != scene.positions.size())
        throw std::runtime_error("Scene has no triangles or inconsistent geometry");

    m_positions.upload(scene.positions);
    m_normals.upload(scene.normals);
    m_indices.upload(scene.indices);
    m_triangleMaterial.upload(scene.triangleMaterial);
    m_materials.upload(scene.materials);
    m_cameraDesc = scene.camera;
    m_light = scene.light;
    buildAccel(scene);
}

void Renderer::buildAccel(const Scene& scene)
{
    const CUdeviceptr vertexBuffer = m_positions.dptr();
    const unsigned int geometryFlags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexStrideInBytes = sizeof(float3);
    input.triangleArray.numVertices = static_cast<unsigned int>(scene.positions.size());
    input.triangleArray.vertexBuffers = &vertexBuffer;
    input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input.triangleArray.indexStrideInBytes = sizeof(uint3);
    input.triangleArray.numIndexTriplets = static_cast<unsigned int>(scene.indices.size());
    input.triangleArray.indexBuffer = m_indices.dptr();
    input.triangleArray.flags = geometryFlags;
    input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &options, &input, 1, &sizes));

    CudaBuffer temp;
    temp.resize(sizes.tempSizeInBytes);
    m_gas.resize(sizes.outputSizeInBytes);
    OPTIX_CHECK(optixAccelBuild(m_context, nullptr, &options, &input, 1, temp.dptr(), temp.size(), m_gas.dptr(),
                                m_gas.size(), &m_gasHandle, nullptr, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Renderer::resizeFrame(unsigned int width, unsigned int height)
{
    if (width == m_width && height == m_height)
        return;
    m_width = width;
    m_height = height;
    const size_t bytes = static_cast<size_t>(width) * height * sizeof(float4);
    m_beauty.resize(bytes);
    m_albedo.resize(bytes);
    m_normal.resize(bytes);
    m_denoised.resize(bytes);
}

void Renderer::beginProgressive(const RenderSettings& settings)
{
    if (!m_gasHandle)
        throw std::runtime_error("beginProgressive() called before setScene()");
    if (settings.width == 0 || settings.height == 0)
        throw std::runtime_error("Image size must be at least 1x1");

    resizeFrame(settings.width, settings.height);
    m_camera = makeCamera(m_cameraDesc, settings.width, settings.height);

    LaunchParams& lp = m_launch;
    lp = {};
    lp.width = settings.width;
    lp.height = settings.height;
    lp.samplesPerLaunch = std::max(1u, settings.samplesPerLaunch);
    lp.maxDepth = std::max(1u, settings.maxDepth);
    lp.maxSampleValue = settings.maxSampleValue;
    lp.background = make_float3(settings.background, settings.background, settings.background);
    lp.beauty = m_beauty.as<float4>();
    lp.albedo = m_albedo.as<float4>();
    lp.normal = m_normal.as<float4>();
    lp.camera = m_camera;
    lp.light = m_light;
    lp.positions = m_positions.as<const float3>();
    lp.normals = m_normals.as<const float3>();
    lp.indices = m_indices.as<const uint3>();
    lp.triangleMaterial = m_triangleMaterial.as<const unsigned int>();
    lp.materials = m_materials.as<const Material>();
    lp.handle = m_gasHandle;
    lp.subframeIndex = 0;
    m_samplesRendered = 0;
}

void Renderer::renderPass()
{
    if (m_launch.width == 0)
        throw std::runtime_error("renderPass() called before beginProgressive()");
    // Every launch traces the same number of samples, so the running average stays exact.
    CUDA_CHECK(cudaMemcpy(m_launchParams.get(), &m_launch, sizeof(m_launch), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(m_pipeline, nullptr, m_launchParams.dptr(), sizeof(LaunchParams), &m_sbt,
                            m_launch.width, m_launch.height, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    m_launch.subframeIndex += 1;
    m_samplesRendered = m_launch.subframeIndex * m_launch.samplesPerLaunch;
}

double Renderer::render(const RenderSettings& settings, const std::function<void(unsigned int, unsigned int)>& progress)
{
    beginProgressive(settings);
    const unsigned int perLaunch = m_launch.samplesPerLaunch;
    const unsigned int launches = std::max(1u, (settings.samplesPerPixel + perLaunch - 1) / perLaunch);
    const unsigned int total = launches * perLaunch;

    const auto start = now();
    for (unsigned int i = 0; i < launches; ++i)
    {
        renderPass();
        if (progress)
            progress(m_samplesRendered, total);
    }
    return msSince(start);
}

void Renderer::setupDenoiser()
{
    if (m_denoiser && m_denoiserWidth == m_width && m_denoiserHeight == m_height)
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
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_width, m_height, &sizes));
    m_denoiserState.resize(sizes.stateSizeInBytes);
    m_denoiserScratch.resize(sizes.withoutOverlapScratchSizeInBytes);
    OPTIX_CHECK(optixDenoiserSetup(m_denoiser, nullptr, m_width, m_height, m_denoiserState.dptr(),
                                   m_denoiserState.size(), m_denoiserScratch.dptr(), m_denoiserScratch.size()));
    m_denoiserWidth = m_width;
    m_denoiserHeight = m_height;
}

double Renderer::denoise()
{
    if (m_samplesRendered == 0)
        throw std::runtime_error("denoise() called before render()");

    const auto start = now();
    setupDenoiser();

    auto image = [this](const CudaBuffer& buffer) {
        OptixImage2D img = {};
        img.data = buffer.dptr();
        img.width = m_width;
        img.height = m_height;
        img.rowStrideInBytes = m_width * static_cast<unsigned int>(sizeof(float4));
        img.pixelStrideInBytes = static_cast<unsigned int>(sizeof(float4));
        img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
        return img;
    };

    OptixDenoiserGuideLayer guide = {};
    guide.albedo = image(m_albedo);
    guide.normal = image(m_normal);  // world-space normals, as the AOV model expects

    OptixDenoiserLayer layer = {};
    layer.input = image(m_beauty);
    layer.output = image(m_denoised);
    layer.type = OPTIX_DENOISER_AOV_TYPE_BEAUTY;

    OptixDenoiserParams params = {};  // automatic exposure, no blending

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, nullptr, &params, m_denoiserState.dptr(), m_denoiserState.size(),
                                    &guide, &layer, 1, 0, 0, m_denoiserScratch.dptr(), m_denoiserScratch.size()));
    CUDA_CHECK(cudaDeviceSynchronize());
    return msSince(start);
}

}  // namespace moa
