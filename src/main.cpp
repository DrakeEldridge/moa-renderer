// MOA Renderer - command-line renderer.
// Renders the built-in Cornell box test scene and writes the beauty, denoised, albedo and
// normal passes as PNGs (plus HDR .pfm files with --hdr).
// SPDX-License-Identifier: Apache-2.0
#include "ImageIO.h"
#include "LiveView.h"
#include "MeshLoader.h"
#include "Renderer.h"
#include "Scene.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr const char* kVersion = "0.1.0";

struct Options
{
    moa::RenderSettings render;
    moa::CornellOptions scene;
    std::string out = "renders/cornell";
    std::string ptx;  // empty = next to the executable
    float exposure = 1.0f;
    bool denoise = true;
    bool hdr = false;
    bool verbose = false;
    bool batch = false;  // render once and save, instead of opening the live window
    std::string model;   // optional .obj/.stl
};

void printUsage()
{
    std::printf(
        "MOA Renderer %s\n"
        "Usage: moa [options]\n"
        "Opens the live render window. Add --batch to render once and save images instead.\n"
        "  --batch              render without a window and write images (see --out)\n"
        "  --width N            image width (default 1024)\n"
        "  --height N           image height (default 1024)\n"
        "  --spp N              samples per pixel (default 256)\n"
        "  --spl N              samples per launch, higher is faster (default 4)\n"
        "  --depth N            max bounces (default 8)\n"
        "  --tall MATERIAL      tall block: diffuse, mirror or glass (default diffuse)\n"
        "  --short MATERIAL     short block: diffuse, mirror or glass (default diffuse)\n"
        "  --model FILE         load an .obj or .stl into the box (replaces the blocks)\n"
        "  --model-material M   model material: diffuse, mirror or glass (default diffuse)\n"
        "  --zup                the model file is Z-up (common for STL)\n"
        "  --rotate DEG         turn the model around the vertical axis\n"
        "  --tilt-x DEG         tilt the model around X (e.g. 90 or -90 to stand it up)\n"
        "  --tilt-z DEG         tilt the model around Z\n"
        "  --blocks             keep the two blocks when a model is loaded\n"
        "  --exposure X         exposure multiplier for the PNGs (default 1.0)\n"
        "  --clamp X            caustic firefly clamp, 0 = off (default 4)\n"
        "  --background X       grey background light behind the open front of the box (default 0)\n"
        "  --out PATH           output path prefix (default renders/cornell)\n"
        "  --no-denoise         skip the AI denoiser\n"
        "  --hdr                also write full-range .pfm images\n"
        "  --ptx FILE           path to pathtracer.ptx (default: next to moa.exe)\n"
        "  --verbose            print OptiX diagnostic messages\n"
        "  --help               show this help\n",
        kVersion);
}

unsigned int parseUInt(const char* flag, const char* value)
{
    char* end = nullptr;
    const unsigned long v = std::strtoul(value, &end, 10);
    if (!end || *end != '\0' || v == 0 || v > 1000000)
        throw std::runtime_error(std::string("Invalid value for ") + flag + ": " + value);
    return static_cast<unsigned int>(v);
}

bool parseArgs(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value after " + a);
            return argv[++i];
        };
        if (a == "--help" || a == "-h") { printUsage(); return false; }
        else if (a == "--width") o.render.width = parseUInt("--width", next());
        else if (a == "--height") o.render.height = parseUInt("--height", next());
        else if (a == "--spp") o.render.samplesPerPixel = parseUInt("--spp", next());
        else if (a == "--spl") o.render.samplesPerLaunch = parseUInt("--spl", next());
        else if (a == "--depth") o.render.maxDepth = parseUInt("--depth", next());
        else if (a == "--out") o.out = next();
        else if (a == "--ptx") o.ptx = next();
        else if (a == "--exposure") o.exposure = std::strtof(next(), nullptr);
        else if (a == "--clamp") o.render.maxSampleValue = std::strtof(next(), nullptr);
        else if (a == "--background") o.render.background = std::strtof(next(), nullptr);
        else if (a == "--no-denoise") o.denoise = false;
        else if (a == "--hdr") o.hdr = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--batch") o.batch = true;
        else if (a == "--model") o.model = next();
        else if (a == "--zup") o.scene.modelZUp = true;
        else if (a == "--blocks") o.scene.showBlocks = true;
        else if (a == "--rotate") o.scene.modelYaw = std::strtof(next(), nullptr);
        else if (a == "--tilt-x") o.scene.rotateModel(0, std::strtof(next(), nullptr));
        else if (a == "--tilt-z") o.scene.rotateModel(2, std::strtof(next(), nullptr));
        else if (a == "--model-material")
        {
            const std::string name = next();
            if (!moa::parseMaterialType(name, o.scene.modelMaterial))
                throw std::runtime_error("Unknown material '" + name + "' (use diffuse, mirror or glass)");
        }
        else if (a == "--tall" || a == "--short")
        {
            const std::string name = next();
            moa::MaterialType& target = (a == "--tall") ? o.scene.tallBlock : o.scene.shortBlock;
            if (!moa::parseMaterialType(name, target))
                throw std::runtime_error("Unknown material '" + name + "' (use diffuse, mirror or glass)");
        }
        else
            throw std::runtime_error("Unknown option: " + a + " (try --help)");
    }
    return true;
}

fs::path executableDir()
{
#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH)
        return fs::path(buf).parent_path();
#elif defined(__linux__)
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0)
    {
        buf[n] = '\0';
        return fs::path(buf).parent_path();
    }
#endif
    return fs::current_path();
}

void save(const std::string& path, bool ok)
{
    if (ok)
        std::printf("  wrote %s\n", path.c_str());
    else
        std::fprintf(stderr, "  FAILED to write %s\n", path.c_str());
}

// Adds a plain-English hint for the errors people most often hit on a first build.
void printHint(const std::string& msg)
{
    if (msg.find("OPTIX_ERROR_UNSUPPORTED_ABI_VERSION") != std::string::npos ||
        msg.find("OPTIX_ERROR_LIBRARY_NOT_FOUND") != std::string::npos ||
        msg.find("OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND") != std::string::npos)
        std::fprintf(stderr, "\nHint: your NVIDIA driver is too old for OptiX 9.1. Install driver R590 or newer.\n");
    else if (msg.find("cudaErrorInsufficientDriver") != std::string::npos)
        std::fprintf(stderr, "\nHint: the NVIDIA driver is older than the CUDA Toolkit. Update the driver.\n");
    else if (msg.find("cudaErrorNoDevice") != std::string::npos)
        std::fprintf(stderr, "\nHint: no NVIDIA GPU was found. Check the driver is installed.\n");
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        Options o;
        if (!parseArgs(argc, argv, o))
            return 0;

        const std::string ptxPath = o.ptx.empty() ? (executableDir() / "pathtracer.ptx").string() : o.ptx;

        std::printf("MOA Renderer %s\n", kVersion);
        moa::Renderer renderer(ptxPath, o.verbose);
        std::printf("GPU: %s\n", renderer.deviceName().c_str());

        if (!o.batch)
        {
            moa::LiveOptions live;
            live.render = o.render;
            live.scene = o.scene;
            live.exposure = o.exposure;
            live.outPrefix = o.out == "renders/cornell" ? "renders/live" : o.out;
            live.modelPath = o.model;
            std::printf("Opening the live render window...\n");
            return moa::runLiveView(renderer, live);
        }

        moa::Mesh model;
        if (!o.model.empty())
        {
            std::string error;
            std::printf("Loading %s ...\n", o.model.c_str());
            if (!moa::loadMesh(o.model, model, error))
                throw std::runtime_error(error);
            std::printf("Model: %zu triangles\n", model.triangleCount());
            o.scene.model = &model;
        }
        const moa::Scene scene = moa::makeCornellBox(o.scene);
        renderer.setScene(scene);
        std::printf("Scene: Cornell box, %zu triangles\n", scene.triangleCount());

        std::printf("Rendering %ux%u, %u spp, max %u bounces\n", o.render.width, o.render.height,
                    o.render.samplesPerPixel, o.render.maxDepth);
        int lastPercent = -1;
        const double renderMs = renderer.render(o.render, [&](unsigned int done, unsigned int total) {
            const int percent = static_cast<int>(100.0 * done / total);
            if (percent / 10 != lastPercent / 10 || done == total)
            {
                std::printf("  %3d%%  (%u / %u spp)\n", percent, done, total);
                std::fflush(stdout);
                lastPercent = percent;
            }
        });
        const double pixels = static_cast<double>(o.render.width) * o.render.height;
        const double msps = pixels * renderer.samplesRendered() / (renderMs * 1000.0);
        std::printf("Render time: %.2f s  (%.1f million samples/s)\n", renderMs / 1000.0, msps);

        double denoiseMs = 0.0;
        if (o.denoise)
        {
            denoiseMs = renderer.denoise();
            std::printf("Denoise time: %.1f ms\n", denoiseMs);
        }

        const fs::path outPrefix(o.out);
        if (outPrefix.has_parent_path())
            fs::create_directories(outPrefix.parent_path());

        std::vector<float4> img;
        const unsigned int w = o.render.width, h = o.render.height;
        renderer.downloadBeauty(img);
        save(o.out + "_beauty.png", moa::writePng(o.out + "_beauty.png", w, h, img, moa::PngEncoding::Beauty, o.exposure));
        if (o.hdr)
            save(o.out + "_beauty.pfm", moa::writePfm(o.out + "_beauty.pfm", w, h, img));

        if (o.denoise)
        {
            renderer.downloadDenoised(img);
            save(o.out + "_denoised.png",
                 moa::writePng(o.out + "_denoised.png", w, h, img, moa::PngEncoding::Beauty, o.exposure));
            if (o.hdr)
                save(o.out + "_denoised.pfm", moa::writePfm(o.out + "_denoised.pfm", w, h, img));
        }

        renderer.downloadAlbedo(img);
        save(o.out + "_albedo.png", moa::writePng(o.out + "_albedo.png", w, h, img, moa::PngEncoding::Albedo));
        renderer.downloadNormal(img);
        save(o.out + "_normal.png", moa::writePng(o.out + "_normal.png", w, h, img, moa::PngEncoding::Normal));

        std::printf("Done.\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\nERROR: %s\n", e.what());
        printHint(e.what());
        return 1;
    }
}
