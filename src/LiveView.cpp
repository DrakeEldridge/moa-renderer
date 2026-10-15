// MOA Renderer - interactive render window.
//
// Each frame: handle input, run as many render passes as fit in a small time budget, denoise
// at growing sample counts, then show the chosen pass. Any change (camera, settings, window size)
// restarts the progressive render, which is what gives the "noisy for a moment, then clears up"
// feel of Redshift's RenderView.
// SPDX-License-Identifier: Apache-2.0
#include "LiveView.h"

#include "ImageIO.h"
#include "device/vec_math.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>  // also brings in the system OpenGL header

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace moa {

namespace {

constexpr float kPanelWidth = 340.0f;  // at UI scale 1

using Clock = std::chrono::steady_clock;

double msBetween(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Camera that orbits a pivot point: tumble changes yaw/pitch, pan moves the pivot, dolly changes distance.
struct OrbitCamera
{
    float3 pivot = make_float3(0.0f, 0.0f, 0.0f);
    float yaw = 0.0f;
    float pitch = 0.0f;
    float distance = 1.0f;
    float fov = 40.0f;

    void set(const CameraDesc& d, float3 pivotPoint)
    {
        pivot = pivotPoint;
        const float3 offset = d.eye - pivot;
        distance = std::max(1e-3f, length(offset));
        yaw = std::atan2(offset.x, offset.z);
        pitch = std::asin(clampf(offset.y / distance, -1.0f, 1.0f));
        fov = d.verticalFovDegrees;
    }

    float3 eye() const
    {
        const float3 dir = make_float3(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw));
        return pivot + dir * distance;
    }

    CameraDesc desc() const
    {
        CameraDesc d;
        d.eye = eye();
        d.lookAt = pivot;
        d.up = make_float3(0.0f, 1.0f, 0.0f);
        d.verticalFovDegrees = fov;
        return d;
    }

    void tumble(float dx, float dy)
    {
        yaw -= dx * 0.005f;
        pitch = clampf(pitch + dy * 0.005f, -1.55f, 1.55f);
    }

    void pan(float dx, float dy)
    {
        const float3 forward = normalize(pivot - eye());
        const float3 right = normalize(cross(forward, make_float3(0.0f, 1.0f, 0.0f)));
        const float3 up = cross(right, forward);
        const float speed = distance * 0.0015f;
        pivot = pivot + (right * -dx + up * dy) * speed;
    }

    void dolly(float amount) { distance = std::max(0.05f, distance * std::exp(amount)); }
};

float3 sceneCenter(const Scene& scene)
{
    float3 lo = scene.vertices.front(), hi = scene.vertices.front();
    for (const float3& v : scene.vertices)
    {
        lo = make_float3(std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z));
        hi = make_float3(std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z));
    }
    return (lo + hi) * 0.5f;
}

// Owns GLFW + ImGui setup so everything is cleaned up even if rendering throws.
struct WindowContext
{
    GLFWwindow* window = nullptr;
    GLuint texture = 0;
    float uiScale = 1.0f;
    ImGuiStyle baseStyle;

    WindowContext()
    {
        glfwSetErrorCallback([](int code, const char* msg) { std::fprintf(stderr, "[GLFW %d] %s\n", code, msg); });
        if (!glfwInit())
            throw std::runtime_error("Couldn't initialise GLFW (window system)");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        // Pick a UI scale: Windows display scaling if set, otherwise based on monitor size
        // (big 4K screens at 100% scaling otherwise get tiny text).
        if (GLFWmonitor* monitor = glfwGetPrimaryMonitor())
        {
            float xs = 1.0f, ys = 1.0f;
            glfwGetMonitorContentScale(monitor, &xs, &ys);
            uiScale = xs;
            if (const GLFWvidmode* mode = glfwGetVideoMode(monitor); mode && xs <= 1.0f)
                uiScale = std::max(1.0f, mode->height / 1080.0f);
        }
        uiScale = clampf(uiScale, 1.0f, 3.0f);
        window = glfwCreateWindow(static_cast<int>(1600 * uiScale), static_cast<int>(960 * uiScale), "MOA Renderer",
                                  nullptr, nullptr);
        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("Couldn't create the render window (OpenGL 3.0 needed)");
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;  // don't litter imgui.ini files
        ImGui::StyleColorsDark();
        baseStyle = ImGui::GetStyle();
        applyUiScale(uiScale);
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void applyUiScale(float scale)
    {
        uiScale = scale;
        ImGui::GetStyle() = baseStyle;
        ImGui::GetStyle().ScaleAllSizes(scale);
        ImGui::GetIO().FontGlobalScale = scale;
    }

    ~WindowContext()
    {
        if (texture)
            glDeleteTextures(1, &texture);
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    WindowContext(const WindowContext&) = delete;
    WindowContext& operator=(const WindowContext&) = delete;
};

enum ViewMode : int { VIEW_BEAUTY = 0, VIEW_DENOISED, VIEW_ALBEDO, VIEW_NORMAL };
const char* const kViewNames[] = {"Beauty (raw)", "Denoised", "Albedo", "Normal"};
const char* const kMaterialNames[] = {"Diffuse", "Mirror", "Glass"};

}  // namespace

int runLiveView(Renderer& renderer, const LiveOptions& options)
{
    WindowContext ctx;

    RenderSettings settings = options.render;
    settings.samplesPerLaunch = std::min(settings.samplesPerLaunch, 2u);  // small passes keep the UI responsive
    CornellOptions sceneOptions = options.scene;
    float exposure = options.exposure;
    int view = VIEW_DENOISED;
    bool denoiseEnabled = true;
    bool denoiseWhileMoving = true;
    int targetSpp = static_cast<int>(settings.samplesPerPixel);
    int samplesPerPass = static_cast<int>(settings.samplesPerLaunch);
    int maxDepth = static_cast<int>(settings.maxDepth);
    int tallMaterial = static_cast<int>(sceneOptions.tallBlock);
    int shortMaterial = static_cast<int>(sceneOptions.shortBlock);

    Scene scene = makeCornellBox(sceneOptions);
    renderer.setScene(scene);
    OrbitCamera cam;
    const float3 pivot = sceneCenter(scene);
    cam.set(scene.camera, pivot);
    const OrbitCamera homeCam = cam;

    bool restart = true;
    unsigned int nextDenoise = 1;
    bool denoisedValid = false;
    bool finalDenoiseDone = false;
    bool displayDirty = true;
    unsigned int shownSamples = ~0u;
    double lastDenoiseMs = 0.0;
    double passMsTotal = 0.0;
    double samplesTotal = 0.0;
    int dragButton = -1;
    std::string saveMessage;

    std::vector<float4> pixels;
    std::vector<uint8_t> rgba;

    while (!glfwWindowShouldClose(ctx.window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuiIO& io = ImGui::GetIO();

        // Layout: render view on the left, settings panel on the right.
        const float panelWidth = kPanelWidth * ctx.uiScale;
        const float viewW = std::max(64.0f, io.DisplaySize.x - panelWidth);
        const float viewH = std::max(64.0f, io.DisplaySize.y);
        const unsigned int renderW = static_cast<unsigned int>(viewW * io.DisplayFramebufferScale.x);
        const unsigned int renderH = static_cast<unsigned int>(viewH * io.DisplayFramebufferScale.y);
        if (renderW != renderer.width() || renderH != renderer.height())
            restart = true;

        // --- Camera input on the render view ---
        bool moving = false;
        const ImVec2 mouse = io.MousePos;
        const bool overView = mouse.x >= 0.0f && mouse.x < viewW && mouse.y >= 0.0f && mouse.y < viewH;
        if (dragButton < 0 && !io.WantCaptureMouse && overView)
        {
            for (int b = 0; b < 3; ++b)
                if (ImGui::IsMouseClicked(b))
                    dragButton = b;
        }
        if (dragButton >= 0)
        {
            if (!ImGui::IsMouseDown(dragButton))
                dragButton = -1;
            else if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f)
            {
                const float dx = io.MouseDelta.x, dy = io.MouseDelta.y;
                if (dragButton == 0)  // ImGui buttons: 0 left, 1 right, 2 middle
                    cam.tumble(dx, dy);
                else if (dragButton == 2)
                    cam.pan(dx, dy);
                else
                    cam.dolly((dx + dy) * -0.01f);
                restart = true;
                moving = true;
            }
        }
        if (!io.WantCaptureMouse && overView && io.MouseWheel != 0.0f)
        {
            cam.dolly(-io.MouseWheel * 0.1f);
            restart = true;
            moving = true;
        }
        if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_F))
        {
            cam = homeCam;
            restart = true;
        }

        // --- Settings panel ---
        ImGui::SetNextWindowPos(ImVec2(viewW, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(panelWidth, viewH));
        ImGui::Begin("MOA Renderer", nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

        const unsigned int samples = renderer.samplesRendered();
        ImGui::TextDisabled("%s", renderer.deviceName().c_str());
        ImGui::Text("%u x %u", renderer.width(), renderer.height());
        const float progress = targetSpp > 0 ? std::min(1.0f, static_cast<float>(samples) / targetSpp) : 1.0f;
        char progressText[64];
        std::snprintf(progressText, sizeof(progressText), "%u / %d spp", samples, targetSpp);
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), progressText);
        const double msps = passMsTotal > 0.0 ? samplesTotal / (passMsTotal * 1000.0) : 0.0;
        ImGui::Text("%.0f million samples/s", msps);
        ImGui::Text("Denoise: %.1f ms", lastDenoiseMs);

        ImGui::SeparatorText("View");
        if (ImGui::Combo("Show", &view, kViewNames, IM_ARRAYSIZE(kViewNames)))
            displayDirty = true;
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
            displayDirty = true;
        if (ImGui::Checkbox("AI denoiser", &denoiseEnabled))
        {
            restart = true;
            if (!denoiseEnabled && view == VIEW_DENOISED)
                view = VIEW_BEAUTY;
        }
        ImGui::Checkbox("Denoise while moving", &denoiseWhileMoving);

        ImGui::SeparatorText("Sampling");
        if (ImGui::SliderInt("Samples", &targetSpp, 1, 8192, "%d", ImGuiSliderFlags_Logarithmic))
            finalDenoiseDone = false;  // raising the target just continues rendering
        if (ImGui::SliderInt("Per pass", &samplesPerPass, 1, 16))
            restart = true;
        if (ImGui::SliderInt("Bounces", &maxDepth, 1, 32))
            restart = true;
        if (ImGui::SliderFloat("Caustic clamp", &settings.maxSampleValue, 0.0f, 20.0f, "%.1f"))
            restart = true;
        if (ImGui::SliderFloat("Background", &settings.background, 0.0f, 2.0f, "%.2f"))
            restart = true;

        ImGui::SeparatorText("Scene");
        const bool tallChanged = ImGui::Combo("Tall block", &tallMaterial, kMaterialNames, IM_ARRAYSIZE(kMaterialNames));
        const bool shortChanged = ImGui::Combo("Short block", &shortMaterial, kMaterialNames, IM_ARRAYSIZE(kMaterialNames));
        if (tallChanged || shortChanged)
        {
            sceneOptions.tallBlock = static_cast<MaterialType>(tallMaterial);
            sceneOptions.shortBlock = static_cast<MaterialType>(shortMaterial);
            scene = makeCornellBox(sceneOptions);
            renderer.setScene(scene);  // resets the renderer's camera; ours is reapplied on restart
            restart = true;
        }

        ImGui::SeparatorText("Camera");
        if (ImGui::SliderFloat("Field of view", &cam.fov, 10.0f, 100.0f, "%.1f deg"))
            restart = true;
        if (ImGui::Button("Reset camera (F)"))
        {
            cam = homeCam;
            restart = true;
        }

        ImGui::SeparatorText("Output");
        if (ImGui::Button("Save images"))
        {
            namespace fs = std::filesystem;
            const fs::path prefix(options.outPrefix);
            if (prefix.has_parent_path())
                fs::create_directories(prefix.parent_path());
            const unsigned int w = renderer.width(), h = renderer.height();
            bool ok = true;
            renderer.downloadBeauty(pixels);
            ok &= writePng(options.outPrefix + "_beauty.png", w, h, pixels, PngEncoding::Beauty, exposure);
            if (denoiseEnabled)
            {
                lastDenoiseMs = renderer.denoise();
                renderer.downloadDenoised(pixels);
                ok &= writePng(options.outPrefix + "_denoised.png", w, h, pixels, PngEncoding::Beauty, exposure);
            }
            renderer.downloadAlbedo(pixels);
            ok &= writePng(options.outPrefix + "_albedo.png", w, h, pixels, PngEncoding::Albedo);
            renderer.downloadNormal(pixels);
            ok &= writePng(options.outPrefix + "_normal.png", w, h, pixels, PngEncoding::Normal);
            saveMessage = ok ? "Saved to " + options.outPrefix + "_*.png" : "Couldn't write some images";
            displayDirty = true;
        }
        if (!saveMessage.empty())
            ImGui::TextWrapped("%s", saveMessage.c_str());

        ImGui::SeparatorText("Interface");
        float uiSize = ctx.uiScale;
        if (ImGui::SliderFloat("UI size", &uiSize, 1.0f, 3.0f, "%.2f"))
            ctx.applyUiScale(uiSize);

        ImGui::SeparatorText("Controls");
        ImGui::TextWrapped("Left drag: tumble\nMiddle drag: pan\nRight drag: dolly\nWheel: dolly\nF: reset camera");
        ImGui::End();

        // --- Rendering ---
        if (restart)
        {
            settings.width = renderW;
            settings.height = renderH;
            settings.samplesPerLaunch = static_cast<unsigned int>(samplesPerPass);
            settings.maxDepth = static_cast<unsigned int>(maxDepth);
            renderer.setCamera(cam.desc());
            renderer.beginProgressive(settings);
            restart = false;
            nextDenoise = 1;
            denoisedValid = false;
            finalDenoiseDone = false;
            displayDirty = true;
            passMsTotal = 0.0;
            samplesTotal = 0.0;
        }

        const auto frameStart = Clock::now();
        const double budgetMs = moving ? 8.0 : 25.0;
        while (renderer.samplesRendered() < static_cast<unsigned int>(targetSpp))
        {
            const auto t0 = Clock::now();
            renderer.renderPass();
            const auto t1 = Clock::now();
            passMsTotal += msBetween(t0, t1);
            samplesTotal += static_cast<double>(renderer.width()) * renderer.height() * settings.samplesPerLaunch;
            if (msBetween(frameStart, t1) > budgetMs)
                break;
        }

        const unsigned int now = renderer.samplesRendered();
        const bool done = now >= static_cast<unsigned int>(targetSpp);
        if (denoiseEnabled && now > 0 && (!moving || denoiseWhileMoving) &&
            (now >= nextDenoise || (done && !finalDenoiseDone)))
        {
            lastDenoiseMs = renderer.denoise();
            denoisedValid = true;
            nextDenoise = now < 16 ? now + 1 : now * 2;
            finalDenoiseDone = done;
            displayDirty = true;
        }

        // --- Show the chosen pass ---
        if (displayDirty || now != shownSamples)
        {
            PngEncoding encoding = PngEncoding::Beauty;
            if (view == VIEW_DENOISED && denoisedValid)
                renderer.downloadDenoised(pixels);
            else if (view == VIEW_ALBEDO)
            {
                renderer.downloadAlbedo(pixels);
                encoding = PngEncoding::Albedo;
            }
            else if (view == VIEW_NORMAL)
            {
                renderer.downloadNormal(pixels);
                encoding = PngEncoding::Normal;
            }
            else
                renderer.downloadBeauty(pixels);
            encodeRgba8(pixels, rgba, encoding, exposure);
            glBindTexture(GL_TEXTURE_2D, ctx.texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, static_cast<GLsizei>(renderer.width()),
                         static_cast<GLsizei>(renderer.height()), 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
            shownSamples = now;
            displayDirty = false;
        }

        // Rows are stored bottom-up, so flip the V coordinate when drawing.
        ImGui::GetBackgroundDrawList()->AddImage(static_cast<ImTextureID>(ctx.texture), ImVec2(0.0f, 0.0f),
                                                 ImVec2(viewW, viewH), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

        ImGui::Render();
        int fbW = 0, fbH = 0;
        glfwGetFramebufferSize(ctx.window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(ctx.window);
    }
    return 0;
}

}  // namespace moa
