// MOA Renderer - interactive render window (progressive preview with camera controls and settings).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Renderer.h"
#include "Scene.h"

#include <string>

namespace moa {

struct LiveOptions
{
    RenderSettings render;   // starting settings; width/height follow the window size
    CornellOptions scene;    // starting materials for the test scene
    float exposure = 1.0f;
    std::string outPrefix = "renders/live";  // where "Save images" writes
    std::string modelPath;                   // optional .obj/.stl to load at start
};

// Opens the window and runs until it is closed. Returns the process exit code.
int runLiveView(Renderer& renderer, const LiveOptions& options);

}  // namespace moa
