# Index

One line per change, fix, decision or failed attempt. Newest at the bottom.
Format: `date | TAG | summary` then `→ files / log` on the next line.

2026-10-16 | DECISION | Clean-room approach: learn from public docs/papers/open source, never decompile commercial renderers
                      → log: logs/2026-10-16_session-01.md
2026-10-16 | DECISION | Stack: C++17, CUDA, OptiX 9.1 (headers fetched from github.com/NVIDIA/optix-dev at configure time); Apache 2.0
                      → CMakeLists.txt, LICENSE
2026-10-16 | SETUP    | Project memory system created (START_HERE, status, index, logs, tools); CLAUDE.md at repo root
                      → Claudes Folder/*, CLAUDE.md
2026-10-16 | FEATURE  | Milestone 1 path tracer: raygen-loop design, NEE on one quad light, diffuse/mirror/glass, Russian roulette, progressive running average
                      → src/device/pathtracer.cu, src/device/LaunchParams.h
2026-10-16 | FEATURE  | OptiX AI denoiser (AOV model, albedo + world-space normal guides)
                      → src/Renderer.cpp (setupDenoiser, denoise)
2026-10-16 | FEATURE  | CLI: size, spp, spl, depth, block materials, exposure, clamp, background, --hdr (PFM), --no-denoise
                      → src/main.cpp
2026-10-16 | FEATURE  | PNG output with ACES-fitted tonemap + sRGB; PFM HDR output
                      → src/ImageIO.cpp
2026-10-16 | FIX      | Fireflies from mirror/glass caustics: clamp only emission found after a diffuse bounce (NEE never clamped), default 4
                      → src/device/pathtracer.cu (clampCaustic, afterDiffuse)
2026-10-16 | FAILED   | First clamp attempt clamped all contributions at depth>0 with value 10: still speckled, and it dimmed direct light seen in mirrors
                      → log: logs/2026-10-16_session-01.md
2026-10-16 | DECISION | Default scene uses diffuse blocks (classic look). Mirror block reflects the open front of the box as black unless --background > 0
                      → src/Scene.h, src/main.cpp
2026-10-16 | BUILD    | Cloud check pipeline: NVRTC compile, g++ host compile, sizeof(LaunchParams) match (208), CPU OptiX mock render
                      → Claudes Folder/tools/check.sh
2026-10-16 | DOCS     | README with build/run/troubleshooting; Moa Studios logo in README (light/dark)
                      → README.md, docs/images/
2026-10-16 | SETUP    | Claude GitHub App installed; first commit pushed to DrakeEldridge/moa-renderer main
                      → Claudes Folder/status.md
2026-10-16 | BUILD    | First Windows build OK (VS 2022 17.14, CUDA 13.4, CMake 4.4.4); first GPU render on RTX 3080: 1.74 s for 1024² @ 256 spp, denoise 41.6 ms
                      → logs/2026-10-16_session-01.md
2026-10-16 | FIX      | MSVC C4245 warning: cast OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE to unsigned int
                      → src/Renderer.cpp
2026-10-16 | BUG      | File-bridge copies of JPGs came out different from git's copies; fixed with `git restore docs/images`. Deliver via git only from now on
                      → Claudes Folder/status.md
2026-10-16 | SETUP    | Drake's folder linked to GitHub (git init + fetch + reset). Computer use approved (click tier) and used to run build.bat/run.bat
                      → Claudes Folder/START_HERE.md
2026-10-16 | DECISION | Milestone 2 = live render window (GLFW + ImGui via FetchContent, progressive display, Houdini-style camera, batch mode kept). Plan in status.md
                      → Claudes Folder/status.md
2026-10-16 | FEATURE  | Milestone 2 live window (GLFW 3.4 + ImGui 1.91.9b): progressive view, orbit camera, settings panel, save; batch via --batch. Works on Drake's PC (~147 Msamples/s live)
                      → src/LiveView.*, src/Renderer.* (beginProgressive/renderPass), src/ImageIO.* (encodeRgba8), CMakeLists.txt, live.bat
2026-10-16 | SETUP    | Computer use: moa.exe must be granted separately (resolve by full exe path, tier full) to see/drive the window
                      → Claudes Folder/START_HERE.md
2026-10-16 | FEATURE  | Live window UI scaling: auto (content scale, or monitor height/1080 for 4K at 100%) + "UI size" slider
                      → src/LiveView.cpp
2026-10-16 | FIX      | Live window laggy at 2x UI on 4K (≈4.7 MP render): added render scale (default 1/uiScale) + half res while dragging, full-res restart on release
                      → src/LiveView.cpp
2026-10-16 | FEATURE  | Resolution dropdown (window full/half/quarter, HD..4K, square, portrait, custom), replacing the render-scale slider
                      → src/LiveView.cpp
