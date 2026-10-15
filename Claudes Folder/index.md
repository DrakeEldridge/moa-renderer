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
