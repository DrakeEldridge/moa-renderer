# Status

_Last updated: 2026-10-16 (session 01)_

## Where we are

GitHub: first commit pushed to `main` on 2026-10-16 (Claude GitHub App installed).


**Phase 1, milestone 1: DONE and verified on Drake's RTX 3080 (2026-10-16).**

First build succeeded with no errors (one harmless MSVC warning, since fixed). Render result:
1024x1024, 256 spp, 8 bounces in **1.74 s** (about 155 million samples/s); OptiX denoise **41.6 ms**.
The GPU image matches the CPU mock exactly.

The command-line renderer `moa.exe` path-traces a built-in Cornell box with OptiX 9.1 and writes
beauty, denoised, albedo and normal PNGs. In the cloud workspace, device and host code compile
cleanly, and the CPU mock renders correct images (diffuse, mirror and glass variants).

## Next steps

**Milestone 2: live render window (Drake chose this on 2026-10-16).** Goal: the Redshift
RenderView feel. Open a window, watch the image clean up progressively, move the camera and see
it restart instantly.

Plan:
1. Window + display: GLFW and Dear ImGui fetched with CMake FetchContent from GitHub (no vcpkg yet).
   OpenGL texture blit of the beauty buffer (start with a host copy and glTexSubImage2D; CUDA-GL
   interop later if needed).
2. Render loop: one OptiX launch per frame (small spl), accumulate, display every frame. Reset
   accumulation on any change. Denoise the displayed frame every N frames or once the camera stops.
3. Camera: Houdini-style controls (Alt or Space + LMB tumble, MMB pan, RMB / wheel dolly). Also
   accept Maya-style Alt combos. Orbit around a pivot (default: scene centre).
4. Fast interaction: quarter resolution while dragging, full resolution when still (later: AI 2x upscale).
5. ImGui panel (Redshift-like): samples target, bounces, clamp, exposure, background, denoise
   toggle, AOV switcher (beauty / denoised / albedo / normal), block material dropdowns, render
   stats (spp, samples/s), "Save image" button.
6. Keep batch mode: `moa.exe --batch ...` = today's command-line render (run.bat uses it).
7. Test via computer use: Claude runs build.bat, then the window. Computer use is click-only on
   terminals; the MOA window itself may need its own approval to drag the camera.

After that: OBJ loading, then USD/Alembic (Phase 2 groundwork), then EXR and more AOVs.

## Blockers / waiting on

- None. Drake's folder is now a git clone tracking `origin/main`: deliver changes by pushing to
  GitHub and having him run `git pull`. **Don't copy files into his folder through the file bridge
  any more** (it makes git see local changes, and it altered JPGs once).

## Known limitations (by design for milestone 1)

- Scene is hard-coded (Cornell box). No file loading yet.
- One rectangular area light; no environment map (only a uniform grey `--background`).
- No caustics through glass (NEE shadow rays are blocked by glass); firefly clamp hides caustic noise.
- Flat-shaded triangles only; no textures, no smooth normals.
