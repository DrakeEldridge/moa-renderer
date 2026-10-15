# Status

_Last updated: 2026-10-16 (session 01)_

## Where we are

GitHub: first commit pushed to `main` on 2026-10-16 (Claude GitHub App installed).


**Milestone 2 (live window) also done, see below.**

**Phase 1, milestone 1: DONE and verified on Drake's RTX 3080 (2026-10-16).**

First build succeeded with no errors (one harmless MSVC warning, since fixed). Render result:
1024x1024, 256 spp, 8 bounces in **1.74 s** (about 155 million samples/s); OptiX denoise **41.6 ms**.
The GPU image matches the CPU mock exactly.

The command-line renderer `moa.exe` path-traces a built-in Cornell box with OptiX 9.1 and writes
beauty, denoised, albedo and normal PNGs. In the cloud workspace, device and host code compile
cleanly, and the CPU mock renders correct images (diffuse, mirror and glass variants).

## Next steps

**Milestone 2 (live window): DONE and working on Drake's PC (2026-10-16).** Built first time;
Drake flew the camera himself. Live speed ~147 Msamples/s, denoise ~8 ms at window size.

Follow-ups for the live window:
- Panel text looks tiny on Drake's large TV monitor: add auto UI scale (glfwGetWindowContentScale)
  and a UI-size slider. (Asked Drake whether he wants it.)
- Quarter-resolution while dragging (not done yet), then AI 2x upscale.
- CUDA-GL interop instead of host copy if display becomes a bottleneck.

Then: OBJ loading, then USD/Alembic (Phase 2 groundwork), then EXR and more AOVs.

## Blockers / waiting on

- None. Drake's folder is now a git clone tracking `origin/main`: deliver changes by pushing to
  GitHub and having him run `git pull`. **Don't copy files into his folder through the file bridge
  any more** (it makes git see local changes, and it altered JPGs once).

## Known limitations (by design for milestone 1)

- Scene is hard-coded (Cornell box). No file loading yet.
- One rectangular area light; no environment map (only a uniform grey `--background`).
- No caustics through glass (NEE shadow rays are blocked by glass); firefly clamp hides caustic noise.
- Flat-shaded triangles only; no textures, no smooth normals.
