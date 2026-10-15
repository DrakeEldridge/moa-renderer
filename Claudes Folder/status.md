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
- UI scale added (auto from Windows scaling or monitor height/1080, plus "UI size" slider). Tested OK.
- Resolution dropdown: Window full / 1/2 / 1/4, fixed presets (HD, Full HD, QHD, 4K, squares,
  portrait) and Custom; fixed sizes letterboxed. Default Window 1/2 on big screens. Half res while
  dragging, added after Drake found the 2x-UI window laggy on his 4K TV. Tested OK ("works perfectly").
  Next speedups: AI 2x upscale for the low-res frames; CUDA-GL interop (the per-frame host copy +
  CPU encode scales with pixel count).
- CUDA-GL interop instead of host copy if display becomes a bottleneck.

**Milestone 3 (OBJ/STL loading): written and CPU-tested, not yet tested on Drake's PC.**
Indexed geometry with per-vertex shading normals; OBJ (polygons fan-triangulated, v//vn pairs,
negative indices, missing normals computed) and STL (binary + ASCII, vertices welded by exact
position, so seams with tiny float differences stay split). Model auto-fits onto the box floor
(height = Size, wide models limited to 1.15 x Size), blocks hidden unless "Show blocks". Live panel:
material, up axis, rotate/size (applied on slider release), remove; drag and drop onto the window.
CLI: --model, --model-material, --zup, --rotate, --blocks. Textures (vt + .mtl) not yet.

Test models are in Drake's `models\` folder (git-ignored):
- `uffizi-torso-5.obj` (93 MB, high-poly, no textures): first clay/mirror/glass test.
- `MCh_S_12_Rzezba_Popiersie_Rozy_Loewenfeld.OBJ` (3.8 MB, uppercase .OBJ extension) with
  `..._Loewenfeld.jpg` and `..._Loewenfeld_.png` (16 MB / 7.7 MB): texture step. Check which is colour
  and which is a normal/other map; there may be no .mtl, so allow choosing textures manually.
- `33 - Leeds University - Augustus.stl` (164 MB): STL is a trivial format, worth supporting alongside OBJ.
- `Classical_Healer_Bust_-_Trellis_2_vs_SupaVoxel.usdz`: wait for USD support (Phase 2).
Plan: first version auto-fits the model into the Cornell box (first version: auto-fit into the Cornell box, one clay/mirror/glass material, smooth normals; textures via .mtl after). Then USD/Alembic (Phase 2 groundwork), then EXR and more AOVs.

## Blockers / waiting on

- None. Drake's folder is now a git clone tracking `origin/main`: deliver changes by pushing to
  GitHub and having him run `git pull`. **Don't copy files into his folder through the file bridge
  any more** (it makes git see local changes, and it altered JPGs once).

## Known limitations (by design for milestone 1)

- Scene is hard-coded (Cornell box). No file loading yet.
- One rectangular area light; no environment map (only a uniform grey `--background`).
- No caustics through glass (NEE shadow rays are blocked by glass); firefly clamp hides caustic noise.
- Flat-shaded triangles only; no textures, no smooth normals.
