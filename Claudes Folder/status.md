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
- Mirror shows black where it reflects the open box front (Background 0). HDRI lighting will fix.

**Milestone 3 (OBJ/STL loading): works on Drake's PC** (uffizi torso, 2,451,630 tris; first build took
8.5 s to load with strtof, now from_chars: 0.5 s for a 2.4M-tri test file here). Models load in
arbitrary orientations, so the model panel now has stacking world-axis 90 degree buttons (X/Y/Z, minus/plus;
accumulated 3x3 matrix in CornellOptions::rotateModel, exact quarter turns), a fine Spin slider, Scale,
and camera-relative move sliders (Left/right = -X, Down/up = Y, Near/far = Z). Rebuild happens on
slider release. Tested OK on Drake's PC: torso upright at 3840x2160, load 0.5 s, 232 Msamples/s,
denoise 40.9 ms. Later: instance transforms (IAS) so moving or rotating
doesn't rebuild the BVH.
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

**Milestone 4 (textures): works on Drake's PC** (bust textured, bump 0-3, mirror, glass, Save images OK).
Added after: Reset bump button + right-click any slider to reset it (untested).
OBJ vt parsing (unique v/vt/vn vertices; smooth normals computed per original position so UV seams
don't show), tangents, .mtl map_Kd / map_Bump|bump|norm, plus auto-discovery of images named like the
model with normal maps detected by colour (looksLikeNormalMap). GPU: CUDA texture objects (RGBA8,
linear filter, wrap, sRGB decode for colour), cached per image across scene rebuilds. Material gets
colorTexture/normalTexture/flags/normalStrength; V flipped by default (image row 0 = top).
UI: texture names, Use textures, Flip V, Flip green, Bump strength (material-only update), drop images.
CPU mock: tex2D mock + BVH (60k-tri bust in ~12 s). The bust loads upright, facing sideways.
No mipmaps yet (8K textures may shimmer while moving at low spp).

**Milestone 5 (HDRI lighting): works on Drake's PC** (Poly Haven goegap_4k, meadow_2_4k,
rogland_clear_night_4k; ~380 Msamples/s clay, ~1000 Msamples/s glass at 3245x2120). White blotches on
glass under the hard desert sun = refracted sun glints, made worse by the normal map on glass and by the
denoiser at low spp; fine with softer HDRIs or bump 0. Ideas: max-brightness clamp for specular env
hits; rough glass/metal (milestone 6).
.hdr via stb_image (EXR not yet). Float4 CUDA texture (wrap U, clamp V). Importance sampling: per-pixel
luminance x sin(theta), marginal/conditional CDFs (buildEnvironmentTables, shared with the CPU mock),
NEE towards the env on diffuse hits + MIS (power heuristic) with cosine bounce pdf; misses after
mirror/glass add env fully; camera rays see it if "Show background". HDRI mode = open 60-unit grey
floor, no walls or quad light (hasQuadLight=0); blocks/model stay. Intensity, rotation and visibility
update without rebuild (setEnvironmentParams). envWidth>0 is the "has env" flag (not envTexture != 0).
UI: Lighting section, drop .hdr. CLI: --hdri, --hdri-intensity, --hdri-rotate, --no-floor.
Drake is downloading Poly Haven HDRIs into an `hdri\` folder.

## NEXT SESSION: Houdini first (Drake's decision, 2026-10-16 05:34)

Drake wants MOA running inside Houdini before more materials/lights. Plan:
1. Split the engine out of moa.exe into a library (moa_core) shared by the live window and plugins.
2. Hydra render delegate (HdMoa) built against Houdini FX 20.5.445's own USD (HDK in
   `C:\Program Files\Side Effects Software\Houdini 20.5.445\toolkit`; check which USD/Python/MSVC
   versions it ships). Goal: "MOA" in the Solaris viewport renderer dropdown, live progressive render.
3. Render settings exposed on Houdini's Render Settings LOP; final renders via husk.
4. Materials: read Houdini's MaterialX nodes (standard_surface) + UsdPreviewSurface first, so Drake
   builds shaders with existing Houdini nodes. Then milestone 6 materials (roughness/metal/coat) and
   more lights, driven from Houdini.
No Houdini in the cloud workspace: testing happens on Drake's PC (him or computer use).

Also wanted (added to the design doc, "Batch rendering and render farms"): batch render inside each
plugin, husk support, and a headless `moa-render` (USD + frame range -> multi-layer EXR, no window/host
licence) with farm-manager support (Deadline/OpenCue/Tractor/HQueue/TOPs), frame splitting across PCs,
later one-frame splitting by averaging different-seed renders, Linux nodes. Design the engine library
split so it runs headless from day one (no GLFW/ImGui dependency in moa_core).
Hydra delegate code can be compile-checked here against pip `usd-core` headers if they match.

## Business model: OPEN CORE (Drake's decision, 2026-10-16 21:10)

- Public repo `moa-renderer` (Apache 2.0): render engine library + standalone MOA Viewer (OBJ/STL/HDRI).
- Private repo (proposed name `moa-studio`, Drake must create it): Hydra delegate + Houdini/Maya/
  Blender/C4D/Unreal plugins, USD scene loading, `moa-render` farm renderer, AI toolkit, licence keys,
  one Windows installer (Inno Setup; finds Houdini/Maya, writes Houdini package .json / Maya .mod).
  The private repo pulls the public engine in (git submodule) so engine fixes flow both ways.
- Pricing idea: free non-commercial (720p clean or 1080p watermark), $100 perpetual commercial with
  3 years of releases, unlimited free farm nodes with a commercial licence. Details in design doc
  "Licensing and funding".
- CONSEQUENCE FOR THE HOUDINI WORK: the engine library (moa_core) goes in the public repo with a clean
  C++ API; the Hydra delegate goes in the private repo. Don't put plugin/USD code in the public repo.

Built-in LAN farm (Drake, 21:12): `MoaRenderer-RenderfarmNode.exe` copied to spare PCs, double-click
installs moa-render + background service, driver check, firewall rule, LAN discovery (mDNS), pairing code,
main PC farm panel sends frames + scene files, collects EXRs, re-queues dropped frames. In design doc.
Installers ask for a render output folder (local or UNC share like \\server\projects); nodes write
there directly and can read scenes from it; test-write on install; UNC not mapped drives (services
can't see mapped drives); per-job override.

Licensing/website (Drake, 21:21): moastudios.com (empty, no host yet) with Products page + MOA reel.
Plan: merchant-of-record shop (Lemon Squeezy/Paddle/Gumroad) issues keys; activation returns a licence
file signed with Moa Studios' private key (owner, updates-until = purchase + 3 years); app verifies
offline with embedded public key (works forever offline); updater and installer refuse releases dated
after updates-until. Static site on free hosting (Cloudflare Pages/Netlify/GitHub Pages), not home PC.

MOA Bridge (Drake, 21:31): one-click scene transfer between hosts (Houdini -> Maya first): bake geo/sims
to USD/Alembic, rebuild MOA material networks node-for-node and MOA lights/cameras natively, copy +
relink textures, convert units. DESIGN RULE NOW: every MOA shader/light node is defined ONCE in a shared
node registry (name, inputs, types, defaults, UI hints) that each host plugin generates its nodes from;
materials/lights serialise to USD (UsdShade/UsdLux + moa: attributes). Apply this when building the
Houdini nodes. Paid feature; needs the Maya plugin, so after v1 (Houdini-only launch).
UPDATE 21:41: host-native nodes (Houdini VOP/MaterialX utilities, Maya ramp/blendColors/etc.) MUST work
inside MOA networks (Redshift strength). Bridge Translator: report green=MOA, blue=built-in match list,
yellow=AI-proposed equivalent (artist approves), red=bake to texture or swap; preview-ball render compare
both versions; MaterialX as middle language; AI local or BYO key.
Bridge rules (from Drake's Redshift 2018 rat-fur pain: Houdini RS network had to be rebuilt in Maya
from Maya-native ramp/blendColors/gammaCorrect/place2dTexture nodes; same-named nodes behaved
differently): MOA-only nodes incl. ramps/mixers/colour-correct/hair-position/random-per-strand;
node behaviour lives in the engine, never in a plugin; same params/defaults, unit conversion on
export, per-texture colour space; keep node names/positions/network boxes/colours; CI renders the
same test scenes through every host and requires pixel-identical output.

## Blockers / waiting on

- None. Drake's folder is now a git clone tracking `origin/main`: deliver changes by pushing to
  GitHub and having him run `git pull`. **Don't copy files into his folder through the file bridge
  any more** (it makes git see local changes, and it altered JPGs once).

## Known limitations (by design for milestone 1)

- Scene is hard-coded (Cornell box). No file loading yet.
- One rectangular area light; no environment map (only a uniform grey `--background`).
- No caustics through glass (NEE shadow rays are blocked by glass); firefly clamp hides caustic noise.
- Flat-shaded triangles only; no textures, no smooth normals.
