# Status

_Last updated: 2026-10-16 (session 01)_

## Where we are

**Phase 1, milestone 1: written, not yet built on Drake's PC.**

The command-line renderer `moa.exe` path-traces a built-in Cornell box with OptiX 9.1 and writes
beauty, denoised, albedo and normal PNGs. In the cloud workspace, device and host code compile
cleanly, and the CPU mock renders correct images (diffuse, mirror and glass variants).

## Next steps

1. **Drake:** install CMake 4.x (cmake.org, tick "Add to PATH") if not done yet.
2. **Drake:** run `build.bat`, then `run.bat`. Send back the console output and `renders\cornell_denoised.png`.
3. **Claude:** fix any Windows/MSVC build errors (first real build, so expect a few).
4. Milestone 2: EXR output (OpenEXR via vcpkg), interactive window (progressive view while
   rendering), then load a USD scene (start of Phase 2 groundwork).

## Blockers / waiting on

- Pushing to GitHub needs the Claude GitHub App installed on `DrakeEldridge/moa-renderer`
  (github.com/apps/claude/installations/select_target). Asked Drake on 2026-10-16.
- Drake's local folder is not a git repo yet. Once GitHub has the first commit, he can either
  clone it into the folder or run `git init` + `git remote add` + `git pull` (walk him through it).

## Known limitations (by design for milestone 1)

- Scene is hard-coded (Cornell box). No file loading yet.
- One rectangular area light; no environment map (only a uniform grey `--background`).
- No caustics through glass (NEE shadow rays are blocked by glass); firefly clamp hides caustic noise.
- Flat-shaded triangles only; no textures, no smooth normals.
