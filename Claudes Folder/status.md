# Status

_Last updated: 2026-10-16 (session 01)_

## Where we are

GitHub: first commit pushed to `main` on 2026-10-16 (Claude GitHub App installed).


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

- Drake's local folder holds the same files but isn't a git repo yet. Steps sent 2026-10-16:
  `git init -b main`, `git remote add origin ...`, `git fetch origin`, `git reset origin/main`,
  `git branch -u origin/main`. Confirm `git status` is clean next session.

## Known limitations (by design for milestone 1)

- Scene is hard-coded (Cornell box). No file loading yet.
- One rectangular area light; no environment map (only a uniform grey `--background`).
- No caustics through glass (NEE shadow rays are blocked by glass); firefly clamp hides caustic noise.
- Flat-shaded triangles only; no textures, no smooth normals.
