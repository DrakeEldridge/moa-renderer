# MOA Renderer: instructions for Claude

This is Drake's (Moa Studios) open-source GPU path tracer. Drake is a 3D/VFX artist
(Houdini, Maya, Unreal), not a programmer. Claude writes the code and Drake builds and runs
it on his PC.

## Before doing anything

1. Read `Claudes Folder/START_HERE.md`, then `Claudes Folder/status.md`.
2. Before fixing a bug or changing a system, search `Claudes Folder/index.md` (and the
   `logs/` folder) for its history, including approaches that FAILED.

## While working

- Code must build on Windows with MSVC + CUDA (Drake's machine). Claude's cloud workspace has no
  GPU: compile-check device code with NVRTC and host code with g++, and use the CPU OptiX
  mock (see `Claudes Folder/tools/`) to test path-tracing changes visually.
- Keep `LaunchParams` identical on host and device (check `sizeof` after changing it).
- Explain things to Drake in artist terms. Give exact steps for anything he has to do.

## Wrap-up (end of every session, or before the chat gets long)

1. Update `Claudes Folder/status.md` (current state, next steps, blockers).
2. Add entries to `Claudes Folder/index.md` (date | TAG | summary, plus files touched and log name).
3. Write or extend the session log in `Claudes Folder/logs/`.
4. Commit and push to GitHub (`DrakeEldridge/moa-renderer`). Git history is the backup.
