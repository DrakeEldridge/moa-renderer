# MOA Renderer: instructions for Claude

The open-source (Apache 2.0) GPU path tracer by Moa Studios: the render engine library
(`moa_core`) and the `moa.exe` test window. Drake is a 3D/VFX artist (Houdini, Maya, Unreal), not a
programmer. Claude writes the code and Drake builds and runs it on his PC.

## Before doing anything

Project memory, test tools and the paid plugins live in Drake's private workspace repo
(`DrakeEldridge/moa-studios-renderer`). This repo is cloned inside it as `MOA Renderer/`.
Follow the workspace's `CLAUDE.md` and read its `Claudes Folder/START_HERE.md` and `status.md`.
If only this public repo is available, say so and ask Drake for access to the workspace repo.

## Rules for this repo

- Engine only: no plugin, USD-loading, licensing or business material here (that's private).
- Code must build on Windows with MSVC + CUDA (Drake's machine). Keep `LaunchParams` identical
  on host and device (check `sizeof` after changing it).
- Explain things to Drake in artist terms. Give exact steps for anything he has to do.
