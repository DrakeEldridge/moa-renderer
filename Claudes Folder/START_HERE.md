# Start here

Project memory for MOA Renderer. A new chat should read this file, then `status.md`,
then search `index.md` for anything related to the task at hand.

## Files in this folder

| File | Purpose |
| --- | --- |
| `START_HERE.md` | This file: how the memory system works, key facts and links |
| `status.md` | Short-term memory: where things are right now, next steps, blockers |
| `index.md` | Dated, tagged list of every change, fix, decision and failed attempt, with pointers |
| `logs/` | Full session logs, one file per session (`YYYY-MM-DD_topic.md`) |
| `tools/` | Claude's test harnesses (NVRTC compile check, CPU OptiX mock) |
| `ClaudeMemoryWorkflow.txt` | The earlier chat where this memory workflow was designed |
| `*.lnk` | Drake's shortcuts (CGI apps, Game DEV, Visual Studio). Not in git. |

Index tags: `FEATURE`, `FIX`, `BUG`, `DECISION`, `FAILED`, `BUILD`, `DOCS`, `SETUP`.

## Key links

- GitHub: https://github.com/DrakeEldridge/moa-renderer
- Design doc (features, roadmap, decisions): https://claude.ai/code/artifact/8bf26f6e-14ce-49b0-b4e8-3c5d4d2b9062
- Local project folder on Drake's PC: `I:\Documents\Projects\MOA Renderer` (this repo's root)

## Drake's setup

| Item | Details |
| --- | --- |
| GPU | NVIDIA RTX 3080 (10 GB VRAM, sm_86) |
| OS | Windows (x64) |
| Compiler | Visual Studio Community 2022 17.14 ("Desktop development with C++") |
| CUDA | Toolkit 13.x installed 2026-10-16 |
| Driver | Updated 2026-10-16 (needs R590+ for OptiX 9.1) |
| Houdini | FX 20.5.445 (has V-Ray) and FX 20.0.547 (has Redshift). Houdini is the primary host. |
| Other apps | Maya, Unreal, Nuke, Mari, ZBrush, Substance. Shortcuts: `C:\Users\Drake\Desktop\CGI`, `C:\Users\Drake\Desktop\Game DEV` |

## How Claude works on this project

- Claude's cloud workspace has **no GPU**. Drake builds with `build.bat` and runs `run.bat`,
  then sends back the console output and images.
- Device code (`src/device/*.cu`) is compile-checked here with NVRTC (`tools/nvrtc_check.py`).
- Host code is compile-checked here with g++ against the CUDA and OptiX headers.
- `tools/cpu_mock/` runs the real `pathtracer.cu` on the CPU through a fake OptiX API, so
  changes to the path tracer can be checked visually before Drake builds them.
- Code reaches Drake through GitHub: Claude pushes, Drake runs `git pull` in the project folder.
- Computer use works on Drake's PC (approved 2026-10-16) for File Explorer, Command Prompt and
  Terminal at "click" tier: Claude can double-click `build.bat` / `run.bat` and read the console,
  but can't type. Read render outputs with the file bridge (device_stage_files) rather than screenshots.
  Never click inside a running console window: Windows "Select" mode pauses it.
  The MOA window itself needs its own grant: resolve `i:\\documents\\projects\\moa renderer\\build\\release\\moa.exe` (tier full).

## Big decisions so far (details in the design doc)

- Clean-room: learn from public sources only, no decompiling commercial renderers.
- Stack: C++17 + CUDA + OptiX 9.1; USD/Hydra for host apps; MaterialX/OpenPBR materials.
- Open source, Apache 2.0, donations tied to milestones. No subscriptions.
- Houdini first, then Maya and Blender (Hydra), then Cinema 4D and Unreal.
- AI features run locally on the GPU; optional cloud AI uses the user's own API key.
