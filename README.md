<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/moa-studios-logo-white.jpg">
    <img alt="Moa Studios" src="docs/images/moa-studios-logo-black.jpg" width="420">
  </picture>
</p>

# MOA Renderer

By Moa Studios. A free, open-source GPU path tracer with built-in AI tools, aiming for the ease of use
Redshift had around 2018: fast interactive previews, simple settings and great shader nodes,
with plugins for Houdini (first), Maya, Blender and more via USD Hydra.

Design doc: https://claude.ai/code/artifact/8bf26f6e-14ce-49b0-b4e8-3c5d4d2b9062

**Status:** Phase 1, milestone 1. A command-line renderer that path-traces a built-in
Cornell box on an NVIDIA RTX GPU and denoises it with the OptiX AI denoiser.

## What it does right now

- GPU path tracing with NVIDIA OptiX 9.1 (hardware ray tracing on RTX cards)
- Global illumination (bounce light), soft shadows, next-event estimation for fast cleanup
- Materials: diffuse, mirror, glass (with Fresnel and refraction)
- Progressive rendering (the image improves with every pass)
- Render passes: beauty, albedo, normals
- OptiX AI denoiser, guided by the albedo and normal passes
- Firefly clamp for caustic speckles, optional grey background light

## Requirements (Windows)

| Tool | Version | Notes |
| --- | --- | --- |
| NVIDIA GPU | RTX 20-series or newer | Developed on an RTX 3080 |
| NVIDIA driver | R590 or newer | Needed by OptiX 9.1 |
| Visual Studio | 2022 (17.14+) or 2026 | Workload: "Desktop development with C++" |
| CUDA Toolkit | 12.9 or 13.x | Install **after** Visual Studio |
| CMake | 4.x | From cmake.org, added to PATH |
| Git | any | Used by CMake to download the OptiX headers |

You don't need to download the OptiX SDK: the build fetches the OptiX headers from
NVIDIA's GitHub automatically, and the OptiX runtime is part of the NVIDIA driver.

## Build

Double-click `build.bat`, or from a terminal in this folder:

```bat
cmake -S . -B build -A x64
cmake --build build --config Release
```

The first configure downloads the OptiX headers, so it needs internet access.
The result is `build\Release\moa.exe`, with `pathtracer.ptx` (the GPU programs) next to it.

## Run

Double-click `run.bat`, or:

```bat
build\Release\moa.exe --out renders\cornell
```

This writes four images to `renders\`:

| File | What it is |
| --- | --- |
| `cornell_beauty.png` | The raw path-traced render (some noise) |
| `cornell_denoised.png` | After the OptiX AI denoiser |
| `cornell_albedo.png` | Surface colour pass (denoiser guide) |
| `cornell_normal.png` | Surface direction pass (denoiser guide) |

Some things to try:

```bat
rem Mirror tall block and glass short block, with a grey background so the mirror has something to reflect
build\Release\moa.exe --tall mirror --short glass --background 0.3 --out renders\mirror_glass

rem Fewer samples to see how much the denoiser does on its own
build\Release\moa.exe --spp 16 --out renders\quick

rem High quality, also writing full-range HDR .pfm files for comp
build\Release\moa.exe --spp 2048 --spl 16 --hdr --out renders\final
```

Run `moa.exe --help` for every option.

## Project layout

```
CMakeLists.txt           build script
build.bat / run.bat      one-click build and test render (Windows)
src/main.cpp             command-line front end
src/Renderer.*           OptiX setup, scene upload, render loop, denoiser
src/Scene.*              scene description and the built-in Cornell box
src/ImageIO.*            PNG and PFM output
src/CudaBuffer.h         GPU memory helper
src/Check.h              CUDA/OptiX error checking
src/device/              code that runs on the GPU
  pathtracer.cu          ray generation, hit and miss programs (the path tracer itself)
  LaunchParams.h         data shared between CPU and GPU
  vec_math.h, random.h   maths and random-number helpers
third_party/stb/         stb_image_write (public domain / MIT)
Claudes Folder/          project memory: status, change index and session logs
```

## Troubleshooting

| Message | Fix |
| --- | --- |
| `No CUDA toolset found` during configure | CUDA was installed before Visual Studio. Re-run the CUDA installer (choose Custom and keep "Visual Studio Integration" ticked). |
| `OPTIX_ERROR_UNSUPPORTED_ABI_VERSION` | Update the NVIDIA driver to R590 or newer. |
| `cudaErrorInsufficientDriver` | The driver is older than the CUDA Toolkit. Update the driver. |
| `Couldn't open GPU program file ... pathtracer.ptx` | Run `moa.exe` from the build output folder, or pass `--ptx path\to\pathtracer.ptx`. |
| Configure fails downloading `optix-dev` | Check internet access, or point to a local SDK with `-DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0"`. |

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party code keeps its own licence
(stb_image_write: public domain / MIT). The OptiX headers are downloaded at build time
under NVIDIA's SDK licence and are not part of this repository.
