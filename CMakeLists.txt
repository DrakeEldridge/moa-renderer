# MOA Renderer
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)

project(MOARenderer VERSION 0.1.0 LANGUAGES C CXX CUDA)

# The live window (moa.exe) needs GLFW and Dear ImGui. Host plugins (Houdini etc.) only need
# the engine library, so they switch the viewer off when they add this project.
option(MOA_BUILD_VIEWER "Build the moa.exe live window and batch renderer" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
include(FetchContent)

# --- OptiX headers ----------------------------------------------------------------------
# OptiX is header-only at build time; the runtime ships inside the NVIDIA driver.
# By default the headers are downloaded from NVIDIA's GitHub. To use a local OptiX SDK
# instead, configure with -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0".
set(OPTIX_ROOT "" CACHE PATH "Optional path to a local OptiX SDK (folder containing include/optix.h)")
if(OPTIX_ROOT)
    set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")
else()
    FetchContent_Declare(optix_headers
        GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
        GIT_TAG v9.1.0
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(optix_headers)
    set(OPTIX_INCLUDE_DIR "${optix_headers_SOURCE_DIR}/include")
endif()
if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "optix.h not found in ${OPTIX_INCLUDE_DIR}")
endif()
message(STATUS "OptiX headers: ${OPTIX_INCLUDE_DIR}")

# --- GPU programs (compiled to PTX, built into the engine library) -----------------------
set(MOA_DEVICE_HEADERS
    src/device/LaunchParams.h src/device/random.h src/device/vec_math.h
    src/device/rays.cuh src/device/sampling.cuh src/device/bsdf.cuh src/device/textures.cuh
    src/device/surface.cuh src/device/environment.cuh src/device/lights.cuh src/device/camera.cuh)
add_library(moa_device OBJECT src/device/pathtracer.cu ${MOA_DEVICE_HEADERS})
# Rebuild the GPU programs whenever any of their building blocks change.
list(TRANSFORM MOA_DEVICE_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE _moa_device_deps)
set_source_files_properties(src/device/pathtracer.cu PROPERTIES OBJECT_DEPENDS "${_moa_device_deps}")
set_target_properties(moa_device PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES 75)  # Turing (RTX 20-series) and newer, which OptiX 9 requires
target_include_directories(moa_device PRIVATE "${OPTIX_INCLUDE_DIR}" src)
target_compile_options(moa_device PRIVATE --use_fast_math -lineinfo)

# Turn the PTX into a C++ array so the library carries its own GPU programs.
set(MOA_EMBEDDED_PTX "${CMAKE_CURRENT_BINARY_DIR}/generated/EmbeddedPtx.cpp")
add_custom_command(
    OUTPUT "${MOA_EMBEDDED_PTX}"
    COMMAND ${CMAKE_COMMAND} -DINPUT=$<TARGET_OBJECTS:moa_device> -DOUTPUT=${MOA_EMBEDDED_PTX}
            -DSYMBOL=moaEmbeddedPtx -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake"
    DEPENDS moa_device $<TARGET_OBJECTS:moa_device> "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedFile.cmake"
    COMMAND_EXPAND_LISTS
    COMMENT "Embedding pathtracer.ptx into the engine")

# --- Engine library (no window code, so it can run inside host apps and on render farms) --
add_library(moa_core STATIC
    src/Renderer.cpp
    src/DenoiserOptix.cpp
    src/Scene.cpp
    src/ImageIO.cpp
    src/MeshLoader.cpp
    src/Image.cpp
    src/Displacement.cpp
    third_party/tinyexr/miniz.c
    "${MOA_EMBEDDED_PTX}")
add_library(moa::core ALIAS moa_core)
target_include_directories(moa_core PUBLIC "${OPTIX_INCLUDE_DIR}" src PRIVATE third_party third_party/tinyexr)
# Static CUDA runtime: plugins then don't need to ship cudart DLLs next to them.
target_link_libraries(moa_core PUBLIC CUDA::cudart_static)
set_target_properties(moa_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WIN32)
    target_compile_definitions(moa_core PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(moa_core PUBLIC Cfgmgr32)  # used by OptiX to locate the driver
else()
    target_link_libraries(moa_core PUBLIC ${CMAKE_DL_LIBS})
endif()
if(MSVC)
    target_compile_options(moa_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /utf-8>)
else()
    target_compile_options(moa_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

if(NOT MOA_BUILD_VIEWER)
    return()
endif()

# --- Window libraries (live render view) -------------------------------------------------
# GLFW (window + input) and Dear ImGui (settings panel) are downloaded from GitHub.
find_package(OpenGL REQUIRED)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.9b
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(glfw imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# --- Live window and command-line renderer -----------------------------------------------
add_executable(moa
    src/main.cpp
    src/LiveView.cpp)
target_link_libraries(moa PRIVATE moa_core imgui)
if(WIN32)
    target_compile_definitions(moa PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
    target_compile_options(moa PRIVATE /W4 /utf-8)
else()
    target_compile_options(moa PRIVATE -Wall -Wextra)
endif()
