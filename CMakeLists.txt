# MOA Renderer
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)

project(MOARenderer VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# --- OptiX headers ----------------------------------------------------------------------
# OptiX is header-only at build time; the runtime ships inside the NVIDIA driver.
# By default the headers are downloaded from NVIDIA's GitHub. To use a local OptiX SDK
# instead, configure with -DOPTIX_ROOT="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0".
set(OPTIX_ROOT "" CACHE PATH "Optional path to a local OptiX SDK (folder containing include/optix.h)")
if(OPTIX_ROOT)
    set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")
else()
    include(FetchContent)
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

# --- GPU programs (compiled to PTX, loaded by OptiX at runtime) --------------------------
add_library(moa_device OBJECT src/device/pathtracer.cu)
set_target_properties(moa_device PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES 75)  # Turing (RTX 20-series) and newer, which OptiX 9 requires
target_include_directories(moa_device PRIVATE "${OPTIX_INCLUDE_DIR}" src)
target_compile_options(moa_device PRIVATE --use_fast_math -lineinfo)

# --- Command-line renderer ---------------------------------------------------------------
add_executable(moa
    src/main.cpp
    src/Renderer.cpp
    src/Scene.cpp
    src/ImageIO.cpp)
target_include_directories(moa PRIVATE "${OPTIX_INCLUDE_DIR}" src third_party)
target_link_libraries(moa PRIVATE CUDA::cudart)
if(WIN32)
    target_compile_definitions(moa PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(moa PRIVATE Cfgmgr32)  # used by OptiX to locate the driver
else()
    target_link_libraries(moa PRIVATE ${CMAKE_DL_LIBS})
endif()
if(MSVC)
    target_compile_options(moa PRIVATE /W4 /utf-8)
else()
    target_compile_options(moa PRIVATE -Wall -Wextra)
endif()

# Copy the compiled PTX next to the executable after every build.
add_dependencies(moa moa_device)
add_custom_command(TARGET moa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:moa_device> $<TARGET_FILE_DIR:moa>
    COMMAND_EXPAND_LISTS
    COMMENT "Copying pathtracer.ptx next to moa")
