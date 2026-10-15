// MOA Renderer - writing rendered images to disk.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector_types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace moa {

enum class PngEncoding
{
    Beauty,  // HDR colour: exposure, filmic tone map, then sRGB
    Albedo,  // linear colour in [0,1]: sRGB only
    Normal,  // unit vectors: remapped from [-1,1] to [0,1]
};

// Converts float pixels to 8-bit RGBA (alpha 255) using the given encoding. Rows stay in render
// order (row 0 = bottom). Fast enough to run every frame in the live window.
void encodeRgba8(const std::vector<float4>& pixels, std::vector<uint8_t>& out, PngEncoding encoding,
                 float exposure = 1.0f);

// Writes an 8-bit PNG. `pixels` is bottom-to-top (row 0 = bottom), as rendered.
// Returns false if the file couldn't be written.
bool writePng(const std::string& path, unsigned int width, unsigned int height,
              const std::vector<float4>& pixels, PngEncoding encoding, float exposure = 1.0f);

// Writes a 32-bit float colour image (Portable Float Map), viewable in Nuke, Photoshop and
// most HDR viewers. Keeps the full HDR range for comp tests until EXR output lands.
bool writePfm(const std::string& path, unsigned int width, unsigned int height, const std::vector<float4>& pixels);

}  // namespace moa
