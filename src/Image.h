// MOA Renderer - loading images: textures (JPG, PNG, TGA, BMP...) as 8-bit RGBA, HDRIs and data
// maps (HDR, EXR) as float RGBA.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace moa {

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;  // row 0 = top of the image, 4 bytes per pixel
    std::string name;           // file name, for display
};

using ImagePtr = std::shared_ptr<const Image>;

// High dynamic range image (e.g. an HDRI environment), linear float RGBA.
struct HdrImage
{
    int width = 0;
    int height = 0;
    std::vector<float> rgba;  // row 0 = top, 4 floats per pixel
    std::string name;
};

using HdrImagePtr = std::shared_ptr<const HdrImage>;

// Loads a Radiance .hdr or OpenEXR file. Returns nullptr and sets `error` on failure.
HdrImagePtr loadHdrImage(const std::string& path, std::string& error);

// Loads any supported image as linear floats without colour conversion: EXR/HDR as stored,
// 8/16-bit images as value / max. For data maps such as displacement.
HdrImagePtr loadFloatImage(const std::string& path, std::string& error);

// Loads an image file. Returns nullptr and sets `error` on failure.
ImagePtr loadImage(const std::string& path, std::string& error);

// True if the image looks like a tangent-space normal map (mostly light blue/purple:
// blue channel high, red and green around the middle).
bool looksLikeNormalMap(const Image& image);

}  // namespace moa
