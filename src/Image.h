// MOA Renderer - loading texture images (JPG, PNG, TGA, BMP...) into 8-bit RGBA.
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

// Loads an image file. Returns nullptr and sets `error` on failure.
ImagePtr loadImage(const std::string& path, std::string& error);

// True if the image looks like a tangent-space normal map (mostly light blue/purple:
// blue channel high, red and green around the middle).
bool looksLikeNormalMap(const Image& image);

}  // namespace moa
