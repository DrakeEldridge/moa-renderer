// MOA Renderer - loading texture images.
// SPDX-License-Identifier: Apache-2.0
#include "Image.h"

#include <cmath>
#include <filesystem>
#include <fstream>

#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR  // HDR/EXR images come later, through a different loader
#include "stb/stb_image.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace moa {

ImagePtr loadImage(const std::string& path, std::string& error)
{
    // Read the bytes ourselves so Unicode paths work on Windows, then decode from memory.
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!f)
    {
        error = "Couldn't open " + path;
        return nullptr;
    }
    const std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    if (size <= 0 || !f.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        error = "Couldn't read " + path;
        return nullptr;
    }

    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &channels, 4);
    if (!pixels)
    {
        error = std::string("Couldn't decode image: ") + stbi_failure_reason();
        return nullptr;
    }
    auto image = std::make_shared<Image>();
    image->width = w;
    image->height = h;
    image->rgba.assign(pixels, pixels + static_cast<size_t>(w) * h * 4);
    image->name = std::filesystem::u8path(path).filename().string();
    stbi_image_free(pixels);
    return image;
}

bool looksLikeNormalMap(const Image& image)
{
    if (image.rgba.empty())
        return false;
    // Sample a grid of pixels: normal maps average about (128, 128, 230+) and rarely have dark blue.
    double r = 0.0, g = 0.0, b = 0.0;
    int count = 0, blueish = 0;
    const int steps = 64;
    for (int j = 0; j < steps; ++j)
        for (int i = 0; i < steps; ++i)
        {
            const int x = (i * image.width) / steps, y = (j * image.height) / steps;
            const uint8_t* p = &image.rgba[(static_cast<size_t>(y) * image.width + x) * 4];
            r += p[0];
            g += p[1];
            b += p[2];
            ++count;
            if (p[2] > 180 && p[2] >= p[0] && p[2] >= p[1])
                ++blueish;
        }
    r /= count;
    g /= count;
    b /= count;
    return b > 190.0 && std::abs(r - 128.0) < 40.0 && std::abs(g - 128.0) < 40.0 && blueish > count * 8 / 10;
}

}  // namespace moa
