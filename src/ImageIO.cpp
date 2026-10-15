// MOA Renderer - writing rendered images to disk.
// SPDX-License-Identifier: Apache-2.0
#include "ImageIO.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

// Third-party code: silence its warnings so ours stay visible.
#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace moa {

namespace {

float linearToSrgb(float x)
{
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

// Krzysztof Narkowicz's fitted approximation of the ACES filmic curve.
float acesFilmic(float x)
{
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

uint8_t toByte(float x) { return static_cast<uint8_t>(std::lround(std::clamp(x, 0.0f, 1.0f) * 255.0f)); }

}  // namespace

bool writePng(const std::string& path, unsigned int width, unsigned int height,
              const std::vector<float4>& pixels, PngEncoding encoding, float exposure)
{
    if (pixels.size() < static_cast<size_t>(width) * height)
        return false;

    std::vector<uint8_t> out(static_cast<size_t>(width) * height * 3);
    for (unsigned int y = 0; y < height; ++y)
    {
        const unsigned int srcRow = height - 1 - y;  // PNG rows go top to bottom
        for (unsigned int x = 0; x < width; ++x)
        {
            const float4 p = pixels[static_cast<size_t>(srcRow) * width + x];
            float c[3] = {p.x, p.y, p.z};
            for (float& v : c)
            {
                switch (encoding)
                {
                case PngEncoding::Beauty: v = linearToSrgb(acesFilmic(v * exposure)); break;
                case PngEncoding::Albedo: v = linearToSrgb(v); break;
                case PngEncoding::Normal: v = v * 0.5f + 0.5f; break;
                }
            }
            uint8_t* dst = &out[(static_cast<size_t>(y) * width + x) * 3];
            dst[0] = toByte(c[0]);
            dst[1] = toByte(c[1]);
            dst[2] = toByte(c[2]);
        }
    }
    return stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 3, out.data(),
                          static_cast<int>(width * 3)) != 0;
}

bool writePfm(const std::string& path, unsigned int width, unsigned int height, const std::vector<float4>& pixels)
{
    if (pixels.size() < static_cast<size_t>(width) * height)
        return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    // Negative scale = little-endian. PFM rows are stored bottom to top, matching our buffers.
    std::fprintf(f, "PF\n%u %u\n-1.0\n", width, height);
    std::vector<float> row(static_cast<size_t>(width) * 3);
    bool ok = true;
    for (unsigned int y = 0; y < height && ok; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            const float4 p = pixels[static_cast<size_t>(y) * width + x];
            row[x * 3 + 0] = p.x;
            row[x * 3 + 1] = p.y;
            row[x * 3 + 2] = p.z;
        }
        ok = std::fwrite(row.data(), sizeof(float), row.size(), f) == row.size();
    }
    std::fclose(f);
    return ok;
}

}  // namespace moa
