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


// Lookup table for linear -> sRGB bytes; avoids a pow() per channel per pixel.
const uint8_t* srgbLut()
{
    static uint8_t lut[4096];
    static bool ready = false;
    if (!ready)
    {
        for (int i = 0; i < 4096; ++i)
            lut[i] = toByte(linearToSrgb(static_cast<float>(i) / 4095.0f));
        ready = true;
    }
    return lut;
}

inline uint8_t srgbByte(const uint8_t* lut, float x)
{
    const float c = std::clamp(x, 0.0f, 1.0f);
    return lut[static_cast<int>(c * 4095.0f + 0.5f)];
}

}  // namespace

void encodeRgba8(const std::vector<float4>& pixels, std::vector<uint8_t>& out, PngEncoding encoding, float exposure)
{
    const uint8_t* lut = srgbLut();
    out.resize(pixels.size() * 4);
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        const float4 p = pixels[i];
        uint8_t* d = &out[i * 4];
        switch (encoding)
        {
        case PngEncoding::Beauty:
            d[0] = srgbByte(lut, acesFilmic(p.x * exposure));
            d[1] = srgbByte(lut, acesFilmic(p.y * exposure));
            d[2] = srgbByte(lut, acesFilmic(p.z * exposure));
            break;
        case PngEncoding::Albedo:
            d[0] = srgbByte(lut, p.x);
            d[1] = srgbByte(lut, p.y);
            d[2] = srgbByte(lut, p.z);
            break;
        case PngEncoding::Normal:
            d[0] = toByte(p.x * 0.5f + 0.5f);
            d[1] = toByte(p.y * 0.5f + 0.5f);
            d[2] = toByte(p.z * 0.5f + 0.5f);
            break;
        }
        d[3] = 255;
    }
}

bool writePng(const std::string& path, unsigned int width, unsigned int height,
              const std::vector<float4>& pixels, PngEncoding encoding, float exposure)
{
    if (pixels.size() < static_cast<size_t>(width) * height)
        return false;
    std::vector<uint8_t> rgba;
    encodeRgba8(pixels, rgba, encoding, exposure);

    // PNG rows go top to bottom; our buffers go bottom to top.
    const size_t rowBytes = static_cast<size_t>(width) * 4;
    std::vector<uint8_t> flipped(rgba.size());
    for (unsigned int y = 0; y < height; ++y)
        std::copy_n(&rgba[static_cast<size_t>(height - 1 - y) * rowBytes], rowBytes, &flipped[y * rowBytes]);
    return stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height), 4, flipped.data(),
                          static_cast<int>(rowBytes)) != 0;
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
