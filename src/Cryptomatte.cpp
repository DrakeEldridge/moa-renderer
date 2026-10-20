// MOA Renderer - Cryptomatte (see Cryptomatte.h).
// SPDX-License-Identifier: Apache-2.0
#include "Cryptomatte.h"

#include "device/LaunchParams.h"  // kCryptoSlots

#include "tinyexr.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <utility>

namespace moa {

uint32_t murmurHash3_32(const std::string& text, uint32_t seed)
{
    const uint8_t* data = reinterpret_cast<const uint8_t*>(text.data());
    const size_t len = text.size();
    const size_t nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51u, c2 = 0x1b873593u;
    auto rotl = [](uint32_t x, int r) { return (x << r) | (x >> (32 - r)); };
    for (size_t i = 0; i < nblocks; ++i)
    {
        uint32_t k1;
        std::memcpy(&k1, data + i * 4, 4);  // little-endian, like the reference implementation on x86
        k1 *= c1;
        k1 = rotl(k1, 15);
        k1 *= c2;
        h1 ^= k1;
        h1 = rotl(h1, 13);
        h1 = h1 * 5 + 0xe6546b64u;
    }
    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3)
    {
    case 3:
        k1 ^= uint32_t(tail[2]) << 16;
        [[fallthrough]];
    case 2:
        k1 ^= uint32_t(tail[1]) << 8;
        [[fallthrough]];
    case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = rotl(k1, 15);
        k1 *= c2;
        h1 ^= k1;
    }
    h1 ^= uint32_t(len);
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6bu;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35u;
    h1 ^= h1 >> 16;
    return h1;
}

float cryptomatteId(const std::string& name)
{
    uint32_t h = murmurHash3_32(name);
    const uint32_t exponent = (h >> 23) & 255u;
    if (exponent == 0u || exponent == 255u)
        h ^= 1u << 23;
    float f;
    std::memcpy(&f, &h, 4);
    return f;
}

std::string cryptomatteHex(float id)
{
    uint32_t bits;
    std::memcpy(&bits, &id, 4);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", bits);
    return buf;
}

namespace {

std::string jsonString(const std::string& s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"' || c == '\\')
        {
            out += '\\';
            out += c;
        }
        else if (static_cast<unsigned char>(c) < 0x20)
        {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        }
        else
            out += c;
    }
    return out + "\"";
}

void addStringAttribute(std::vector<EXRAttribute>& attrs, std::vector<std::string>& storage, const std::string& name,
                        const std::string& value)
{
    EXRAttribute a{};
    std::snprintf(a.name, sizeof(a.name), "%s", name.c_str());
    std::snprintf(a.type, sizeof(a.type), "string");
    storage.push_back(value);
    a.value = reinterpret_cast<unsigned char*>(const_cast<char*>(storage.back().data()));
    a.size = static_cast<int>(storage.back().size());
    attrs.push_back(a);
}

}  // namespace

bool writeCryptomatteExr(const std::string& path, unsigned int width, unsigned int height, unsigned int samples,
                         const std::vector<CryptomatteLayer>& layers, bool bottomUp, std::string& error)
{
    const size_t pixels = size_t(width) * height;
    if (pixels == 0 || layers.empty())
    {
        error = "nothing to write";
        return false;
    }
    const float inv = 1.0f / float(std::max(1u, samples));
    const unsigned int ranks = kCryptoSlots;
    const unsigned int groups = (ranks + 1) / 2;  // each RGBA channel group holds two ranks

    // Channel planes, keyed by name (EXR wants channels sorted by name).
    std::vector<std::pair<std::string, std::vector<float>>> channels;
    for (const CryptomatteLayer& layer : layers)
    {
        if (!layer.slots || layer.slots->size() < pixels * 2 * kCryptoSlots)
        {
            error = "missing Cryptomatte data for " + layer.name;
            return false;
        }
        std::vector<std::vector<float>> planes(groups * 4, std::vector<float>(pixels, 0.0f));
        std::vector<std::pair<float, float>> slot(kCryptoSlots);
        for (unsigned int y = 0; y < height; ++y)
        {
            const unsigned int srcRow = bottomUp ? height - 1 - y : y;
            for (unsigned int x = 0; x < width; ++x)
            {
                const float* s = layer.slots->data() + (size_t(srcRow) * width + x) * 2 * kCryptoSlots;
                for (unsigned int k = 0; k < kCryptoSlots; ++k)
                    slot[k] = {s[2 * k + 1], s[2 * k]};  // (count, id)
                std::sort(slot.begin(), slot.end(),
                          [](const std::pair<float, float>& a, const std::pair<float, float>& b) { return a.first > b.first; });
                const size_t o = size_t(y) * width + x;
                for (unsigned int r = 0; r < ranks; ++r)
                {
                    if (slot[r].first <= 0.0f)
                        break;
                    const unsigned int g = r / 2;
                    const unsigned int c = (r % 2) * 2;  // R,G = rank 0 ; B,A = rank 1
                    planes[g * 4 + c][o] = slot[r].second;
                    planes[g * 4 + c + 1][o] = slot[r].first * inv;
                }
            }
        }
        static const char* const rgba[4] = {"R", "G", "B", "A"};
        for (unsigned int g = 0; g < groups; ++g)
            for (unsigned int c = 0; c < 4; ++c)
            {
                char name[256];
                std::snprintf(name, sizeof(name), "%s%02u.%s", layer.name.c_str(), g, rgba[c]);
                channels.emplace_back(name, std::move(planes[g * 4 + c]));
            }
    }
    std::sort(channels.begin(), channels.end(),
              [](const std::pair<std::string, std::vector<float>>& a, const std::pair<std::string, std::vector<float>>& b) {
                  return a.first < b.first;
              });

    // Metadata: cryptomatte/<first 7 hex digits of the layer name's hash>/{name,hash,conversion,manifest}.
    std::vector<EXRAttribute> attrs;
    std::vector<std::string> storage;
    storage.reserve(layers.size() * 4 + 1);  // keep string addresses stable
    for (const CryptomatteLayer& layer : layers)
    {
        char key[16];
        std::snprintf(key, sizeof(key), "%08x", murmurHash3_32(layer.name));
        const std::string prefix = "cryptomatte/" + std::string(key).substr(0, 7) + "/";
        std::string manifest = "{";
        std::set<std::string> seen;
        for (const std::string& n : layer.names)
        {
            if (n.empty() || !seen.insert(n).second)
                continue;
            if (manifest.size() > 1)
                manifest += ",";
            manifest += jsonString(n) + ":" + jsonString(cryptomatteHex(cryptomatteId(n)));
        }
        manifest += "}";
        addStringAttribute(attrs, storage, prefix + "name", layer.name);
        addStringAttribute(attrs, storage, prefix + "hash", "MurmurHash3_32");
        addStringAttribute(attrs, storage, prefix + "conversion", "uint32_to_float32");
        addStringAttribute(attrs, storage, prefix + "manifest", manifest);
    }

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);
    const int numChannels = static_cast<int>(channels.size());
    std::vector<EXRChannelInfo> infos(numChannels);
    std::vector<int> pixelTypes(numChannels, TINYEXR_PIXELTYPE_FLOAT);
    std::vector<unsigned char*> imagePtrs(numChannels);
    for (int i = 0; i < numChannels; ++i)
    {
        std::memset(&infos[i], 0, sizeof(EXRChannelInfo));
        std::snprintf(infos[i].name, sizeof(infos[i].name), "%s", channels[i].first.c_str());
        imagePtrs[i] = reinterpret_cast<unsigned char*>(channels[i].second.data());
    }
    header.num_channels = numChannels;
    header.channels = infos.data();
    header.pixel_types = pixelTypes.data();
    header.requested_pixel_types = pixelTypes.data();  // IDs must stay full 32-bit floats
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;  // lossless
    header.num_custom_attributes = static_cast<int>(attrs.size());
    header.custom_attributes = attrs.data();
    image.num_channels = numChannels;
    image.images = imagePtrs.data();
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);

    const char* err = nullptr;
    const int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
    if (ret != TINYEXR_SUCCESS)
    {
        error = err ? err : "couldn't write the file";
        if (err)
            FreeEXRErrorMessage(err);
        return false;
    }
    return true;
}

}  // namespace moa
