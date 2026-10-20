// MOA Renderer - Cryptomatte: ID mattes for compositing (Psyop's open standard, supported by Nuke,
// Fusion, After Effects, Photoshop via Exr-IO, Houdini COPs...).
//
// Every object / material name is hashed to a 32-bit ID. While rendering, each pixel counts how many
// of its samples saw each ID (see kCryptoSlots in device/LaunchParams.h). writeCryptomatteExr()
// turns the counts into coverage, keeps the most-covered IDs per pixel (ranks), and writes them as
// float channels <layer>00.RGBA, <layer>01.RGBA, ... (id, coverage, id, coverage) with the
// "cryptomatte/<key>/..." metadata (name, hash, conversion, manifest) compositing tools look for.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moa {

// MurmurHash3 (x86, 32-bit), the hash Cryptomatte uses.
uint32_t murmurHash3_32(const std::string& text, uint32_t seed = 0);

// The Cryptomatte ID of a name: its hash as float bits (with the exponent nudged so it is never
// zero, denormal, inf or NaN), exactly as the standard specifies.
float cryptomatteId(const std::string& name);

// "%08x" of the ID's bits, as written in the manifest.
std::string cryptomatteHex(float id);

struct CryptomatteLayer
{
    std::string name;                 // e.g. "CryptoObject", "CryptoMaterial"
    const std::vector<float>* slots;  // width x height x kCryptoSlots x (id, sample count)
    std::vector<std::string> names;   // every name that can appear (for the manifest)
};

// Writes one EXR with all layers. `samples` = samples per pixel (turns counts into coverage).
// bottomUp: row 0 of the slot data is the bottom of the image (MOA's render buffers).
bool writeCryptomatteExr(const std::string& path, unsigned int width, unsigned int height, unsigned int samples,
                         const std::vector<CryptomatteLayer>& layers, bool bottomUp, std::string& error);

}  // namespace moa
