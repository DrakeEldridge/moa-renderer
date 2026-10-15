// MOA Renderer - small, fast random number helpers for the GPU.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "vec_math.h"

namespace moa {

// Tiny Encryption Algorithm hash, used to give every pixel and launch its own seed.
template <unsigned int Rounds>
MOA_HD unsigned int tea(unsigned int v0, unsigned int v1)
{
    unsigned int s0 = 0;
    for (unsigned int n = 0; n < Rounds; ++n)
    {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
    }
    return v0;
}

// PCG hash step: advances the state and returns a float in [0, 1).
MOA_HD float rnd(unsigned int& state)
{
    state = state * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return static_cast<float>(word >> 8) * (1.0f / 16777216.0f);
}

}  // namespace moa
