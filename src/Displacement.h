// MOA Renderer - displacement: subdivides a mesh and pushes its surface along the normals by a
// height map (e.g. Megascans' Displacement.exr), so silhouettes and shadows get the real shape.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Image.h"
#include "MeshLoader.h"

#include <cstddef>

namespace moa {

struct DisplacementSettings
{
    // height = map value (red channel) * scale + offset, in the mesh's own units, along the normal.
    // MaterialX "displacement" node: scale = its scale. Megascans-style maps centred on 0.5 use
    // offset = -0.5 * scale.
    float scale = 1.0f;
    float offset = 0.0f;
    // Subdivision: each level splits every triangle into 4. The level is picked so the result stays
    // under maxTriangles (and at most maxLevel).
    size_t maxTriangles = 4000000;
    int maxLevel = 6;
    bool flipV = false;  // texture V runs top-down (same meaning as MATFLAG_FLIP_V)
};

struct DisplacementResult
{
    int level = 0;          // subdivision levels used
    size_t triangles = 0;   // triangles after subdivision
};

// Subdivides `mesh` (positions, normals, texcoords; tangents are recomputed) and displaces it.
// Coincident vertices (UV seams, hard edges) move together, so the surface doesn't crack open.
DisplacementResult displaceMesh(Mesh& mesh, const HdrImage& heightMap, const DisplacementSettings& settings);

}  // namespace moa
