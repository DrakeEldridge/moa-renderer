// MOA Renderer - in-memory scene description.
// For this first milestone the scene is built in code (a Cornell box). USD and Alembic
// loading will fill the same structure in a later milestone.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "device/LaunchParams.h"

#include <string>
#include <vector>

namespace moa {

struct CameraDesc
{
    float3 eye;
    float3 lookAt;
    float3 up;
    float verticalFovDegrees;
};

struct Scene
{
    std::vector<float3> vertices;                // 3 per triangle
    std::vector<unsigned int> triangleMaterial;  // one per triangle
    std::vector<Material> materials;
    QuadLight light{};
    CameraDesc camera{};

    size_t triangleCount() const { return triangleMaterial.size(); }

    unsigned int addMaterial(const Material& m)
    {
        materials.push_back(m);
        return static_cast<unsigned int>(materials.size() - 1);
    }

    // Adds a planar quad a-b-c-d as two triangles. The winding is flipped if needed so the
    // geometric normal points along `facing` (dot(normal, facing) > 0).
    void addQuad(float3 a, float3 b, float3 c, float3 d, unsigned int material, float3 facing);

    // Adds a closed prism: `base` is a 4-corner footprint (any winding), extruded by `height` along +Y.
    // All faces are wound so their normals point outwards (needed for glass).
    void addBlock(const float3 base[4], float height, unsigned int material);
};

// Options for the built-in test scene.
struct CornellOptions
{
    MaterialType tallBlock = MAT_DIFFUSE;
    MaterialType shortBlock = MAT_DIFFUSE;
};

Scene makeCornellBox(const CornellOptions& options);

// Parses "diffuse", "mirror" or "glass". Returns false if the name is unknown.
bool parseMaterialType(const std::string& name, MaterialType& out);

}  // namespace moa
