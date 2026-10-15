// MOA Renderer - in-memory scene description.
// For this first milestone the scene is built in code (a Cornell box). USD and Alembic
// loading will fill the same structure in a later milestone.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "MeshLoader.h"
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
    std::vector<float3> positions;               // vertex positions
    std::vector<float3> normals;                 // one per position (shading normals)
    std::vector<uint3> indices;                  // three positions per triangle
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

    // Adds a loaded mesh, transformed so it stands on the floor at `floorCenter` and its height
    // (or largest size, if wider than tall) equals `fitSize`. zUp: the file's up axis is +Z.
    // rotation: degrees around X, then Z, then Y (applied after the Z-up conversion).
    void addFittedMesh(const Mesh& mesh, unsigned int material, float3 floorCenter, float fitSize, bool zUp,
                       float3 rotationDegrees);
};

// Options for the built-in test scene.
struct CornellOptions
{
    MaterialType tallBlock = MAT_DIFFUSE;
    MaterialType shortBlock = MAT_DIFFUSE;

    // Optional model placed in the middle of the box (replaces the blocks unless showBlocks).
    const Mesh* model = nullptr;
    MaterialType modelMaterial = MAT_DIFFUSE;
    bool showBlocks = false;
    bool modelZUp = false;     // many STL files are Z-up
    float modelYaw = 0.0f;     // degrees around the vertical axis (Y)
    float modelTiltX = 0.0f;   // degrees around X (fixes models lying down)
    float modelTiltZ = 0.0f;   // degrees around Z
    float modelSize = 3.6f;    // fitted height in scene units (the box is 5.5 tall)
};

Scene makeCornellBox(const CornellOptions& options);

// Parses "diffuse", "mirror" or "glass". Returns false if the name is unknown.
bool parseMaterialType(const std::string& name, MaterialType& out);

}  // namespace moa
