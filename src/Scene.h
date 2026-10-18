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

// A texture used by the scene. sRGB = colour images (decoded to linear when sampled);
// normal maps and other data images are linear.
struct SceneTexture
{
    ImagePtr image;
    bool srgb = true;
};

struct Scene
{
    std::vector<float3> positions;               // vertex positions
    std::vector<float3> normals;                 // one per position (shading normals)
    std::vector<float2> texcoords;               // one per position
    std::vector<float4> tangents;                // one per position (for normal maps)
    std::vector<uint3> indices;                  // three positions per triangle
    std::vector<unsigned int> triangleMaterial;  // one per triangle
    std::vector<Material> materials;
    std::vector<SceneTexture> textures;
    QuadLight light{};
    bool hasQuadLight = true;

    // Environment light (HDRI). No image = no environment.
    HdrImagePtr environment;
    float environmentIntensity = 1.0f;
    float environmentRotation = 0.0f;  // degrees
    bool environmentVisible = true;
    CameraDesc camera{};

    size_t triangleCount() const { return triangleMaterial.size(); }

    unsigned int addMaterial(const Material& m)
    {
        materials.push_back(m);
        return static_cast<unsigned int>(materials.size() - 1);
    }

    int addTexture(const ImagePtr& image, bool srgb)
    {
        if (!image)
            return -1;
        textures.push_back({image, srgb});
        return static_cast<int>(textures.size() - 1);
    }

    // Adds a planar quad a-b-c-d as two triangles. The winding is flipped if needed so the
    // geometric normal points along `facing` (dot(normal, facing) > 0).
    void addQuad(float3 a, float3 b, float3 c, float3 d, unsigned int material, float3 facing);

    // Adds a closed prism: `base` is a 4-corner footprint (any winding), extruded by `height` along +Y.
    // All faces are wound so their normals point outwards (needed for glass).
    void addBlock(const float3 base[4], float height, unsigned int material);

    // Adds a loaded mesh, transformed so it stands on the floor at `floorCenter` and its height
    // (or largest size, if wider than tall) equals `fitSize`. zUp: the file's up axis is +Z.
    // rotation: 3x3 row-major matrix applied after the Z-up conversion, then `spinDegrees` around Y.
    // The model is fitted (stood on the floor, sized) after rotating, then moved by `offset`.
    void addFittedMesh(const Mesh& mesh, unsigned int material, float3 floorCenter, float fitSize, bool zUp,
                       const float rotation[9], float spinDegrees, float3 offset);
};

// Tables for importance-sampling an HDRI (brightness x sin(theta) per pixel, then row/column CDFs).
struct EnvironmentTables
{
    std::vector<float> func;         // width x height
    std::vector<float> marginal;     // height + 1
    std::vector<float> conditional;  // height x (width + 1)
    float funcIntegral = 0.0f;
};
EnvironmentTables buildEnvironmentTables(const HdrImage& image);

// Ready-made materials for the test scene and loaded models.
enum MaterialPreset : int
{
    PRESET_DIFFUSE = 0,  // matte (clay)
    PRESET_MIRROR,
    PRESET_GLASS,
    PRESET_PLASTIC,      // glossy dielectric over the base colour
    PRESET_CHROME,       // polished metal
    PRESET_BRUSHED,      // rough metal
    PRESET_GOLD,
    PRESET_FROSTED,      // rough glass
    PRESET_COUNT
};

// Display names, in MaterialPreset order.
extern const char* const kMaterialPresetNames[PRESET_COUNT];

// Builds the material for a preset (standard material for everything except matte/mirror/glass).
Material materialFromPreset(MaterialPreset preset);

// Options for the built-in test scene.
struct CornellOptions
{
    MaterialPreset tallBlock = PRESET_DIFFUSE;
    MaterialPreset shortBlock = PRESET_DIFFUSE;

    // Optional model placed in the middle of the box (replaces the blocks unless showBlocks).
    const Mesh* model = nullptr;
    MaterialPreset modelMaterial = PRESET_DIFFUSE;
    bool showBlocks = false;
    bool modelZUp = false;     // many STL files are Z-up
    // Orientation built from button clicks (each one turns the model around a world axis).
    float modelRotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float modelYaw = 0.0f;     // extra fine spin around the vertical axis, degrees
    float modelSize = 3.6f;    // fitted height in scene units (the box is 5.5 tall)
    float3 modelOffset = {0.0f, 0.0f, 0.0f};  // move from the fitted spot (scene units)

    // Turns the model around world axis 0 = X, 1 = Y, 2 = Z by `degrees`, on top of its current orientation.
    void rotateModel(int axis, float degrees);

    // HDRI lighting: replaces the Cornell box with an open scene lit by the environment image.
    HdrImagePtr environment;
    bool useEnvironment = false;
    float environmentIntensity = 1.0f;
    float environmentRotation = 0.0f;  // degrees
    bool environmentVisible = true;    // show the HDRI behind the model
    bool showFloor = true;             // grey ground plane in HDRI mode

    // Model textures (optional) and how to read them.
    ModelTextures modelTextures;
    bool useTextures = true;
    bool flipV = false;            // for textures authored with V running downwards
    bool flipNormalGreen = false;  // for DirectX-style normal maps
    float normalStrength = 1.0f;
};

Scene makeCornellBox(const CornellOptions& options);

// Parses a preset name: diffuse, mirror, glass, plastic, chrome, brushed, gold, frosted.
// Returns false if the name is unknown.
bool parseMaterialPreset(const std::string& name, MaterialPreset& out);

}  // namespace moa
