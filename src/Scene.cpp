// MOA Renderer - built-in test scene.
// The Cornell box is the classic rendering test scene (Cornell University, 1984); the
// dimensions follow the published measurements, scaled from millimetres to decimetres.
// SPDX-License-Identifier: Apache-2.0
#include "Scene.h"

#include "device/vec_math.h"

namespace moa {

namespace {

constexpr float kScale = 0.01f;

float3 P(float x, float y, float z) { return make_float3(x * kScale, y * kScale, z * kScale); }

Material diffuse(float r, float g, float b)
{
    Material m{};
    m.albedo = make_float3(r, g, b);
    m.emission = make_float3(0.0f, 0.0f, 0.0f);
    m.type = MAT_DIFFUSE;
    m.ior = 1.0f;
    return m;
}

Material ofType(MaterialType type)
{
    switch (type)
    {
    case MAT_MIRROR:
    {
        Material m = diffuse(0.95f, 0.95f, 0.95f);
        m.type = MAT_MIRROR;
        return m;
    }
    case MAT_GLASS:
    {
        Material m = diffuse(1.0f, 1.0f, 1.0f);
        m.type = MAT_GLASS;
        m.ior = 1.5f;
        return m;
    }
    case MAT_DIFFUSE:
    default:
        return diffuse(0.73f, 0.73f, 0.73f);
    }
}

}  // namespace

void Scene::addQuad(float3 a, float3 b, float3 c, float3 d, unsigned int material, float3 facing)
{
    const float3 n = cross(b - a, c - a);
    if (dot(n, facing) >= 0.0f)
    {
        vertices.insert(vertices.end(), {a, b, c, a, c, d});
    }
    else
    {
        vertices.insert(vertices.end(), {a, c, b, a, d, c});
    }
    triangleMaterial.push_back(material);
    triangleMaterial.push_back(material);
}

void Scene::addBlock(const float3 base[4], float height, unsigned int material)
{
    const float3 up = make_float3(0.0f, height, 0.0f);
    const float3 top[4] = {base[0] + up, base[1] + up, base[2] + up, base[3] + up};
    const float3 center = (base[0] + base[1] + base[2] + base[3]) * 0.25f + up * 0.5f;

    addQuad(top[0], top[1], top[2], top[3], material, make_float3(0.0f, 1.0f, 0.0f));
    addQuad(base[0], base[1], base[2], base[3], material, make_float3(0.0f, -1.0f, 0.0f));
    for (int i = 0; i < 4; ++i)
    {
        const int j = (i + 1) % 4;
        const float3 faceCenter = (base[i] + base[j] + top[j] + top[i]) * 0.25f;
        addQuad(base[i], base[j], top[j], top[i], material, faceCenter - center);
    }
}

Scene makeCornellBox(const CornellOptions& options)
{
    Scene s;

    const unsigned int white = s.addMaterial(diffuse(0.73f, 0.73f, 0.73f));
    const unsigned int red = s.addMaterial(diffuse(0.65f, 0.05f, 0.05f));
    const unsigned int green = s.addMaterial(diffuse(0.12f, 0.45f, 0.15f));
    const unsigned int tall = s.addMaterial(ofType(options.tallBlock));
    const unsigned int shrt = s.addMaterial(ofType(options.shortBlock));

    Material lightMat = diffuse(0.0f, 0.0f, 0.0f);
    lightMat.emission = make_float3(17.0f, 12.0f, 4.0f);
    const unsigned int lightId = s.addMaterial(lightMat);

    // Room surfaces are wound to face inwards (shading flips normals anyway; this just keeps data tidy).
    const float3 roomCenter = P(278.0f, 274.4f, 279.6f);

    // Floor, ceiling, back wall (white), left wall (red), right wall (green), as seen from the camera.
    s.addQuad(P(552.8f, 0.0f, 0.0f), P(0.0f, 0.0f, 0.0f), P(0.0f, 0.0f, 559.2f), P(549.6f, 0.0f, 559.2f),
              white, make_float3(0.0f, 1.0f, 0.0f));
    s.addQuad(P(556.0f, 548.8f, 0.0f), P(556.0f, 548.8f, 559.2f), P(0.0f, 548.8f, 559.2f), P(0.0f, 548.8f, 0.0f),
              white, make_float3(0.0f, -1.0f, 0.0f));
    s.addQuad(P(549.6f, 0.0f, 559.2f), P(0.0f, 0.0f, 559.2f), P(0.0f, 548.8f, 559.2f), P(556.0f, 548.8f, 559.2f),
              white, make_float3(0.0f, 0.0f, -1.0f));
    s.addQuad(P(552.8f, 0.0f, 0.0f), P(549.6f, 0.0f, 559.2f), P(556.0f, 548.8f, 559.2f), P(556.0f, 548.8f, 0.0f),
              red, roomCenter - P(552.8f, 274.4f, 279.6f));
    s.addQuad(P(0.0f, 0.0f, 559.2f), P(0.0f, 0.0f, 0.0f), P(0.0f, 548.8f, 0.0f), P(0.0f, 548.8f, 559.2f),
              green, roomCenter - P(0.0f, 274.4f, 279.6f));

    // Ceiling light, just below the ceiling, emitting downwards.
    const float3 l0 = P(343.0f, 548.7f, 227.0f);
    const float3 l1 = P(213.0f, 548.7f, 227.0f);
    const float3 l2 = P(213.0f, 548.7f, 332.0f);
    const float3 l3 = P(343.0f, 548.7f, 332.0f);
    s.addQuad(l0, l1, l2, l3, lightId, make_float3(0.0f, -1.0f, 0.0f));
    s.light.corner = l0;
    s.light.edge1 = l1 - l0;
    s.light.edge2 = l3 - l0;
    s.light.normal = make_float3(0.0f, -1.0f, 0.0f);
    s.light.emission = lightMat.emission;
    s.light.area = length(cross(s.light.edge1, s.light.edge2));

    // The two blocks.
    const float3 shortBase[4] = {P(130.0f, 0.0f, 65.0f), P(82.0f, 0.0f, 225.0f), P(240.0f, 0.0f, 272.0f),
                                 P(290.0f, 0.0f, 114.0f)};
    s.addBlock(shortBase, 165.0f * kScale, shrt);
    const float3 tallBase[4] = {P(423.0f, 0.0f, 247.0f), P(265.0f, 0.0f, 296.0f), P(314.0f, 0.0f, 456.0f),
                                P(472.0f, 0.0f, 406.0f)};
    s.addBlock(tallBase, 330.0f * kScale, tall);

    s.camera.eye = P(278.0f, 273.0f, -800.0f);
    s.camera.lookAt = P(278.0f, 273.0f, 0.0f);
    s.camera.up = make_float3(0.0f, 1.0f, 0.0f);
    s.camera.verticalFovDegrees = 39.3f;
    return s;
}

bool parseMaterialType(const std::string& name, MaterialType& out)
{
    if (name == "diffuse") { out = MAT_DIFFUSE; return true; }
    if (name == "mirror") { out = MAT_MIRROR; return true; }
    if (name == "glass") { out = MAT_GLASS; return true; }
    return false;
}

}  // namespace moa
