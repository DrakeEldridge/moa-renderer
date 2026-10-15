// MOA Renderer - built-in test scene.
// The Cornell box is the classic rendering test scene (Cornell University, 1984); the
// dimensions follow the published measurements, scaled from millimetres to decimetres.
// SPDX-License-Identifier: Apache-2.0
#include "Scene.h"

#include "device/vec_math.h"

#include <algorithm>
#include <cmath>

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
    m.colorTexture = -1;
    m.normalTexture = -1;
    m.flags = 0;
    m.normalStrength = 1.0f;
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
    float3 n = normalize(cross(b - a, c - a));
    const bool flip = dot(n, facing) < 0.0f;
    if (flip)
        n = -n;
    // Flat surface: four vertices of its own, all with the face normal.
    const unsigned int base = static_cast<unsigned int>(positions.size());
    positions.insert(positions.end(), {a, b, c, d});
    normals.insert(normals.end(), {n, n, n, n});
    const float3 t = normalize(b - a);
    const float4 tangent = make_float4(t.x, t.y, t.z, 1.0f);
    texcoords.insert(texcoords.end(), {make_float2(0.0f, 0.0f), make_float2(1.0f, 0.0f), make_float2(1.0f, 1.0f), make_float2(0.0f, 1.0f)});
    tangents.insert(tangents.end(), {tangent, tangent, tangent, tangent});
    if (!flip)
    {
        indices.push_back(make_uint3(base, base + 1, base + 2));
        indices.push_back(make_uint3(base, base + 2, base + 3));
    }
    else
    {
        indices.push_back(make_uint3(base, base + 2, base + 1));
        indices.push_back(make_uint3(base, base + 3, base + 2));
    }
    triangleMaterial.push_back(material);
    triangleMaterial.push_back(material);
}

void Scene::addFittedMesh(const Mesh& mesh, unsigned int material, float3 floorCenter, float fitSize, bool zUp,
                          const float rotation[9], float spinDegrees, float3 offset)
{
    if (mesh.positions.empty() || mesh.indices.empty())
        return;
    const float spin = spinDegrees * kPi / 180.0f;
    const float cy = std::cos(spin), sy = std::sin(spin);
    const float* r = rotation;
    // Re-orient into this scene's Y-up space, apply the clicked rotations, then the fine spin.
    auto orient = [&](float3 v) {
        if (zUp)
            v = make_float3(v.x, v.z, -v.y);
        v = make_float3(r[0] * v.x + r[1] * v.y + r[2] * v.z, r[3] * v.x + r[4] * v.y + r[5] * v.z,
                        r[6] * v.x + r[7] * v.y + r[8] * v.z);
        return make_float3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z);
    };

    float3 lo = orient(mesh.positions[0]), hi = lo;
    for (const float3& p : mesh.positions)
    {
        const float3 v = orient(p);
        lo = make_float3(std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z));
        hi = make_float3(std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z));
    }
    const float3 size = hi - lo;
    // Fit the height to fitSize, but keep wide models inside the box too.
    const float scale = std::min(fitSize / std::max(size.y, 1e-6f), (fitSize * 1.15f) / std::max({size.x, size.z, 1e-6f}));
    const float3 pivot = make_float3((lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f);

    const unsigned int base = static_cast<unsigned int>(positions.size());
    positions.reserve(positions.size() + mesh.positions.size());
    normals.reserve(normals.size() + mesh.normals.size());
    for (size_t i = 0; i < mesh.positions.size(); ++i)
    {
        positions.push_back(floorCenter + offset + (orient(mesh.positions[i]) - pivot) * scale);
        normals.push_back(normalize(orient(mesh.normals[i])));  // uniform scale keeps normals valid
        texcoords.push_back(i < mesh.texcoords.size() ? mesh.texcoords[i] : make_float2(0.0f, 0.0f));
        const float4 mt = i < mesh.tangents.size() ? mesh.tangents[i] : make_float4(1.0f, 0.0f, 0.0f, 1.0f);
        const float3 rt = orient(make_float3(mt.x, mt.y, mt.z));
        tangents.push_back(make_float4(rt.x, rt.y, rt.z, mt.w));
    }
    indices.reserve(indices.size() + mesh.indices.size());
    for (const uint3& t : mesh.indices)
        indices.push_back(make_uint3(base + t.x, base + t.y, base + t.z));
    triangleMaterial.insert(triangleMaterial.end(), mesh.indices.size(), material);
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
    if (!options.model || options.showBlocks)
        s.addBlock(shortBase, 165.0f * kScale, shrt);
    const float3 tallBase[4] = {P(423.0f, 0.0f, 247.0f), P(265.0f, 0.0f, 296.0f), P(314.0f, 0.0f, 456.0f),
                                P(472.0f, 0.0f, 406.0f)};
    if (!options.model || options.showBlocks)
        s.addBlock(tallBase, 330.0f * kScale, tall);

    if (options.model)
    {
        Material m = ofType(options.modelMaterial);
        if (options.modelMaterial == MAT_DIFFUSE)
            m.albedo = make_float3(0.8f, 0.8f, 0.8f);  // light clay
        if (options.useTextures && options.model->hasTexcoords)
        {
            if (options.modelTextures.color && options.modelMaterial == MAT_DIFFUSE)
            {
                m.colorTexture = s.addTexture(options.modelTextures.color, true);
                m.albedo = make_float3(1.0f, 1.0f, 1.0f);
            }
            if (options.modelTextures.normal)
                m.normalTexture = s.addTexture(options.modelTextures.normal, false);
            m.flags = (options.flipV ? MATFLAG_FLIP_V : 0u) | (options.flipNormalGreen ? MATFLAG_FLIP_NORMAL_GREEN : 0u);
            m.normalStrength = options.normalStrength;
        }
        const unsigned int modelId = s.addMaterial(m);
        s.addFittedMesh(*options.model, modelId, P(278.0f, 0.0f, 279.6f), options.modelSize, options.modelZUp,
                        options.modelRotation, options.modelYaw, options.modelOffset);
    }

    s.camera.eye = P(278.0f, 273.0f, -800.0f);
    s.camera.lookAt = P(278.0f, 273.0f, 0.0f);
    s.camera.up = make_float3(0.0f, 1.0f, 0.0f);
    s.camera.verticalFovDegrees = 39.3f;
    return s;
}

void CornellOptions::rotateModel(int axis, float degrees)
{
    // Exact values for quarter turns, so repeated clicks never drift.
    float c, s;
    const float q = std::fmod(degrees, 360.0f);
    if (q == 90.0f || q == -270.0f) { c = 0.0f; s = 1.0f; }
    else if (q == -90.0f || q == 270.0f) { c = 0.0f; s = -1.0f; }
    else if (q == 180.0f || q == -180.0f) { c = -1.0f; s = 0.0f; }
    else { const float a = degrees * kPi / 180.0f; c = std::cos(a); s = std::sin(a); }

    float m[9];
    if (axis == 0)
        m[0] = 1, m[1] = 0, m[2] = 0, m[3] = 0, m[4] = c, m[5] = -s, m[6] = 0, m[7] = s, m[8] = c;
    else if (axis == 1)
        m[0] = c, m[1] = 0, m[2] = s, m[3] = 0, m[4] = 1, m[5] = 0, m[6] = -s, m[7] = 0, m[8] = c;
    else
        m[0] = c, m[1] = -s, m[2] = 0, m[3] = s, m[4] = c, m[5] = 0, m[6] = 0, m[7] = 0, m[8] = 1;

    // New = m * current: the turn happens around the world axis, on top of what's already there.
    float out[9];
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            out[row * 3 + col] = m[row * 3 + 0] * modelRotation[0 * 3 + col] + m[row * 3 + 1] * modelRotation[1 * 3 + col] +
                                 m[row * 3 + 2] * modelRotation[2 * 3 + col];
    std::copy(out, out + 9, modelRotation);
}

bool parseMaterialType(const std::string& name, MaterialType& out)
{
    if (name == "diffuse") { out = MAT_DIFFUSE; return true; }
    if (name == "mirror") { out = MAT_MIRROR; return true; }
    if (name == "glass") { out = MAT_GLASS; return true; }
    return false;
}

}  // namespace moa
