// MOA Renderer - displacement (see Displacement.h).
// SPDX-License-Identifier: Apache-2.0
#include "Displacement.h"

#include "device/vec_math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace moa {

namespace {

// Splits every triangle into 4 at its edge midpoints. Triangles sharing an edge (same two vertex
// indices) share the new midpoint vertex.
void subdivideOnce(Mesh& mesh)
{
    const size_t oldTris = mesh.indices.size();
    std::vector<uint3> tris;
    tris.reserve(oldTris * 4);
    std::unordered_map<uint64_t, uint32_t> midpoints;
    midpoints.reserve(oldTris * 2);
    auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
        const uint64_t key = (static_cast<uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
        auto it = midpoints.find(key);
        if (it != midpoints.end())
            return it->second;
        const uint32_t index = static_cast<uint32_t>(mesh.positions.size());
        // (pa + pb) * 0.5 is the same number whichever way round, so coincident edges of
        // separate triangles (UV seams) get bit-identical midpoints.
        const float3 pa = mesh.positions[a], pb = mesh.positions[b];
        mesh.positions.push_back((pa + pb) * 0.5f);
        const float3 n = mesh.normals[a] + mesh.normals[b];
        const float len = length(n);
        mesh.normals.push_back(len > 0.0f ? n / len : mesh.normals[a]);
        const float2 ta = mesh.texcoords[a], tb = mesh.texcoords[b];
        mesh.texcoords.push_back(make_float2((ta.x + tb.x) * 0.5f, (ta.y + tb.y) * 0.5f));
        midpoints.emplace(key, index);
        return index;
    };
    for (size_t i = 0; i < oldTris; ++i)
    {
        const uint3 t = mesh.indices[i];
        const uint32_t ab = midpoint(t.x, t.y), bc = midpoint(t.y, t.z), ca = midpoint(t.z, t.x);
        tris.push_back(make_uint3(t.x, ab, ca));
        tris.push_back(make_uint3(ab, t.y, bc));
        tris.push_back(make_uint3(ca, bc, t.z));
        tris.push_back(make_uint3(ab, bc, ca));
    }
    mesh.indices.swap(tris);
}

// Bilinear lookup of the red channel, wrapping like the GPU textures.
float sampleHeight(const HdrImage& map, float u, float v, bool flipV)
{
    const float tv = flipV ? v : 1.0f - v;  // image row 0 is the top
    const float x = u * map.width - 0.5f;
    const float y = tv * map.height - 0.5f;
    const float fx = std::floor(x), fy = std::floor(y);
    const float ax = x - fx, ay = y - fy;
    auto wrap = [](int i, int n) { return ((i % n) + n) % n; };
    const int x0 = wrap(static_cast<int>(fx), map.width), x1 = wrap(static_cast<int>(fx) + 1, map.width);
    const int y0 = wrap(static_cast<int>(fy), map.height), y1 = wrap(static_cast<int>(fy) + 1, map.height);
    auto at = [&](int px, int py) { return map.rgba[(static_cast<size_t>(py) * map.width + px) * 4]; };
    const float top = at(x0, y0) * (1.0f - ax) + at(x1, y0) * ax;
    const float bottom = at(x0, y1) * (1.0f - ax) + at(x1, y1) * ax;
    return top * (1.0f - ay) + bottom * ay;
}

bool positionLess(const float3& a, const float3& b)
{
    if (a.x != b.x)
        return a.x < b.x;
    if (a.y != b.y)
        return a.y < b.y;
    return a.z < b.z;
}

bool positionEqual(const float3& a, const float3& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

}  // namespace

DisplacementResult displaceMesh(Mesh& mesh, const HdrImage& heightMap, const DisplacementSettings& settings)
{
    DisplacementResult result;
    if (mesh.indices.empty() || heightMap.width <= 0 || heightMap.height <= 0 || heightMap.rgba.empty())
    {
        result.triangles = mesh.indices.size();
        return result;
    }
    if (mesh.normals.size() != mesh.positions.size())
        mesh.normals.assign(mesh.positions.size(), make_float3(0.0f, 1.0f, 0.0f));
    if (mesh.texcoords.size() != mesh.positions.size())
        mesh.texcoords.assign(mesh.positions.size(), make_float2(0.0f, 0.0f));

    // 1. Subdivide.
    int level = 0;
    while (level < settings.maxLevel && mesh.indices.size() * 4 <= settings.maxTriangles)
    {
        subdivideOnce(mesh);
        ++level;
    }
    const size_t vertexCount = mesh.positions.size();

    // 2. How far each vertex moves.
    const std::vector<float3> originalNormals = mesh.normals;
    std::vector<float3> offsets(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i)
    {
        const float h = sampleHeight(heightMap, mesh.texcoords[i].x, mesh.texcoords[i].y, settings.flipV) *
                            settings.scale + settings.offset;
        offsets[i] = originalNormals[i] * h;
    }

    // 3. Vertices at the same spot (UV seams, hard edges, separate triangles) move together,
    //    using the average of their offsets, so no cracks open up.
    std::vector<uint32_t> order(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i)
        order[i] = static_cast<uint32_t>(i);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return positionLess(mesh.positions[a], mesh.positions[b]); });
    std::vector<std::pair<uint32_t, uint32_t>> groups;  // [begin, end) ranges in `order`
    for (size_t i = 0; i < vertexCount;)
    {
        size_t j = i + 1;
        while (j < vertexCount && positionEqual(mesh.positions[order[i]], mesh.positions[order[j]]))
            ++j;
        groups.emplace_back(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
        i = j;
    }
    for (const auto& g : groups)
    {
        if (g.second - g.first == 1)
        {
            const uint32_t v = order[g.first];
            mesh.positions[v] = mesh.positions[v] + offsets[v];
            continue;
        }
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        for (uint32_t k = g.first; k < g.second; ++k)
            sum = sum + offsets[order[k]];
        const float3 avg = sum * (1.0f / static_cast<float>(g.second - g.first));
        for (uint32_t k = g.first; k < g.second; ++k)
            mesh.positions[order[k]] = mesh.positions[order[k]] + avg;
    }

    // 4. New smooth normals from the displaced shape. Corners that shared a smooth normal before
    //    are smoothed together; hard edges (very different original normals) stay hard.
    std::vector<float3> accum(vertexCount, make_float3(0.0f, 0.0f, 0.0f));
    for (const uint3& t : mesh.indices)
    {
        const float3 fn = cross(mesh.positions[t.y] - mesh.positions[t.x], mesh.positions[t.z] - mesh.positions[t.x]);
        accum[t.x] = accum[t.x] + fn;
        accum[t.y] = accum[t.y] + fn;
        accum[t.z] = accum[t.z] + fn;
    }
    for (const auto& g : groups)
    {
        for (uint32_t k = g.first; k < g.second; ++k)
        {
            const uint32_t v = order[k];
            float3 n = make_float3(0.0f, 0.0f, 0.0f);
            for (uint32_t m = g.first; m < g.second; ++m)
            {
                const uint32_t u = order[m];
                if (dot(originalNormals[u], originalNormals[v]) > 0.7f)
                    n = n + accum[u];
            }
            const float len = length(n);
            mesh.normals[v] = len > 0.0f ? n / len : originalNormals[v];
        }
    }

    computeTangents(mesh);
    result.level = level;
    result.triangles = mesh.indices.size();
    return result;
}

}  // namespace moa
