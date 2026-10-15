// MOA Renderer - loading triangle meshes from OBJ and STL files.
// SPDX-License-Identifier: Apache-2.0
#include "MeshLoader.h"

#include "device/vec_math.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace moa {

namespace {

bool readFile(const std::string& path, std::vector<char>& data)
{
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!f)
        return false;
    const std::streamsize size = f.tellg();
    f.seekg(0);
    data.resize(static_cast<size_t>(size) + 1);
    if (size > 0 && !f.read(data.data(), size))
        return false;
    data[static_cast<size_t>(size)] = '\0';  // lets the parsers use strtof safely
    data.resize(static_cast<size_t>(size));
    return true;
}

std::string lowerExtension(const std::string& path)
{
    std::string ext = std::filesystem::u8path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// ---------------------------------------------------------------------------------------- OBJ

const char* skipSpaces(const char* p)
{
    while (*p == ' ' || *p == '\t')
        ++p;
    return p;
}

const char* nextLine(const char* p, const char* end)
{
    while (p < end && *p != '\n')
        ++p;
    return p < end ? p + 1 : end;
}

// Fast float parser (std::from_chars is several times quicker than strtof on big files).
// Skips leading spaces, reads one number, advances p. Returns 0 if there's no number.
float parseFloat(const char*& p, const char* end)
{
    while (p < end && (*p == ' ' || *p == '\t'))
        ++p;
    if (p < end && *p == '+')
        ++p;  // from_chars doesn't accept a leading '+'
    float value = 0.0f;
    const auto result = std::from_chars(p, end, value);
    if (result.ec == std::errc())
        p = result.ptr;
    return value;
}

// OBJ indices are 1-based; negative values count back from the end.
bool resolveIndex(long idx, size_t count, uint32_t& out)
{
    if (idx > 0 && static_cast<size_t>(idx) <= count)
        out = static_cast<uint32_t>(idx - 1);
    else if (idx < 0 && static_cast<size_t>(-idx) <= count)
        out = static_cast<uint32_t>(count + idx);
    else
        return false;
    return true;
}

bool loadObj(const std::vector<char>& data, Mesh& mesh, std::string& error)
{
    std::vector<float3> filePositions;
    std::vector<float3> fileNormals;
    struct Corner { uint32_t v; uint32_t vn; bool hasNormal; };
    std::vector<Corner> corners;  // one per triangle corner, filled while parsing faces
    bool allHaveNormals = true;
    std::vector<Corner> face;

    const char* p = data.data();
    const char* end = p + data.size();
    size_t lineNumber = 0;
    while (p < end)
    {
        ++lineNumber;
        const char* line = skipSpaces(p);
        p = nextLine(p, end);
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t'))
        {
            const char* q = line + 2;
            float3 v;
            v.x = parseFloat(q, end);
            v.y = parseFloat(q, end);
            v.z = parseFloat(q, end);
            filePositions.push_back(v);
        }
        else if (line[0] == 'v' && line[1] == 'n' && (line[2] == ' ' || line[2] == '\t'))
        {
            const char* q = line + 3;
            float3 n;
            n.x = parseFloat(q, end);
            n.y = parseFloat(q, end);
            n.z = parseFloat(q, end);
            fileNormals.push_back(n);
        }
        else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t'))
        {
            face.clear();
            const char* q = line + 2;
            while (true)
            {
                q = skipSpaces(q);
                if (*q == '\0' || *q == '\n' || *q == '\r' || *q == '#')
                    break;
                char* after = nullptr;
                const long vi = std::strtol(q, &after, 10);
                if (after == q)
                    break;
                q = after;
                long ni = 0;
                if (*q == '/')
                {
                    ++q;
                    if (*q != '/')
                    {
                        std::strtol(q, &after, 10);  // texture coordinate index (not used yet)
                        q = after;
                    }
                    if (*q == '/')
                    {
                        ++q;
                        ni = std::strtol(q, &after, 10);
                        q = after;
                    }
                }
                Corner c{};
                if (!resolveIndex(vi, filePositions.size(), c.v))
                {
                    error = "Bad vertex index on line " + std::to_string(lineNumber);
                    return false;
                }
                c.hasNormal = ni != 0 && resolveIndex(ni, fileNormals.size(), c.vn);
                allHaveNormals &= c.hasNormal;
                face.push_back(c);
                while (*q && *q != ' ' && *q != '\t' && *q != '\n' && *q != '\r')
                    ++q;  // skip anything unexpected in this corner
            }
            // Fan-triangulate polygons (fine for the convex quads/ngons scans and models use).
            for (size_t i = 1; i + 1 < face.size(); ++i)
            {
                corners.push_back(face[0]);
                corners.push_back(face[i]);
                corners.push_back(face[i + 1]);
            }
        }
    }

    if (corners.empty())
    {
        error = "No faces found in the OBJ file";
        return false;
    }

    if (allHaveNormals)
    {
        // Each unique (position, normal) pair becomes one vertex.
        std::unordered_map<uint64_t, uint32_t> remap;
        remap.reserve(corners.size());
        std::vector<uint32_t> idx(corners.size());
        for (size_t i = 0; i < corners.size(); ++i)
        {
            const uint64_t key = (static_cast<uint64_t>(corners[i].v) << 32) | corners[i].vn;
            auto [it, inserted] = remap.try_emplace(key, static_cast<uint32_t>(mesh.positions.size()));
            if (inserted)
            {
                mesh.positions.push_back(filePositions[corners[i].v]);
                mesh.normals.push_back(normalize(fileNormals[corners[i].vn]));
            }
            idx[i] = it->second;
        }
        for (size_t i = 0; i < idx.size(); i += 3)
            mesh.indices.push_back(make_uint3(idx[i], idx[i + 1], idx[i + 2]));
    }
    else
    {
        // No (or partial) normals: share positions and compute smooth normals.
        mesh.positions = std::move(filePositions);
        for (size_t i = 0; i < corners.size(); i += 3)
            mesh.indices.push_back(make_uint3(corners[i].v, corners[i + 1].v, corners[i + 2].v));
        computeSmoothNormals(mesh);
    }
    return true;
}

// ---------------------------------------------------------------------------------------- STL

struct WeldKey
{
    uint32_t x, y, z;
    bool operator==(const WeldKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct WeldHash
{
    size_t operator()(const WeldKey& k) const
    {
        return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^
               (static_cast<size_t>(k.z) * 83492791u);
    }
};

// STL stores three separate corners per triangle. Merge identical positions so smooth normals work.
void addWeldedTriangle(Mesh& mesh, std::unordered_map<WeldKey, uint32_t, WeldHash>& weld, const float3 v[3])
{
    uint32_t idx[3];
    for (int i = 0; i < 3; ++i)
    {
        WeldKey key;
        std::memcpy(&key.x, &v[i].x, 4);
        std::memcpy(&key.y, &v[i].y, 4);
        std::memcpy(&key.z, &v[i].z, 4);
        auto [it, inserted] = weld.try_emplace(key, static_cast<uint32_t>(mesh.positions.size()));
        if (inserted)
            mesh.positions.push_back(v[i]);
        idx[i] = it->second;
    }
    if (idx[0] != idx[1] && idx[1] != idx[2] && idx[0] != idx[2])  // skip collapsed triangles
        mesh.indices.push_back(make_uint3(idx[0], idx[1], idx[2]));
}

bool loadStl(const std::vector<char>& data, Mesh& mesh, std::string& error)
{
    std::unordered_map<WeldKey, uint32_t, WeldHash> weld;

    // Binary STL: 80-byte header, uint32 triangle count, then 50 bytes per triangle.
    if (data.size() >= 84)
    {
        uint32_t count = 0;
        std::memcpy(&count, data.data() + 80, 4);
        if (data.size() == 84 + static_cast<size_t>(count) * 50)
        {
            weld.reserve(count);
            mesh.positions.reserve(count / 2 + 16);
            mesh.indices.reserve(count);
            for (uint32_t t = 0; t < count; ++t)
            {
                const char* rec = data.data() + 84 + static_cast<size_t>(t) * 50;
                float3 v[3];
                std::memcpy(v, rec + 12, sizeof(v));  // skip the stored facet normal
                addWeldedTriangle(mesh, weld, v);
            }
            computeSmoothNormals(mesh);
            return !mesh.indices.empty() || (error = "STL file has no triangles", false);
        }
    }

    // ASCII STL: "vertex x y z" lines, three per facet.
    const char* p = data.data();
    const char* end = p + data.size();
    float3 v[3];
    int n = 0;
    while (p < end)
    {
        const char* line = p;
        while (line < end && (*line == ' ' || *line == '\t'))
            ++line;
        p = nextLine(p, end);
        if (end - line > 7 && std::strncmp(line, "vertex", 6) == 0)
        {
            char* q = nullptr;
            v[n].x = std::strtof(line + 6, &q);
            v[n].y = std::strtof(q, &q);
            v[n].z = std::strtof(q, &q);
            if (++n == 3)
            {
                addWeldedTriangle(mesh, weld, v);
                n = 0;
            }
        }
    }
    if (mesh.indices.empty())
    {
        error = "STL file has no triangles (or isn't a valid STL)";
        return false;
    }
    computeSmoothNormals(mesh);
    return true;
}

}  // namespace

void computeSmoothNormals(Mesh& mesh)
{
    mesh.normals.assign(mesh.positions.size(), make_float3(0.0f, 0.0f, 0.0f));
    for (const uint3& t : mesh.indices)
    {
        const float3 a = mesh.positions[t.x], b = mesh.positions[t.y], c = mesh.positions[t.z];
        const float3 n = cross(b - a, c - a);  // length = 2 x area, so bigger triangles count more
        mesh.normals[t.x] += n;
        mesh.normals[t.y] += n;
        mesh.normals[t.z] += n;
    }
    for (float3& n : mesh.normals)
    {
        const float len = length(n);
        n = len > 0.0f ? n / len : make_float3(0.0f, 1.0f, 0.0f);
    }
}

bool loadMesh(const std::string& path, Mesh& mesh, std::string& error)
{
    mesh = Mesh{};
    mesh.name = std::filesystem::u8path(path).filename().string();
    std::vector<char> data;
    if (!readFile(path, data))
    {
        error = "Couldn't open " + path;
        return false;
    }
    const std::string ext = lowerExtension(path);
    bool ok = false;
    if (ext == ".obj")
        ok = loadObj(data, mesh, error);
    else if (ext == ".stl")
        ok = loadStl(data, mesh, error);
    else
        error = "Unsupported file type '" + ext + "' (use .obj or .stl)";
    if (!ok)
        mesh = Mesh{};
    return ok;
}

}  // namespace moa
