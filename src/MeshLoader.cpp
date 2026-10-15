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

struct ObjCorner
{
    uint32_t v, vt, vn;
    bool hasTexcoord, hasNormal;
};

struct CornerKey
{
    uint32_t v, vt, vn;
    bool operator==(const CornerKey& o) const { return v == o.v && vt == o.vt && vn == o.vn; }
};
struct CornerHash
{
    size_t operator()(const CornerKey& k) const
    {
        return (static_cast<size_t>(k.v) * 73856093u) ^ (static_cast<size_t>(k.vt) * 19349663u) ^
               (static_cast<size_t>(k.vn) * 83492791u);
    }
};

bool loadObj(const std::vector<char>& data, Mesh& mesh, std::string& error, std::string& mtlFile)
{
    std::vector<float3> filePositions;
    std::vector<float3> fileNormals;
    std::vector<float2> fileTexcoords;
    std::vector<ObjCorner> corners;  // three per triangle
    bool allHaveNormals = true;
    bool allHaveTexcoords = true;
    std::vector<ObjCorner> face;

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
        else if (line[0] == 'v' && line[1] == 't' && (line[2] == ' ' || line[2] == '\t'))
        {
            const char* q = line + 3;
            float2 t;
            t.x = parseFloat(q, end);
            t.y = parseFloat(q, end);
            fileTexcoords.push_back(t);
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
                long ti = 0, ni = 0;
                if (*q == '/')
                {
                    ++q;
                    if (*q != '/')
                    {
                        ti = std::strtol(q, &after, 10);
                        q = after;
                    }
                    if (*q == '/')
                    {
                        ++q;
                        ni = std::strtol(q, &after, 10);
                        q = after;
                    }
                }
                ObjCorner c{};
                if (!resolveIndex(vi, filePositions.size(), c.v))
                {
                    error = "Bad vertex index on line " + std::to_string(lineNumber);
                    return false;
                }
                c.hasTexcoord = ti != 0 && resolveIndex(ti, fileTexcoords.size(), c.vt);
                c.hasNormal = ni != 0 && resolveIndex(ni, fileNormals.size(), c.vn);
                allHaveTexcoords &= c.hasTexcoord;
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
        else if (std::strncmp(line, "mtllib", 6) == 0 && (line[6] == ' ' || line[6] == '\t'))
        {
            const char* q = skipSpaces(line + 7);
            const char* e = q;
            while (e < end && *e != '\n' && *e != '\r')
                ++e;
            mtlFile.assign(q, e);
            while (!mtlFile.empty() && (mtlFile.back() == ' ' || mtlFile.back() == '\t'))
                mtlFile.pop_back();
        }
    }

    if (corners.empty())
    {
        error = "No faces found in the OBJ file";
        return false;
    }

    const bool useNormals = allHaveNormals && !fileNormals.empty();
    const bool useTexcoords = allHaveTexcoords && !fileTexcoords.empty();
    mesh.hasTexcoords = useTexcoords;

    if (!useNormals && !useTexcoords)
    {
        // Positions only: share them and compute smooth normals.
        mesh.positions = std::move(filePositions);
        for (size_t i = 0; i < corners.size(); i += 3)
            mesh.indices.push_back(make_uint3(corners[i].v, corners[i + 1].v, corners[i + 2].v));
        computeSmoothNormals(mesh);
        mesh.texcoords.assign(mesh.positions.size(), make_float2(0.0f, 0.0f));
        return true;
    }

    // Each unique (position, texcoord, normal) combination becomes one vertex. UV seams split
    // vertices, so smooth normals are computed per original position and shared across seams.
    std::unordered_map<CornerKey, uint32_t, CornerHash> remap;
    remap.reserve(corners.size());
    std::vector<uint32_t> sourcePosition;
    std::vector<uint32_t> idx(corners.size());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        const ObjCorner& c = corners[i];
        const CornerKey key{c.v, useTexcoords ? c.vt : 0u, useNormals ? c.vn : 0u};
        auto [it, inserted] = remap.try_emplace(key, static_cast<uint32_t>(mesh.positions.size()));
        if (inserted)
        {
            mesh.positions.push_back(filePositions[c.v]);
            mesh.texcoords.push_back(useTexcoords ? fileTexcoords[c.vt] : make_float2(0.0f, 0.0f));
            if (useNormals)
                mesh.normals.push_back(normalize(fileNormals[c.vn]));
            sourcePosition.push_back(c.v);
        }
        idx[i] = it->second;
    }
    for (size_t i = 0; i < idx.size(); i += 3)
        mesh.indices.push_back(make_uint3(idx[i], idx[i + 1], idx[i + 2]));

    if (!useNormals)
    {
        Mesh shared;
        shared.positions = std::move(filePositions);
        for (size_t i = 0; i < corners.size(); i += 3)
            shared.indices.push_back(make_uint3(corners[i].v, corners[i + 1].v, corners[i + 2].v));
        computeSmoothNormals(shared);
        mesh.normals.resize(mesh.positions.size());
        for (size_t i = 0; i < mesh.positions.size(); ++i)
            mesh.normals[i] = shared.normals[sourcePosition[i]];
    }
    return true;
}

// Reads map_Kd (colour) and map_Bump / bump / norm (normal) from an .mtl file. Paths are
// resolved relative to the .mtl file.
void readMtl(const std::filesystem::path& mtlPath, Mesh& mesh)
{
    std::vector<char> data;
    if (!readFile(mtlPath.u8string(), data))
        return;
    const char* p = data.data();
    const char* end = p + data.size();
    while (p < end)
    {
        const char* line = skipSpaces(p);
        p = nextLine(p, end);
        const char* e = line;
        while (e < end && *e != '\n' && *e != '\r')
            ++e;
        std::string text(line, e);
        auto lastToken = [&](size_t from) {
            // Options like "-bm 1.0" may come first; the file name is the last word.
            std::string rest = text.substr(from);
            while (!rest.empty() && (rest.back() == ' ' || rest.back() == '\t'))
                rest.pop_back();
            const size_t space = rest.find_last_of(" \t");
            return space == std::string::npos ? rest : rest.substr(space + 1);
        };
        auto resolve = [&](const std::string& name) { return (mtlPath.parent_path() / std::filesystem::u8path(name)).u8string(); };
        if (text.rfind("map_Kd ", 0) == 0 && mesh.colorTexturePath.empty())
            mesh.colorTexturePath = resolve(lastToken(7));
        else if ((text.rfind("map_Bump ", 0) == 0 || text.rfind("map_bump ", 0) == 0) && mesh.normalTexturePath.empty())
            mesh.normalTexturePath = resolve(lastToken(9));
        else if ((text.rfind("bump ", 0) == 0 || text.rfind("norm ", 0) == 0) && mesh.normalTexturePath.empty())
            mesh.normalTexturePath = resolve(lastToken(5));
    }
}

std::string lowerString(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Finds texture files for a model: from its .mtl if it has one, plus any images in the same
// folder whose names start with the model's name.
void findTextures(const std::string& modelPath, const std::string& mtlFile, Mesh& mesh)
{
    namespace fs = std::filesystem;
    const fs::path model = fs::u8path(modelPath);
    const fs::path folder = model.parent_path().empty() ? fs::path(".") : model.parent_path();
    if (!mtlFile.empty())
        readMtl(folder / fs::u8path(mtlFile), mesh);
    std::error_code ec;
    if (!fs::exists(mesh.colorTexturePath, ec))
        mesh.colorTexturePath.clear();
    if (!fs::exists(mesh.normalTexturePath, ec))
        mesh.normalTexturePath.clear();

    const std::string stem = lowerString(model.stem().u8string());
    for (const fs::directory_entry& entry : fs::directory_iterator(folder, ec))
    {
        if (!entry.is_regular_file(ec) || !isImageFile(entry.path().u8string()))
            continue;
        if (lowerString(entry.path().stem().u8string()).rfind(stem, 0) == 0)
            mesh.candidateTextures.push_back(entry.path().u8string());
    }
    std::sort(mesh.candidateTextures.begin(), mesh.candidateTextures.end());
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

void computeTangents(Mesh& mesh)
{
    const size_t n = mesh.positions.size();
    std::vector<float3> tan(n, make_float3(0.0f, 0.0f, 0.0f));
    std::vector<float3> bit(n, make_float3(0.0f, 0.0f, 0.0f));
    if (mesh.hasTexcoords)
    {
        for (const uint3& t : mesh.indices)
        {
            const float3 p0 = mesh.positions[t.x], p1 = mesh.positions[t.y], p2 = mesh.positions[t.z];
            const float2 w0 = mesh.texcoords[t.x], w1 = mesh.texcoords[t.y], w2 = mesh.texcoords[t.z];
            const float3 e1 = p1 - p0, e2 = p2 - p0;
            const float du1 = w1.x - w0.x, dv1 = w1.y - w0.y, du2 = w2.x - w0.x, dv2 = w2.y - w0.y;
            const float det = du1 * dv2 - du2 * dv1;
            if (std::fabs(det) < 1e-12f)
                continue;
            const float r = 1.0f / det;
            const float3 sdir = (e1 * dv2 - e2 * dv1) * r;  // direction of increasing U
            const float3 tdir = (e2 * du1 - e1 * du2) * r;  // direction of increasing V
            for (uint32_t i : {t.x, t.y, t.z})
            {
                tan[i] += sdir;
                bit[i] += tdir;
            }
        }
    }
    mesh.tangents.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        const float3 nrm = mesh.normals[i];
        float3 t = tan[i] - nrm * dot(nrm, tan[i]);  // make it perpendicular to the normal
        if (dot(t, t) < 1e-20f)
        {
            // No usable UVs here: pick any direction perpendicular to the normal.
            t = std::fabs(nrm.x) < 0.9f ? cross(nrm, make_float3(1.0f, 0.0f, 0.0f)) : cross(nrm, make_float3(0.0f, 1.0f, 0.0f));
        }
        t = normalize(t);
        const float w = dot(cross(nrm, t), bit[i]) < 0.0f ? -1.0f : 1.0f;
        mesh.tangents[i] = make_float4(t.x, t.y, t.z, w);
    }
}

bool isImageFile(const std::string& path)
{
    const std::string ext = lowerExtension(path);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tga" || ext == ".bmp";
}

ModelTextures loadModelTextures(const Mesh& mesh, std::string& log)
{
    ModelTextures out;
    std::string error;
    auto note = [&](const std::string& line) { log += line + "\n"; };
    if (!mesh.colorTexturePath.empty())
    {
        out.color = loadImage(mesh.colorTexturePath, error);
        note(out.color ? "Colour (from .mtl): " + out.color->name : "Colour texture failed: " + error);
    }
    if (!mesh.normalTexturePath.empty())
    {
        out.normal = loadImage(mesh.normalTexturePath, error);
        note(out.normal ? "Normal map (from .mtl): " + out.normal->name : "Normal map failed: " + error);
    }
    for (const std::string& path : mesh.candidateTextures)
    {
        if ((out.color && out.normal) || path == mesh.colorTexturePath || path == mesh.normalTexturePath)
            continue;
        ImagePtr image = loadImage(path, error);
        if (!image)
        {
            note("Skipped " + path + ": " + error);
            continue;
        }
        if (looksLikeNormalMap(*image))
        {
            if (!out.normal)
            {
                out.normal = image;
                note("Normal map: " + image->name);
            }
        }
        else if (!out.color)
        {
            out.color = image;
            note("Colour: " + image->name);
        }
    }
    return out;
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
    std::string mtlFile;
    if (ext == ".obj")
        ok = loadObj(data, mesh, error, mtlFile);
    else if (ext == ".stl")
        ok = loadStl(data, mesh, error);
    else
        error = "Unsupported file type '" + ext + "' (use .obj or .stl)";
    if (!ok)
    {
        mesh = Mesh{};
        return false;
    }
    if (mesh.texcoords.size() != mesh.positions.size())
        mesh.texcoords.assign(mesh.positions.size(), make_float2(0.0f, 0.0f));
    computeTangents(mesh);
    if (mesh.hasTexcoords)
        findTextures(path, mtlFile, mesh);
    return true;
}

}  // namespace moa
