// MOA Renderer - loading triangle meshes from OBJ and STL files.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector_types.h>

#include <string>
#include <vector>

namespace moa {

// An indexed triangle mesh with one normal per vertex.
struct Mesh
{
    std::vector<float3> positions;
    std::vector<float3> normals;  // same size as positions (filled by the loader)
    std::vector<uint3> indices;   // one entry per triangle
    std::string name;             // file name, for display

    size_t triangleCount() const { return indices.size(); }
};

// Loads .obj or .stl (ASCII or binary), chosen by file extension, case-insensitive.
// Polygons are split into triangles. If the file has no normals, smooth normals are computed
// (area-weighted, with STL vertices welded first). Returns false and sets `error` on failure.
bool loadMesh(const std::string& path, Mesh& mesh, std::string& error);

// Recomputes smooth per-vertex normals from the triangles (area-weighted).
void computeSmoothNormals(Mesh& mesh);

}  // namespace moa
