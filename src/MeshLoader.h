// MOA Renderer - loading triangle meshes from OBJ and STL files.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Image.h"

#include <vector_types.h>

#include <string>
#include <vector>

namespace moa {

// An indexed triangle mesh with one normal, texture coordinate and tangent per vertex.
struct Mesh
{
    std::vector<float3> positions;
    std::vector<float3> normals;    // same size as positions (filled by the loader)
    std::vector<float2> texcoords;  // same size as positions; (0,0) when the file has none
    std::vector<float4> tangents;   // xyz = tangent along +U, w = bitangent sign (for normal maps)
    std::vector<uint3> indices;     // one entry per triangle
    bool hasTexcoords = false;
    std::string name;               // file name, for display

    // Texture files found for this model: from its .mtl file, or images next to it whose names
    // start with the model's name. Not loaded yet (see loadModelTextures).
    std::string colorTexturePath;
    std::string normalTexturePath;
    std::vector<std::string> candidateTextures;

    size_t triangleCount() const { return indices.size(); }
};

// Textures picked for a model.
struct ModelTextures
{
    ImagePtr color;
    ImagePtr normal;
};

// Loads .obj or .stl (ASCII or binary), chosen by file extension, case-insensitive.
// Polygons are split into triangles. If the file has no normals, smooth normals are computed
// (area-weighted, with STL vertices welded first). Returns false and sets `error` on failure.
bool loadMesh(const std::string& path, Mesh& mesh, std::string& error);

// Recomputes smooth per-vertex normals from the triangles (area-weighted).
void computeSmoothNormals(Mesh& mesh);

// Computes per-vertex tangents from the texture coordinates (needed for normal maps).
void computeTangents(Mesh& mesh);

// Loads the textures found for a mesh. Candidate images are sorted by looking at them:
// light-blue images become the normal map, the first other image the colour map.
// `log` gets one line per texture used (or skipped).
ModelTextures loadModelTextures(const Mesh& mesh, std::string& log);

// True if the path has an image extension we can load (.jpg .jpeg .png .tga .bmp).
bool isImageFile(const std::string& path);

}  // namespace moa
