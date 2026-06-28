#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "BoundaryParticles.h"

#include <cmath>
#include <stdexcept>

std::vector<glm::vec4> BoundaryParticles::loadOBJ(const std::string& path, float spacing) {
  tinyobj::ObjReaderConfig cfg;
  cfg.triangulate = true;

  tinyobj::ObjReader reader;
  if(!reader.ParseFromFile(path, cfg)) {
    throw std::runtime_error("BoundaryParticles: " + reader.Error());
  }

  std::vector<glm::vec4> out;
  out.reserve(MAX_PARTICLES);

  const auto& attrib = reader.GetAttrib();
  const auto& verts  = attrib.vertices;

  for(const auto& shape : reader.GetShapes()) {
    size_t idxOff = 0;
    for(size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
      auto iv0 = shape.mesh.indices[idxOff + 0];
      auto iv1 = shape.mesh.indices[idxOff + 1];
      auto iv2 = shape.mesh.indices[idxOff + 2];
      idxOff += 3;

      glm::vec3 v0(verts[3 * iv0.vertex_index], verts[3 * iv0.vertex_index + 1], verts[3 * iv0.vertex_index + 2]);
      glm::vec3 v1(verts[3 * iv1.vertex_index], verts[3 * iv1.vertex_index + 1], verts[3 * iv1.vertex_index + 2]);
      glm::vec3 v2(verts[3 * iv2.vertex_index], verts[3 * iv2.vertex_index + 1], verts[3 * iv2.vertex_index + 2]);

      sampleTriangle(v0, v1, v2, spacing, out);
      if(out.size() >= MAX_PARTICLES) break;
    }
    if(out.size() >= MAX_PARTICLES) break;
  }

  return out;
}

BoundaryMesh BoundaryParticles::loadOBJ(const std::string& path, float spacing, float scale, glm::vec3 offset, bool yup_to_zup) {
  tinyobj::ObjReaderConfig cfg;
  cfg.triangulate = true;

  tinyobj::ObjReader reader;
  if(!reader.ParseFromFile(path, cfg)) {
    throw std::runtime_error("BoundaryParticles: " + reader.Error());
  }

  BoundaryMesh result;
  result.particles.reserve(MAX_PARTICLES);

  const auto& attrib = reader.GetAttrib();
  const auto& verts  = attrib.vertices;

  auto transform = [&](float x, float y, float z) -> glm::vec3 {
    glm::vec3 p(x, y, z);
    if(yup_to_zup) std::swap(p.y, p.z);
    p *= scale;
    p += offset;
    return p;
  };

  for(const auto& shape : reader.GetShapes()) {
    size_t idxOff = 0;
    for(size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
      auto iv0 = shape.mesh.indices[idxOff + 0];
      auto iv1 = shape.mesh.indices[idxOff + 1];
      auto iv2 = shape.mesh.indices[idxOff + 2];
      idxOff += 3;

      glm::vec3 v0 = transform(verts[3 * iv0.vertex_index], verts[3 * iv0.vertex_index + 1], verts[3 * iv0.vertex_index + 2]);
      glm::vec3 v1 = transform(verts[3 * iv1.vertex_index], verts[3 * iv1.vertex_index + 1], verts[3 * iv1.vertex_index + 2]);
      glm::vec3 v2 = transform(verts[3 * iv2.vertex_index], verts[3 * iv2.vertex_index + 1], verts[3 * iv2.vertex_index + 2]);

      result.triVerts.push_back(v0);
      result.triVerts.push_back(v1);
      result.triVerts.push_back(v2);

      sampleTriangle(v0, v1, v2, spacing, result.particles);
      if(result.particles.size() >= MAX_PARTICLES) break;
    }
    if(result.particles.size() >= MAX_PARTICLES) break;
  }

  return result;
}

void BoundaryParticles::sampleTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, float spacing, std::vector<glm::vec4>& out) {
  glm::vec3 e1 = b - a;
  glm::vec3 e2 = c - a;

  int n1 = std::max(1, (int)(glm::length(e1) / spacing));
  int n2 = std::max(1, (int)(glm::length(e2) / spacing));

  for(int i = 0; i <= n1; ++i) {
    float u = float(i) / float(n1);
    for(int j = 0; j <= n2; ++j) {
      float v = float(j) / float(n2);
      if(u + v > 1.0f) continue;
      out.push_back(glm::vec4(a + u * e1 + v * e2, 1.0f));
      if(out.size() >= MAX_PARTICLES) return;
    }
  }
}
