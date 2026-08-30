#pragma once
#include <glm/glm.hpp>
#include <string>
#include <vector>

// OBJ からサンプリングした境界粒子と、描画用三角形頂点をまとめた構造体。
struct BoundaryMesh {
  std::vector<glm::vec4> particles; // GPU にアップロードする境界粒子 (xyzw)
  std::vector<glm::vec3> triVerts;  // 描画用三角形頂点 (3つで1三角形)
};

class BoundaryParticles {
public:
  static constexpr uint32_t MAX_PARTICLES = 50000;

  // transform 付きの OBJ 読み込み。
  // yup_to_zup=true の場合、先に (x,y,z)→(x,z,y) 座標変換を行う。
  // 変換順: yup_to_zup swap → scale 倍 → offset 加算
  BoundaryMesh loadOBJ(const std::string& path, float spacing, float scale, glm::vec3 offset, bool yup_to_zup);

private:
  void sampleTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, float spacing, std::vector<glm::vec4>& out);
};

// obj からメッシュ表面をサンプリングし境界粒子+描画用三角形を生成する(BoundaryParticles::loadOBJ の薄いラッパー)。
BoundaryMesh generate_particles_from_mesh_surface(const std::string& obj, float spacing, float scale = 1.0f, glm::vec3 offset = glm::vec3(0.0f), bool yup_to_zup = false);
