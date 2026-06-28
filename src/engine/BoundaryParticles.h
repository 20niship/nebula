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

  // OBJ ファイルを読み込み、表面を spacing 間隔でサンプリングして粒子配列を返す。
  // 最大 MAX_PARTICLES 個に制限する。
  std::vector<glm::vec4> loadOBJ(const std::string& path, float spacing);

  // transform 付きの OBJ 読み込み。
  // yup_to_zup=true の場合、先に (x,y,z)→(x,z,y) 座標変換を行う。
  // 変換順: yup_to_zup swap → scale 倍 → offset 加算
  BoundaryMesh loadOBJ(const std::string& path, float spacing, float scale, glm::vec3 offset, bool yup_to_zup);

private:
  void sampleTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, float spacing, std::vector<glm::vec4>& out);
};
