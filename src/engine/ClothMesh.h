#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

// N×N グリッド布メッシュ。エッジをグラフ彩色して GPU 拘束ソルバー用データを生成する。
class ClothMesh {
public:
  int gridN     = 128;
  float spacing = 0.07f; // 頂点間隔 (world units)
  float mass    = 1.0f;

  // GPU アップロード用データ
  std::vector<glm::vec4> positions; // 初期位置 (XZ平面、Y=worldMax付近)
  std::vector<glm::vec4> invMasses; // .x = invMass (ピン留め=0)
  std::vector<uint32_t> typeFlags;  // 全頂点 typeFlag=2 (Cloth)

  // エッジバッファ: stride=3 uints per edge [p, q, floatBitsToUint(restLen)]
  std::vector<uint32_t> edgeData;
  // カラーバッチ境界: [start_0, start_1, ..., total_edge_count]
  std::vector<uint32_t> colorBatch;
  int nColors = 0;

  // 三角形インデックス (描画用)
  std::vector<uint32_t> triIndices;

  // メッシュを生成する。invMass は全頂点 1/mass で初期化（ピン留めは外部の制約として設定）
  void build(int n, float spacing, float originX, float originY, float originZ);

  // グリッド座標 (i,j) → 頂点ローカルインデックス (public: ClothSceneEngine から参照)
  int idx(int i, int j) const { return i * gridN + j; }

  int vertexCount() const { return gridN * gridN; }
  int edgeCount() const { return (int)edgeData.size() / 3; }

private:
  // エッジ種別ごとに別の色グループを使う
  // stretch_h, stretch_v, shear_a, shear_b, bend_h, bend_v の 6 種 × 2 色 = 12 色
  enum class EdgeType { StretchH, StretchV, ShearA, ShearB, BendH, BendV };

  void addEdge(uint32_t p, uint32_t q, float restLen, int color);
  void buildEdgesAndColoring();
  void buildTriIndices();

  // color → エッジリスト (一時バッファ、build後は edgeData/colorBatch に統合)
  std::vector<std::vector<uint32_t>> coloredEdges_; // per-color raw edge data
};
