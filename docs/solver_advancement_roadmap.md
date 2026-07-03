# MPM / FLIP / XPBD / PBF / Pyro 発展ロードマップ

現状実装(Vulkan compute, bindless SSBO, Morton格子ベース)を起点に、**高速化・精度向上・大規模化(粒子数/グリッド)・機能追加**の4軸で各ソルバーをどう発展させられるかを、直近(2023〜2026年前半)の研究動向を踏まえてまとめる。commitは行わず、本ドキュメントのみを追加する。

## 目次
1. [現状アーキテクチャの要点と共通のボトルネック](#1-現状アーキテクチャの要点と共通のボトルネック)
2. [MPM / FLIP](#2-mpm--flip-srcenginempmengine)
3. [XPBD ソフトボディ](#3-xpbd-ソフトボディ-srcenginesoftbodyengine)
4. [PBF 流体](#4-pbf-流体-srcenginefluidengine)
5. [Pyro (グリッドベース煙・火炎)](#5-pyro-グリッドベース煙火炎-srcenginepyroengine)
6. [横断的インフラ改善](#6-横断的インフラ改善-全ソルバー共通)
7. [優先度つきロードマップ案](#7-優先度つきロードマップ案)
8. [参考文献](#8-参考文献-2023-2026-中心)

---

## 1. 現状アーキテクチャの要点と共通のボトルネック

- **Bindless SSBO + AttributeBuffer**: `src/core/AttributeBuffer.h` で `MAX_BINDLESS_BUFFERS = 16` に固定。
  MoltenVK (Apple GPU) の `maxPerStageDescriptorStorageBuffers` 制約に由来する値だが、機能追加のたびに
  この上限に近づいており(Pyroは既に velA/B, densA/B, tempA/B, fuelA/B, flame, presA/B, div, curl,
  colliderSDF, pyroSources で14枠使用)、今後の拡張のボトルネックになりつつある。
- **単一 Dense グリッド (Morton Z-order)**: MPM/Pyro とも `grid_res³` の密な配列を確保する。
  空間の大部分が空でも同じメモリ・計算コストがかかり、大規模化の直接的な壁になっている。
  Pyroは2のべき乗のgrid_resが必須という制約もある(Morton decode の未定義動作を避けるため、
  `PyroEngine::init()` で検証済み — `src/engine/PyroEngine.h:15-18`)。
- **MoltenVK 由来のシェーダー制約**: `buffers[]` へのアクセスは `main()` 内でのみ許可され、関数分割や
  ヘルパー抽象化ができない (`shaders/mpm_common.glsl:72-73` 等に明記)。これはコード保守性だけでなく、
  将来的な複雑な数値ソルバー (Newton法・多重格子など) をシェーダー側に実装する際の大きな障壁になる。
- **全て陽解法・単一キュー**: MPM/PBF/XPBD/Pyroいずれも明示的な時間積分 (陽的オイラー相当) で、
  暗黙解法・準ニュートン法の類は未実装。計算キューも単一の compute queue のみで、
  シミュレーションとレンダリング/複数シム間のオーバーラップは timeline semaphore による
  フレーム間パイプライン化のみ (`src/App.cpp` 周辺) に留まる。
- **オフラインシェーダーコンパイル**: `glslc` で `.comp` を `.spv` に事前コンパイルする方式
  (`CMakeLists.txt` の `SHADER_SOURCES`)。実行時のシェーダー特殊化 (specialization constants) は
  未使用で、grid_res や material数などをコンパイル時定数化する余地がある。

これらは5ソルバー共通の制約であり、個別の改善に入る前に **6章の横断的インフラ改善**が効果的に効いてくる領域が多い。

---

## 2. MPM / FLIP (`src/engine/MPMEngine.*`)

### 現状
- P2G/G2P はハッシュグリッド (`hash_count`→`hash_scan_local/global`→`hash_add_base`→`hash_sort`) 経由。
- 転写スキームは `flip_ratio` (`MPMEngine.h:74`) で PIC(0) / APIC(-1) / FLIP(0~1) を切り替え可能。
- マテリアルモデルは ELASTIC / VON_MISES / DRUCKER_PRAGER / GRANULAR_POWDER / FLUID /
  VISCOPLASTIC_MUD の6種 (`shaders/mpm_common.glsl:122-127`)、Hencky弾性+SVDベースのreturn mapping。
- 障害物は解析コライダー (`ColliderSet`, 平面/球/箱/カプセル) と任意メッシュSDF
  (`setColliderSDF`, Morton密配列) の二本立て。
- 粒子数は既存デモで最大 ~80K (`mpm_avalanche`)、grid_res は64〜128。

### 高速化
- **G2P2G カーネル融合**: 現状 P2G→GridUpdate→G2P は3回のディスパッチに分かれ、グリッドバッファへの
  書き込み・読み出しをまたぐ。Wang et al. のマルチGPU MPM (SIGGRAPH 2020, [Zhu et al.](https://yzhu.io/publication/mpmgpu2020siggraph/)) が
  示した G2P2G 融合 (前ステップのG2Pと当該ステップのP2Gを1カーネルに統合しグローバルメモリの
  往復を削減) は Vulkan compute でも有効。MoltenVKの `buffers[]` 制約下でも1つの `main()` 内で
  両方のロジックをインライン展開すれば適用可能。
- **座標コアレッシング・アトミック削減**: 現行はハッシュソート後の `sortedIdx` 経由で P2G を行っており
  atomic add は使っていない設計だが、粒子データのメモリレイアウト (SoA vs AoS) を
  ウォープ/サブグループ単位でコアレッシングされるよう再配置する余地がある
  (`src/core/AttributeBuffer.cpp` のバッファ確保順を見直す)。
- **Specialization Constants によるシェーダー特殊化**: `grid_res`, `materialCount` 等を
  push constant ではなく `layout(constant_id=...)` のspecialization constantにし、
  コンパイラに定数畳み込みさせることで分岐削減・レジスタ圧縮が期待できる。
- **サブステップ数の適応化**: 現状 `numSubsteps` は固定値 (例: 20〜30)。CFL条件から動的に
  必要最小限のサブステップ数を求める適応タイムステップ (arXiv 2508.11722,
  [Substepping the Material Point Method (2025)](https://arxiv.org/pdf/2508.11722)) を導入すれば、
  粒子速度が小さい間はサブステップを削減し高速化できる。

### 精度向上
- **MLS-MPM 系の Affine Projection Stabilizer (APS, 2025)**: 明示的MLS-MPMの安定性を、
  粒子レベルでXPBD的なアフィン射影補正を挟むことで改善する手法
  ([Enhanced MPM with affine projection stabilizer](https://link.springer.com/article/10.1007/s00371-025-03953-2))。
  現行のAPIC (`flip_ratio=-1`) にこの補正を足すことで、大変形時の数値的破綻を抑えられる可能性がある。
- **PolyPIC への転写拡張**: 現状 PIC/APIC/FLIP のみだが、APICのアフィン速度場をさらに高次多項式に
  拡張した PolyPIC (Fu et al. 2017、関連: [PolyPIC](https://www.researchgate.net/publication/328068605_PolyPIC_The_Polymorphic-Particle-in-Cell_Method_for_Fluid-Kinetic_Coupling)) は
  渦度の保存性がAPICより高く、水しぶきや乱流的挙動の解像度が上がる。既存の `readVec4`/`writeVec4`
  マクロパターンを拡張しB行列を2次項まで持たせれば実装可能 (`mpm_common.glsl` の `B0-2` バッファ拡張)。
- **暗黙解法 (Implicit MPM)**: 現状は完全陽解法のため、剛性の高い材料 (硬岩・弾性率大) では
  `dt` が小さく制限される。Hierarchical Optimization Time-stepping (HOT) やGeoWarp
  ([NVIDIA Warpベースの微分可能・GPU実装implicit MPM, 2025](https://arxiv.org/html/2507.09435v2)) の
  ようなNewton-多重格子法をCFL律速な材料 (`mpm_geolayer`の硬岩層など) に限定適用することで、
  大きな `dt` でも安定した弾性応答が得られる。
- **摩擦接触の改善**: 現状の壁境界・コライダーBCは速度クランプによる簡易反射のみ
  (`shaders/mpm_grid_update.comp`)。MPM向けの摩擦接触解法 (2026年のarXiv 2602.02038,
  [Frictional Contact Solving for MPM](https://arxiv.org/pdf/2602.02038)) を導入すれば、
  雪崩・粒状体シムでの接触の物理精度が向上する。

### 大規模化
- **スパースグリッド化 (最重要)**: 現行はgrid_res³の密配列。SPGrid
  ([Setaluri et al. 2014、疎ページグリッド](https://dl.acm.org/doi/10.1145/2661229.2661269)) や
  最近の統一スパースフレームワーク ([arXiv 2605.28525, Unified sparse framework for
  large-scale MPM](https://arxiv.org/pdf/2605.28525)) のように、アクティブノードのみを
  ハッシュベースでインデックスする方式に切り替えれば、既存のNanoVDB依存
  (`third_party/openvdb/nanovdb`) をコライダーSDFだけでなく**シミュレーション格子そのもの**に
  適用できる。実装コストは中〜大 (P2G/G2P全パスの書き換えが必要) だが、粒子数を数百Kまで
  伸ばす場合はほぼ必須。
- **マルチGPU / 複数キュー分割**: 現状は単一compute queueで全ディスパッチを直列実行。
  空間分割 (Morton順を利用したドメイン分割) して複数キューやマルチGPU
  ([massively parallel multi-GPU MPM, SIGGRAPH 2020](https://dl.acm.org/doi/10.1145/3386569.3392442)) へ
  分散すれば粒子数を線形に伸ばせるが、Vulkan上でのマルチGPU同期コストの検証が必要。
- **Compact-Kernel MPM (CK-MPM, 2024)**: [arXiv 2412.10399](https://arxiv.org/html/2412.10399) は
  形状関数のサポート半径を縮小しメモリ帯域を削減する手法。B-spline2次 (現行 `bspline2`/`bspline2g`,
  `mpm_common.glsl:194-206`) から低次元核への切り替えで大規模シムのメモリ帯域を緩和できる。

### 機能追加
- **破壊・トポロジー変化**: 現行は連続体としてのみ扱われ破断モデルなし。応力ベースの破断条件を
  material paramsに追加し、粒子分離を許容する拡張 (既存の `MaterialParams::model` を拡張)。
- **異方性材料 (布・毛髪・繊維)**: 現行はスカラー等方性のE/ν/ρのみ。異方性Hencky弾性
  (繊維方向テンソル追加) で布・木材・毛髪材質を追加できる。
- **双方向カップリング**: `src/engine/MultiPhysicsEngine.cpp` は既に複数エンジンの組み合わせ基盤が
  あるため、MPM⇔PBF流体 や MPM⇔XPBDソフトボディ の双方向カップリング (圧力・摩擦力の相互作用)
  は比較的低コストで追加できる拡張ポイント。

---

## 3. XPBD ソフトボディ (`src/engine/SoftBodyEngine.*`)

### 現状
- 四面体メッシュの Edge距離制約 + Volume制約をXPBD (`stretchCompliance`, `volCompliance`) で解く。
- グラフ彩色 (`edgeColorBatch`/`tetColorBatch`) による並列Gauss-Seidel、`solverIterations=5`,
  `numSubsteps=15` がデフォルト。
- SDFコライダー・パーティクル間衝突 (`kParticleCollision_`) を実装済み。

### 高速化
- **Vertex Block Descent (VBD, 2024) への移行検討**: [arXiv 2403.06321](https://arxiv.org/html/2403.06321v1) は
  XPBDと同じGauss-Seidel的な並列構造を持ちながら、制約を力ベースの変分エネルギー最小化として
  直接解くため収束が速く無条件安定。特に **Augmented VBD (SIGGRAPH 2025 Real-Time Live!,
  [Chris Giles](https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25.pdf))** は
  ハード制約 (接触・関節) の扱いにAugmented Lagrangianを組み込み、XPBDのcompliance調整に依存しない
  安定した硬い拘束を実現している。現行のグラフ彩色バッチ構造はVBDにもほぼそのまま転用できるため、
  移行コストは比較的小さい。
- **多重格子XPBD (MGPBD, 2025)**: [arXiv/MGPBD](https://www.researchgate.net/publication/394032574_MGPBD_A_Multigrid_Accelerated_Global_XPBD_Solver) は
  XPBDのGauss-Seidel反復に多重格子前処理を組み合わせ、低周波成分の収束を高速化する。
  現行の `solverIterations=5` を多重格子1レベルに置き換えるだけでも大規模メッシュでの収束が改善する
  可能性がある。
- **制約バッチの動的再彩色**: 現状は初期化時に固定彩色 (`edgeColorBatch_`)。動的トポロジー変化
  (破断等) を将来追加する場合は彩色の再計算コストが問題になるため、事前に彩色フリーな並列化
  (Jacobiベースの下限緩和、あるいはVBDへの移行) を検討する価値がある。

### 精度向上
- **Augmented Lagrangian による厳密拘束**: 現行のXPBD complianceは「柔らかい制約」の近似であり、
  剛体接触や関節などの厳密拘束には不向き。Augmented VBD のAL項を導入すれば、
  接触・摩擦をより物理的に正確に解ける。
- **二次収束のスモールステップ法**: PBD/XPBDは基本的に線形収束(Gauss-Seidel)だが、
  Implicit Position-Based Fluids (IPBF, 4章参照) と同系統の「VBD類似の二次収束緩和Jacobi」を
  ソフトボディにも適用すれば同じ反復回数でより高精度な解に到達できる。
- **セルフコリジョンの連続衝突検出 (CCD)**: 現行の `kParticleCollision_` は離散的パーティクル衝突と
  推測され、高速変形時のトンネリングに弱い可能性が高い。VBDの文脈で示されている
  intersection-free self-contact (布の自己接触を貫通なく解く手法) の導入は、
  高速回転・高圧縮シーンでの頑健性向上に直結する。

### 大規模化
- **階層的LOD / マルチレゾリューションテトラメッシュ**: 現状は `.sb` ファイル単位で単一解像度の
  四面体メッシュをロードする構成 (`SoftBodyEngine::loadSBFile`)。大規模シーン (多数のソフトボディ) では
  遠方のオブジェクトを粗いテトラメッシュに切り替えるLODが有効。
  多重格子XPBD (MGPBD) の階層構造をそのままLODレンダリング/衝突判定に転用できる。
- **インスタンス間の空間ハッシュ衝突**: 現状 `SoftBodyInstance` は複数登録可能
  (`addInstance`) だが、インスタンス間衝突がO(N²)ブルートフォースだと大規模化のボトルネックになる。
  MPM/PBFで既に実装済みのハッシュグリッド (`hash_count`/`hash_sort` 系) をそのまま
  パーティクル間衝突に再利用すれば大規模ソフトボディシーンに対応できる。

### 機能追加
- **布・シェル要素との統合**: 現状クロスは別エンジン (`ClothSceneEngine`) で三角形メッシュの
  伸縮/曲げ制約として独立実装。四面体ベースのXPBDと共通の制約ソルバー基盤に統合すれば、
  布-ソフトボディの一体化シミュレーションが可能になる。
- **塑性・破壊**: Augmented VBDの応用例に挙がっている「ケーブルの曲げ塑性」のように、
  制約のrest lengthを応力履歴に応じて更新する塑性変形をXPBDに追加。
- **筋骨格・関節システム**: Augmented Lagrangianによる厳密拘束が実装できれば、
  剛体関節 (ヒンジ・ボールジョイント) をXPBDフレームワーク内に統合したキャラクター物理へ発展可能。

---

## 4. PBF 流体 (`src/engine/FluidEngine.*`)

### 現状
- Macklin & Müller (2013) のPosition Based Fluidsに準拠。CFM ε=3000, 人工圧力 k=0.001, 渦度閉じ込め
  (`vorticityEpsilon`)、粘性 (`viscosityC`) を実装。
- ハッシュグリッド近傍探索はMPMと同一パターン。境界はOBJメッシュからの境界粒子サンプリング
  (`BoundaryParticles.cpp`)。
- 粒子数は既存デモで ~110K (`fluid_pbf` TC1-3)。
- コメントに「論文忠実化(ε↓・人工圧力有効化)は発散したため一旦既定に戻している」
  (`FluidEngine.h:86-87`) との記載があり、**パラメータチューニングの限界**に既に直面している。

### 高速化
- **Dynamic Smoothing Length (2025)**: [MDPI論文](https://www.mdpi.com/2073-431X/15/1/11) は
  局所近傍数と密度変動に応じて平滑化長を動的調整するモデルを提案しており、GPU並列化も
  同論文で実証済み。疎な領域では平滑化長を広げ近傍探索コストを削減、密な領域では狭めて精度を保つ
  ため、現状の固定 `cellSize()` ベースのハッシュグリッドより効率的。
- **サブステップの適応化**: MPMと同様、CFL条件から動的にサブステップ数を決定する仕組みは未実装
  (現状固定 `numSubsteps=2`)。

### 精度向上
- **Implicit Position-Based Fluids (IPBF, SIGGRAPH Asia 2025)**: [graphics.cs.utah.edu/research/projects/ipbf](https://graphics.cs.utah.edu/research/projects/ipbf/) は
  PBFの「過圧縮 (excessive compression)」問題 — まさに `FluidEngine.h:86-87` のコメントが
  示唆する発散問題そのもの — を、VBD類似の二次収束緩和Jacobi反復による完全暗黙SPH定式化で解決する。
  現行のCFM近似 (`cfmEpsilon`) をIPBFのVBD的反復に置き換えることで、より小さいεでも安定させつつ
  精度を上げられる可能性が高く、**この論文はPBF改善の最有力候補**。
  “過圧縮を避けつつノイズも出さない” という設計目標が、まさにコード中のコメントが指摘する課題と一致する。
- **人工圧力(tensile instability対策)の再チューニング**: `scorrK=0.001` は元論文の推奨値 (0.1程度) より
  大幅に小さい。IPBFやDynamic Smoothing Lengthの手法と組み合わせれば、粒子クラスタリング
  (tensile instability) を起こさずに元論文相当の値へ近づけられる可能性がある。
- **XSPH粘性からのDFSPH的圧力投影への切り替え検討**: PBFは密度制約を反復的に解く近似解法だが、
  Divergence-Free SPH (DFSPH) のような直接圧力投影法は非圧縮性の精度が高い。大規模シーンでは
  収束の速さでPBFが有利な場面も多く、要件次第でハイブリッド化 (低反復PBF + 최終ステップDFSPH射影)
  も選択肢になる。

### 大規模化
- **疎ハッシュグリッドのページ化**: 現行のハッシュグリッドは `totalCells()` (grid_res³) 分の
  `cellCount`/`cellOffset` 配列を確保する擬似密構造 (MPMと共通)。SPGrid的なページングに変更すれば、
  水面のみに粒子が存在する大規模シーン (ダムブレイクの広い平面など) でのメモリ効率が向上する。
- **マルチGPU分散**: PBFは近傍探索が局所的なため領域分割との相性が良く、MPMと同様の
  マルチGPU分割で数百万粒子規模への拡張が見込める。

### 機能追加
- **二相流体 (油と水、気泡)**: 現状は単一密度・単一粘性。密度場を粒子ごとに持たせ相境界張力を
  追加すれば混相流に対応できる (既存の `typeFlag` を密度/粘性のマテリアルIDとして再利用可能)。
- **表面張力の明示的モデル化**: 現状は人工圧力による疑似的な表面凝集のみ。CSF (Continuum Surface
  Force) 等の表面張力モデルを追加すれば水滴・液柱の挙動が改善する。
- **PBF⇔Pyro連携 (沸騰・水蒸気)**: 高温領域のPBF粒子をPyroの density/temperature フィールドへ
  「相変化」させる (蒸発) パイプラインは、既存の `MultiPhysicsEngine` 基盤の上に比較的自然に載る
  機能追加候補。

---

## 5. Pyro (グリッドベース煙・火炎) (`src/engine/PyroEngine.*`)

### 現状
本セッションで新規実装。Morton密グリッド、Stable Fluids系の圧力投影 (Jacobi反復)、渦度閉じ込め、
燃焼反応 (fuel/temperature/flame)、複数Source、STLメッシュ動的障害物対応。可視化はC++側では行わず
`.pvox`ダンプ→Python (`tools/pyro_raymarch.py`) でレイマーチング。動的volume (ドメインの動的リサイズ)
は未実装。5ソルバーの中で最も新しく、伸びしろが最も大きい。

### 高速化
- **Jacobi→Red-Black Gauss-Seidel/多重格子への置き換え (最優先)**: 現行の `pyro_pressure_jacobi.comp`
  は素朴なJacobi反復を40〜60回行っており、収束が遅い (反復数がグリッドの直径に比例して必要)。
  Red-Black Gauss-Seidelにするだけで同反復数で収束半分程度に、幾何マルチグリッド
  (V-cycle) を導入すればO(N)で収束し反復回数を数回〜十数回まで削減できる。これはPyroの
  フレームあたりコストの支配項であり、最も投資対効果が高い改善。
- **MacCormack/BFECC移流への切り替え**: 現行の `pyro_advect.comp` はRK2逆追跡の一次精度
  semi-Lagrangian (plan通り単純さ優先で実装)。MacCormack補正 (前進+後退の2パス、誤差を1回だけ
  補正) を追加すれば数値拡散が大幅に減り、同じ視覚品質をより低いgrid_resで達成できる
  → 実質的な高速化になる。
- **圧力ダブルバッファのメモリ再利用**: 現状 `pressureIdx_[2]` は毎フレームの反復専用に確保済み
  (問題なし)だが、`curlIdx_`はvorticityEps=0でも常に確保・書き込みされる。未使用時にディスパッチを
  スキップする分岐は既にstep()内にあるが (`if (vorticityEps > 0.0f)`)、燃焼無効時
  (`burnRate=0`) の `pyro_combustion.comp` ディスパッチも同様にスキップする最適化の余地がある。

### 精度向上
- **多重格子/Red-Black反復による圧力精度向上**: 高速化と表裏一体。反復不足による非圧縮性誤差
  (divergenceの残差) が減れば、渦の保存性・煙の巻き上がりの視覚的リアリズムが向上する。
- **MacCormack/BFECC移流** (上記、精度面でも効く): 一次semi-Lagrangianは煙の細部構造を
  過度に減衰させる。二次精度化は「渦のディテールが数フレームで消える」現象を大きく改善する。
- **Neural Flow Maps的な長時間一貫性**: [Fluid Simulation on Neural Flow Maps
  (ACM TOG)](https://dl.acm.org/doi/10.1145/3618392) は、疎な多重解像度グリッド上の小型ニューラル場
  (Spatially Sparse Neural Fields) で長時間の速度場を圧縮表現し、従来のsemi-Lagrangian移流より
  低数値拡散・長時間安定な渦運動を実現する。実装コストは高いが、Pyroの「渦がすぐ滲む」問題への
  根本的解決策として長期的に検討する価値がある。
- **渦法 (Vortex Method) とのハイブリッド化**: [An Eulerian vortex method on flow maps
  (ACM TOG 2024)](https://dl.acm.org/doi/10.1145/3618392) や Vortex Particle Flow Maps (2025) は、
  格子法だけでは失われやすい渦度を粒子的に保持し格子へ再投影するハイブリッド手法。
  爆発のキノコ雲のような強い渦構造 (`pyro_explosion.cpp`) の巻き上がりをより明瞭にできる可能性がある。

### 大規模化
- **NanoVDBスパースグリッドへの本格移行**: プロジェクトは既に `third_party/openvdb/nanovdb` を
  コライダーSDF用に部分導入済み。Pyroの密度・温度・速度場そのものをNanoVDBのスパース構造に
  載せ替えれば、煙が存在しない領域の計算・メモリを完全にスキップできる。SPGridの発想
  (アダプティブスモークシム) と同系統で、grid_resを64→256程度に上げても計算コストは
  「煙が実際に占める体積」にしか比例しなくなる。既存のMorton符号化はページ内インデックスとして
  そのまま流用でき、移行の設計コストを抑えられる。
- **2のべき乗制約の緩和 (中優先度)**: 現行 `PyroConfig.grid_res` は2のべき乗必須
  (`PyroEngine.h:15-18`)。任意解像度に対応するには、Morton符号化を「タイル化+線形インデックス」の
  ハイブリッド方式に変更する必要があるが、スパースグリッド化と同時に解決するのが効率的。
- **動的volume (ドメインの動的リサイズ)**: 前回セッションでスコープ外とした機能。Houdini Pyroの
  "Dynamic Volume Resize" 相当。スパースグリッド化が完了すれば、アクティブノード集合の
  バウンディングボックスを追跡するだけで実質的に実現でき、密グリッド前提の現行実装より
  はるかに低コストで達成可能になる。

### 機能追加
- **レンダリング統合**: 現状Pythonのオフラインレイマーチングのみ。Vulkanのフラグメントシェーダーで
  3Dテクスチャ (`VK_IMAGE_TYPE_3D`) にPyroのフィールドを転送しリアルタイムレイマーチングする
  フラグメントシェーダーを追加すれば、`examples/smoke.cpp` のような可視化統合が可能になる
  (現状dumpFrame経由のCPU読み戻しはオフライン専用)。
- **熱伝導・輻射**: 現状 `tempDissipation` は単純な指数減衰のみで、隣接セル間の熱拡散
  (対流ではなく伝導) は未実装。`pyro_forces.comp` に温度のラプラシアン拡散項を追加すれば
  火災の延焼シミュレーションなどに応用できる。
- **障害物からの熱伝達・着火**: 現状オブジェクト表面 (colliderSDF) は流体力学的境界のみで
  燃焼と無関係。表面温度を追跡し引火点を超えたら周辺セルにfuelを供給する仕組みを追加すれば、
  「牛に燃え移る」ような延焼デモも作れる。
- **複数Pyroインスタンスの合成**: 現状1シーン=1 PyroEngine。密度場同士を合成する
  `MultiPhysicsEngine` 的な複数Pyroインスタンスの合成レイヤーを追加すれば、
  複数の火災源が離れた場所にある大規模シーンを扱いやすくなる (現状も複数Sourceで対応可能だが、
  グリッド全体を共有するため遠く離れた現象では無駄な計算が発生する — これもスパース化で解決)。

---

## 6. 横断的インフラ改善 (全ソルバー共通)

1. **Bindlessバッファ上限 (`MAX_BINDLESS_BUFFERS=16`) の緩和**:
   `VkPhysicalDeviceDescriptorIndexingFeatures` を使い `descriptorCount` を実行時にクエリして
   上限を動的に引き上げる。特にPyroは既に14/16を消費しており、機能追加の直接的な障害になっている。
2. **Specialization Constants の活用**: grid_res・material数・sourceCount上限などコンパイル時に
   確定可能な値をpush constantからspecialization constantへ移行し、シェーダーコンパイラによる
   最適化 (ループ展開・分岐削除) を促進する。
3. **非同期コンピュート/複数キューの活用**: 現行はcompute queueとgraphics queueの2本のみ。
   独立したソルバー同士 (例: MultiPhysicsEngineでMPMとPBFを同時に走らせる場合) を別々の
   compute queueに載せてGPU内並列度を上げる。
4. **プロファイリング基盤の追加**: 現状 `VULKAN_VALIDATION` (Debugビルド時) はあるが、
   `VK_KHR_pipeline_executable_properties` や GPUタイムスタンプクエリによる
   パス単位のプロファイル計測が未整備。ボトルネック特定 (特にPyroのJacobi反復回数チューニング) に
   有効。
5. **スパースグリッド共通基盤の抽出**: MPM・Pyro双方が「Morton密グリッド」を独立に実装している。
   NanoVDBベースのスパースグリッドへ移行する際、両ソルバーで共有可能な
   `SparseGridBuffer` 抽象を `src/core/` に切り出せば実装コストを削減できる。
6. **微分可能化 / 学習ベース補助**: GeoWarp (NVIDIA Warp) やNeural Flow Mapsに見られるように、
   2024〜2026年の潮流は「古典ソルバー + 自動微分/ニューラル補助」のハイブリッドが主流になりつつある。
   本プロジェクトはVulkan computeという性質上、自動微分の直接導入は困難だが、
   将来的にオフラインでニューラル補正場を学習しシェーダーで推論するくらいの軽量な統合は
   検討の余地がある (優先度は低い)。

---

## 7. 優先度つきロードマップ案

### 短期 (低コスト・高効果)
- Pyro: 圧力Jacobi → Red-Black Gauss-Seidel化 (実装コスト小、収束2倍以上)
- Pyro: MacCormack/BFECC移流 (数値拡散の大幅削減)
- PBF: Implicit Position-Based Fluids (IPBF) の反復則を部分適用し、既知の過圧縮問題
  (`FluidEngine.h:86-87` のコメント) を解消
- XPBD: Vertex Block Descent (VBD) への切り替え検証 (既存のグラフ彩色バッチ構造を流用可能)
- 共通: Bindlessバッファ上限の動的化 (Pyroの拡張余地を確保)

### 中期
- MPM/Pyro: NanoVDBベースのスパースグリッド化 (大規模化の本命、実装コスト中〜大)
- MPM: G2P2Gカーネル融合、APS (Affine Projection Stabilizer) 導入
- XPBD: 多重格子XPBD (MGPBD) 導入
- Pyro: リアルタイムレイマーチングのVulkan統合、動的volume (スパース化後に低コスト化)

### 長期
- MPM: 準ニュートン多重格子による部分Implicit化 (硬い材料のみ)
- Pyro: Neural Flow Maps / Vortex Method ハイブリッド
- 全体: マルチGPU分散、双方向カップリング (MPM⇔PBF⇔XPBD⇔Pyro) の統合基盤

---

## 8. 参考文献 (2023-2026 中心)

**MPM**
- Zhu et al., "A massively parallel and scalable multi-GPU material point method", SIGGRAPH 2020.
  https://dl.acm.org/doi/10.1145/3386569.3392442
- "Unified sparse framework for large-scale material point method simulations", arXiv 2605.28525.
  https://arxiv.org/pdf/2605.28525
- "Enhanced material point method with affine projection stabilizer", The Visual Computer, 2025.
  https://link.springer.com/article/10.1007/s00371-025-03953-2
- "GeoWarp: differentiable GPU-accelerated implicit MPM (NVIDIA Warp)", 2025.
  https://arxiv.org/html/2507.09435v2
- "Substepping the Material Point Method", arXiv 2508.11722, 2025.
  https://arxiv.org/pdf/2508.11722
- "Frictional Contact Solving for Material Point Method", arXiv 2602.02038, 2026.
  https://arxiv.org/pdf/2602.02038
- "CK-MPM: A Compact-Kernel Material Point Method", arXiv 2412.10399, 2024.
  https://arxiv.org/html/2412.10399
- Fu et al., "A polynomial particle-in-cell method (PolyPIC)", 2017.
  https://www.researchgate.net/publication/328068605

**XPBD / ソフトボディ**
- Chen et al., "Vertex Block Descent", arXiv 2403.06321, 2024. https://arxiv.org/html/2403.06321v1
- Giles, "Augmented Vertex Block Descent", SIGGRAPH 2025 Real-Time Live!.
  https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25.pdf
- "MGPBD: A Multigrid Accelerated Global XPBD Solver", 2025.
  https://www.researchgate.net/publication/394032574

**PBF / SPH**
- Macklin & Müller, "Position Based Fluids", ACM TOG 2013 (原論文).
  https://dl.acm.org/doi/10.1145/2461912.2461984
- "Implicit Position-Based Fluids (IPBF)", SIGGRAPH Asia 2025.
  https://graphics.cs.utah.edu/research/projects/ipbf/
- "A Position-Based Fluid Method with Dynamic Smoothing Length", MDPI, 2025.
  https://www.mdpi.com/2073-431X/15/1/11

**Pyro / 煙・火炎**
- "Fluid Simulation on Neural Flow Maps", ACM TOG 2024. https://dl.acm.org/doi/10.1145/3618392
- "Physics-Informed Learning of Characteristic Trajectories for Smoke Reconstruction",
  SIGGRAPH 2024. https://dl.acm.org/doi/10.1145/3641519.3657483
- Setaluri et al., "SPGrid: a sparse paged grid structure applied to adaptive smoke simulation",
  ACM TOG 2014. https://dl.acm.org/doi/10.1145/2661229.2661269
- "Fluid Simulation on Vortex Particle Flow Maps", ACM TOG 2025.
