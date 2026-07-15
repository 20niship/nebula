cmake_minimum_required(VERSION 3.22)
project(vulkan_sim CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Vulkan SDK (MoltenVK) ──────────────────────────────────────────────────
# Priority: -DVULKAN_SDK=... > $VULKAN_SDK env > local LunarG install
if(NOT DEFINED VULKAN_SDK)
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK "$ENV{VULKAN_SDK}")
    else()
        set(VULKAN_SDK "$ENV{HOME}/VulkanSDK/1.3.280.1/macOS")
    endif()
endif()
set(ENV{VULKAN_SDK} "${VULKAN_SDK}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_PREFIX_PATH "${VULKAN_SDK}")

find_package(Vulkan REQUIRED)

find_program(GLSLC glslc HINTS "${VULKAN_SDK}/bin" REQUIRED)

# ── shaderc (実行時 GLSL→SPIR-V 動的コンパイル; issue #30 Force システム用) ──
# Vulkan SDK に同梱の shaderc_combined を使用。新規パッケージ導入は不要。
find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.hpp
    HINTS "${VULKAN_SDK}/include"
    REQUIRED
)
find_library(SHADERC_LIB
    NAMES shaderc_combined
    HINTS "${VULKAN_SDK}/lib"
    REQUIRED
)

# ── Homebrew packages ──────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(glfw3   REQUIRED)
find_package(glm     REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ── NanoVDB (header-only, OpenVDB submodule の nanovdb/ のみ) ────────────
set(NANOVDB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/openvdb/nanovdb)
list(APPEND CMAKE_PREFIX_PATH ${NANOVDB_INCLUDE_DIR})

# Mac (MoltenVK): shaderFloat64=false → PNanoVDB.h の double を float に自動パッチ
if(APPLE)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/nanovdb_glsl)
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/cmake/patch_pnanovdb.py
                ${NANOVDB_INCLUDE_DIR}/nanovdb/PNanoVDB.h
                ${CMAKE_BINARY_DIR}/nanovdb_glsl/PNanoVDB.h
        RESULT_VARIABLE _patch_result
    )
    if(NOT _patch_result EQUAL 0)
        message(WARNING "patch_pnanovdb.py failed: NanoVDB GLSL shaders may not compile on MoltenVK")
    endif()
    set(PNANOVDB_GLSL_INCLUDE ${CMAKE_BINARY_DIR}/nanovdb_glsl)
else()
    set(PNANOVDB_GLSL_INCLUDE ${NANOVDB_INCLUDE_DIR}/nanovdb)
endif()

# ── VulkanMemoryAllocator (header-only) ───────────────────────────────────
add_subdirectory(third_party/VulkanMemoryAllocator)

# ── ImGui ─────────────────────────────────────────────────────────────────
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

# ── Shaders ───────────────────────────────────────────────────────────────
set(SHADER_SRC_DIR ${CMAKE_SOURCE_DIR}/shaders)
set(SHADER_OUT_DIR ${CMAKE_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

set(SHADER_SOURCES
    # predict.comp / predict_sdf.comp / mpm_grid_update.comp は FORCE_AUTOGEN
    # マーカーを含み、Engine 側で ForceShaderCompiler により実行時に shaderc で
    # コンパイルされる (issue #30)。静的 glslc ビルド対象には含めない。
    ${SHADER_SRC_DIR}/sdf_collision.comp
    ${SHADER_SRC_DIR}/hash_count.comp
    ${SHADER_SRC_DIR}/hash_scan_local.comp
    ${SHADER_SRC_DIR}/hash_scan_global.comp
    ${SHADER_SRC_DIR}/hash_add_base.comp
    ${SHADER_SRC_DIR}/hash_sort.comp
    ${SHADER_SRC_DIR}/solve_density.comp
    ${SHADER_SRC_DIR}/solve_stretch.comp
    ${SHADER_SRC_DIR}/update_velocity.comp
    ${SHADER_SRC_DIR}/pbf_density.comp
    ${SHADER_SRC_DIR}/pbf_delta_p.comp
    ${SHADER_SRC_DIR}/pbf_viscosity.comp
    ${SHADER_SRC_DIR}/pbf_vorticity_omega.comp
    ${SHADER_SRC_DIR}/pbf_vorticity_force.comp
    ${SHADER_SRC_DIR}/particle.vert
    ${SHADER_SRC_DIR}/particle.frag
    ${SHADER_SRC_DIR}/cloth.vert
    ${SHADER_SRC_DIR}/cloth.frag
    ${SHADER_SRC_DIR}/string2d.vert
    ${SHADER_SRC_DIR}/string2d.frag
    ${SHADER_SRC_DIR}/fluid_particle.vert
    ${SHADER_SRC_DIR}/fluid.frag
    ${SHADER_SRC_DIR}/coupling_cloth.comp
    ${SHADER_SRC_DIR}/zero_cells.comp
    ${SHADER_SRC_DIR}/zero_lambdas.comp
    ${SHADER_SRC_DIR}/move_pins.comp
    ${SHADER_SRC_DIR}/fluid_absorb.comp
    # ── XPBD Soft Body シェーダー ────────────────────────────────────────
    ${SHADER_SRC_DIR}/sb_solve_volume.comp
    ${SHADER_SRC_DIR}/sb_zero_vol_lambdas.comp
    ${SHADER_SRC_DIR}/sb_particle_collision.comp
    ${SHADER_SRC_DIR}/sb_wire.vert
    ${SHADER_SRC_DIR}/sb_wire.frag
    # ── MPM シェーダー ────────────────────────────────────────────────────
    ${SHADER_SRC_DIR}/mpm_hash_count.comp
    ${SHADER_SRC_DIR}/mpm_hash_sort.comp
    ${SHADER_SRC_DIR}/mpm_zero_grid.comp
    ${SHADER_SRC_DIR}/mpm_p2g.comp
    ${SHADER_SRC_DIR}/mpm_grid_update.comp
    ${SHADER_SRC_DIR}/mpm_g2p.comp
    ${SHADER_SRC_DIR}/mpm_nanovdb_bc.comp
    # ── Pyro (グリッドベース煙・火炎) シェーダー ─────────────────────────────
    ${SHADER_SRC_DIR}/pyro_emit.comp
    ${SHADER_SRC_DIR}/pyro_combustion.comp
    ${SHADER_SRC_DIR}/pyro_forces.comp
    ${SHADER_SRC_DIR}/pyro_obstacle_bc.comp
    ${SHADER_SRC_DIR}/pyro_advect.comp
    ${SHADER_SRC_DIR}/pyro_curl.comp
    ${SHADER_SRC_DIR}/pyro_vorticity_force.comp
    ${SHADER_SRC_DIR}/pyro_divergence.comp
    ${SHADER_SRC_DIR}/pyro_pressure_jacobi.comp
    ${SHADER_SRC_DIR}/pyro_project.comp
)

set(SHADER_SPV)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(FNAME ${SHADER} NAME)
    set(SPV "${SHADER_OUT_DIR}/${FNAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSLC} --target-env=vulkan1.2
                         -I ${SHADER_SRC_DIR}
                         -I ${PNANOVDB_GLSL_INCLUDE}
                         -o ${SPV} ${SHADER}
        DEPENDS ${SHADER} ${SHADER_SRC_DIR}/common.glsl
                ${SHADER_SRC_DIR}/mpm_common.glsl ${SHADER_SRC_DIR}/collider_common.glsl
                ${SHADER_SRC_DIR}/pyro_common.glsl
                ${PNANOVDB_GLSL_INCLUDE}/PNanoVDB.h
        COMMENT "Compiling shader ${FNAME}"
    )
    list(APPEND SHADER_SPV ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_SPV})

# ── doctest (ヘッダオンリー、FetchContent 経由) ───────────────────────────
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    TRUE
)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(doctest)

# ── tinyobjloader (OBJ 境界粒子ローダー) ─────────────────────────────────
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(tinyobjloader)

# assets ディレクトリ
set(ASSET_DIR ${CMAKE_SOURCE_DIR}/assets)

add_subdirectory(src)

# ── 共通設定ヘルパー ─────────────────────────────────────────────────────
function(add_sim_example TARGET)
    # ARGN = source files
    add_executable(${TARGET} ${ARGN})
    add_dependencies(${TARGET} shaders)
    target_link_libraries(${TARGET} PRIVATE nebula)
    if(APPLE)
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH   "${VULKAN_SDK}/lib"
            INSTALL_RPATH "${VULKAN_SDK}/lib"
        )
    endif()
endfunction()

enable_testing()
add_subdirectory(tests)

# ── cloth_3d: 3D 布シミュレーション (Phase 3) ─────────────────────────────
add_sim_example(cloth_3d examples/cloth_3d.cpp)

# ── string_2d: 2D 糸シミュレーション (自己衝突検証) ──────────────────────
add_sim_example(string_2d examples/string_2d.cpp)

# ── fluid_pbf: Phase 4 PBF 流体シミュレーション ───────────────────────────
add_sim_example(fluid_pbf examples/fluid_pbf.cpp)

# ── fluid_absorb: 楕円水たまり + 移動円柱吸収ポート ────────────────────────────
add_sim_example(fluid_absorb examples/fluid_absorb.cpp)

# ── fluid_sphere: 球体容器内 PBF 流体 (UI なし自動スポーン) ──────────────────
add_sim_example(fluid_sphere examples/fluid_sphere.cpp)

# ── smoke: 煙パーティクルシミュレーション ─────────────────────────────────────
add_sim_example(smoke examples/smoke.cpp)

# ── screw_fluid: TC8 回転スクリュー in PBF 流体 ──────────────────────────────
add_sim_example(screw_fluid examples/screw_fluid.cpp)

# ── cloth_scene: TC5 布2枚 / TC7 4隅ねじれ ──────────────────────────────────
add_sim_example(cloth_scene examples/cloth_scene.cpp)

# ── xpbd_softbody: XPBD 四面体ソフトボディシミュレーション ──────────────────
add_sim_example(xpbd_softbody examples/xpbd_softbody.cpp)

# ── multi_physics: Phase 5 布×流体 双方向連成 ────────────────────────────────
add_sim_example(multi_physics examples/multi_physics.cpp)

# ── mpm_elastic: GPU MPM 弾性/塑性シミュレーション ─────────────────────────
add_sim_example(mpm_elastic examples/mpm_elastic.cpp)

# ── mpm_multimaterial: 弾性体 + 砂 混在シミュレーション ─────────────────
add_sim_example(mpm_multimaterial examples/mpm_multimaterial.cpp)

# ── mpm_avalanche: 山岳地形 + Drucker-Prager 雪崩 (~80K 粒子) ───────────
add_sim_example(mpm_avalanche examples/mpm_avalanche.cpp)

# ── mpm_geolayer: 地層崩壊 (複数マテリアル) ──────────────────────────────────
add_sim_example(mpm_geolayer examples/mpm_geolayer.cpp)

# ── mpm_snow_impact: 移動箱 SDF による雪衝突 ─────────────────────────────────
add_sim_example(mpm_snow_impact examples/mpm_snow_impact.cpp)

# ── Pyro (グリッドベース煙・火炎) ヘッドレス実行例 ───────────────────────────
# ウィンドウ/ImGui を使わないため add_sim_example ではなく tests/vulkan_tests と
# 同じ最小構成 (HeadlessCtx) を用いる。ボクセルを .pvox にダンプするのみで、
# 可視化は tools/pyro_raymarch.py (Python) 側で行う。
if(APPLE)
    set(PYRO_PLATFORM_WRAPPERS tests/helpers/VkWrapper.mm)
else()
    set(PYRO_PLATFORM_WRAPPERS "")
endif()

function(add_pyro_example TARGET SOURCE_FILE)
    add_executable(${TARGET}
        ${SOURCE_FILE}
        tests/helpers/HeadlessCtx.cpp
        ${PYRO_PLATFORM_WRAPPERS}
    )
    add_dependencies(${TARGET} shaders)
    target_include_directories(${TARGET} PRIVATE tests)
    target_link_libraries(${TARGET} PRIVATE nebula)
    if(APPLE)
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH   "${VULKAN_SDK}/lib"
            INSTALL_RPATH "${VULKAN_SDK}/lib"
        )
    endif()
endfunction()

add_pyro_example(pyro_basic       examples/pyro_basic.cpp)
add_pyro_example(pyro_cow_blast   examples/pyro_cow_blast.cpp)
add_pyro_example(pyro_explosion   examples/pyro_explosion.cpp)

# ── capture_all: capture_sim.py が使う実行ファイルを一括ビルドするための集約ターゲット ──
add_custom_target(capture_all DEPENDS
    fluid_pbf cloth_3d cloth_scene smoke fluid_absorb xpbd_softbody
    screw_fluid mpm_elastic mpm_avalanche mpm_multimaterial
    mpm_snow_impact mpm_geolayer pyro_cow_blast pyro_explosion
)
