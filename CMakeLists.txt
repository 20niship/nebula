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

# ── Homebrew packages ──────────────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
find_package(glfw3 REQUIRED)
find_package(glm   REQUIRED)

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
    ${SHADER_SRC_DIR}/predict.comp
    ${SHADER_SRC_DIR}/predict_sdf.comp
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
)

set(SHADER_SPV)
foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(FNAME ${SHADER} NAME)
    set(SPV "${SHADER_OUT_DIR}/${FNAME}.spv")
    add_custom_command(
        OUTPUT  ${SPV}
        COMMAND ${GLSLC} --target-env=vulkan1.2
                         -I ${SHADER_SRC_DIR}
                         -o ${SPV} ${SHADER}
        DEPENDS ${SHADER} ${SHADER_SRC_DIR}/common.glsl
        COMMENT "Compiling shader ${FNAME}"
    )
    list(APPEND SHADER_SPV ${SPV})
endforeach()

add_custom_target(shaders ALL DEPENDS ${SHADER_SPV})

# ── 共通設定ヘルパー ─────────────────────────────────────────────────────
set(SIM_INCLUDE_DIRS
    src
    src/core
    src/engine
    src/graphics
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    third_party/VulkanMemoryAllocator/include
    third_party/argparse/include
)
set(SIM_LINK_LIBS
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui
    GPUOpen::VulkanMemoryAllocator
)

function(add_sim_example TARGET)
    # ARGN = source files
    if(APPLE)
        add_executable(${TARGET} ${ARGN} src/VkSafeAlloc.mm)
    else()
        add_executable(${TARGET} ${ARGN})
    endif()
    add_dependencies(${TARGET} shaders)
    target_include_directories(${TARGET} PRIVATE ${SIM_INCLUDE_DIRS})
    target_link_libraries(${TARGET} PRIVATE ${SIM_LINK_LIBS})
    target_compile_definitions(${TARGET} PRIVATE
        SHADER_DIR="${SHADER_OUT_DIR}"
        ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
    )
    if(APPLE)
        set_target_properties(${TARGET} PROPERTIES
            BUILD_RPATH   "${VULKAN_SDK}/lib"
            INSTALL_RPATH "${VULKAN_SDK}/lib"
        )
        target_compile_definitions(${TARGET} PRIVATE VK_USE_PLATFORM_METAL_EXT)
        target_link_libraries(${TARGET} PRIVATE "-framework Foundation")
    endif()
    target_compile_definitions(${TARGET} PRIVATE
        $<$<CONFIG:Debug>:VULKAN_VALIDATION>
    )
endfunction()

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

enable_testing()
add_subdirectory(tests)

# ── cloth_3d: 3D 布シミュレーション (Phase 3) ─────────────────────────────
add_sim_example(cloth_3d
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/ClothMesh.cpp
    src/graphics/ClothRenderer.cpp
    src/engine/SimulationEngine.cpp
    src/App.cpp
    examples/cloth_3d.cpp
)

# ── string_2d: 2D 糸シミュレーション (自己衝突検証) ──────────────────────
add_sim_example(string_2d
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    examples/string_2d.cpp
)

# ── fluid_pbf: Phase 4 PBF 流体シミュレーション ───────────────────────────
add_sim_example(fluid_pbf
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/FluidEngine.cpp
    src/engine/BoundaryParticles.cpp
    src/App.cpp
    examples/fluid_pbf.cpp
)
target_link_libraries(fluid_pbf PRIVATE tinyobjloader)

# ── fluid_absorb: 楕円水たまり + 移動円柱吸収ポート ────────────────────────────
add_sim_example(fluid_absorb
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/FluidEngine.cpp
    src/engine/BoundaryParticles.cpp
    src/App.cpp
    examples/fluid_absorb.cpp
)
target_link_libraries(fluid_absorb PRIVATE tinyobjloader)

# ── fluid_sphere: 球体容器内 PBF 流体 (UI なし自動スポーン) ──────────────────
add_sim_example(fluid_sphere
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/FluidEngine.cpp
    src/engine/BoundaryParticles.cpp
    src/App.cpp
    examples/fluid_sphere.cpp
)
target_link_libraries(fluid_sphere PRIVATE tinyobjloader)

# ── smoke: 煙パーティクルシミュレーション ─────────────────────────────────────
add_sim_example(smoke
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/FluidEngine.cpp
    src/engine/BoundaryParticles.cpp
    src/App.cpp
    examples/smoke.cpp
)
target_link_libraries(smoke PRIVATE tinyobjloader)

# ── screw_fluid: TC8 回転スクリュー in PBF 流体 ──────────────────────────────
add_sim_example(screw_fluid
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/FluidEngine.cpp
    src/engine/BoundaryParticles.cpp
    src/App.cpp
    examples/screw_fluid.cpp
)
target_link_libraries(screw_fluid PRIVATE tinyobjloader)

# ── cloth_scene: TC5 布2枚 / TC7 4隅ねじれ ──────────────────────────────────
add_sim_example(cloth_scene
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/graphics/ClothRenderer.cpp
    src/engine/ClothMesh.cpp
    src/engine/ClothSceneEngine.cpp
    src/App.cpp
    examples/cloth_scene.cpp
)

# ── multi_physics: Phase 5 布×流体 双方向連成 ────────────────────────────────
add_sim_example(multi_physics
    src/core/VulkanContext.cpp
    src/core/AttributeBuffer.cpp
    src/core/ComputePipeline.cpp
    src/graphics/GraphicsPipeline.cpp
    src/engine/ClothMesh.cpp
    src/graphics/ClothRenderer.cpp
    src/engine/MultiPhysicsEngine.cpp
    src/App.cpp
    examples/multi_physics.cpp
)
