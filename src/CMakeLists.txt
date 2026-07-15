# ── nebula: 共通ライブラリ ───────────────────────────────────────────────────
# VulkanContext / AttributeBuffer / ComputePipeline / GraphicsPipeline や各種
# Engine (Fluid, Cloth, MPM, Pyro, ...) をまとめた静的ライブラリ。
# 従来は例ごとに同じ .cpp を再コンパイルしていたため、ここに集約して一度だけビルドする。
add_library(nebula STATIC
    core/VulkanContext.cpp
    core/AttributeBuffer.cpp
    core/ComputePipeline.cpp
    graphics/GraphicsPipeline.cpp
    graphics/ClothRenderer.cpp
    engine/ClothMesh.cpp
    engine/SimulationEngine.cpp
    engine/ClothSceneEngine.cpp
    engine/FluidEngine.cpp
    engine/BoundaryParticles.cpp
    engine/SoftBodyEngine.cpp
    engine/MultiPhysicsEngine.cpp
    engine/MPMEngine.cpp
    engine/PyroEngine.cpp
    App.cpp
)

if(APPLE)
    target_sources(nebula PRIVATE VkSafeAlloc.mm)
endif()

target_include_directories(nebula PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/graphics
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_SOURCE_DIR}/third_party/VulkanMemoryAllocator/include
    ${CMAKE_SOURCE_DIR}/third_party/argparse/include
    ${NANOVDB_INCLUDE_DIR}  # NanoVDB: #include <nanovdb/NanoVDB.h>
    ${SHADERC_INCLUDE_DIR}  # shaderc: #include <shaderc/shaderc.hpp> (issue #30 Force動的コンパイル用)
)

target_link_libraries(nebula PUBLIC
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui
    GPUOpen::VulkanMemoryAllocator
    tinyobjloader
    ${SHADERC_LIB}
)

# SHADER_DIR / ASSET_DIR は example・test からも参照されるため PUBLIC にして伝播させる
# SHADER_SRC_DIR: Force動的シェーダー生成 (issue #30) が #include "common.glsl" 等を
# 実行時に解決する際のソースディレクトリ (glslc向けの SHADER_DIR = コンパイル済み.spv出力先とは別)
target_compile_definitions(nebula PUBLIC
    SHADER_DIR="${SHADER_OUT_DIR}"
    SHADER_SRC_DIR="${SHADER_SRC_DIR}"
    ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
)

# VULKAN_VALIDATION は VulkanContext.cpp 内でのみ使用
target_compile_definitions(nebula PRIVATE
    $<$<CONFIG:Debug>:VULKAN_VALIDATION>
)

if(APPLE)
    target_compile_definitions(nebula PUBLIC VK_USE_PLATFORM_METAL_EXT)
    target_link_libraries(nebula PUBLIC "-framework Foundation")
endif()

# Apple libc++ (SDK 14) は std::atomic_ref を未実装のため、
# NanoVDB Util.h が C++20 ブランチに入らないよう MPMEngine.cpp のみ C++17 でコンパイル。
if(APPLE)
    set_source_files_properties(engine/MPMEngine.cpp
        PROPERTIES COMPILE_OPTIONS "-std=c++17")
endif()

add_dependencies(nebula shaders)
