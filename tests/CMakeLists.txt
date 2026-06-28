cmake_minimum_required(VERSION 3.22)

# ── テスト実行ファイル ────────────────────────────────────────────────────────
if(APPLE)
    set(PLATFORM_WRAPPERS
        helpers/VkWrapper.mm                      # ObjC++ NSException catcher (vkCreateInstance)
        ${CMAKE_SOURCE_DIR}/src/VkSafeAlloc.mm    # ObjC++ NSException catcher (vmaCreateBuffer)
    )
else()
    set(PLATFORM_WRAPPERS "")
endif()

add_executable(vulkan_tests
    main.cpp
    helpers/HeadlessCtx.cpp
    ${PLATFORM_WRAPPERS}
    helpers/PhysicsHarness.cpp
    helpers/PBFHarness.cpp
    test_string2d.cpp
    test_string3d.cpp
    test_fluid.cpp
    test_pbf_fluid.cpp
    test_pbf.cpp
    # 共通シミュレーションコード (メインアプリと共有)
    ${CMAKE_SOURCE_DIR}/src/core/AttributeBuffer.cpp
    ${CMAKE_SOURCE_DIR}/src/core/ComputePipeline.cpp
)

add_dependencies(vulkan_tests shaders)

target_include_directories(vulkan_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/engine
    ${CMAKE_SOURCE_DIR}/src/graphics
    ${CMAKE_SOURCE_DIR}/third_party/VulkanMemoryAllocator/include
    ${doctest_SOURCE_DIR}               # doctest/doctest.h はここの下にある
)

target_compile_definitions(vulkan_tests PRIVATE
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)
if(APPLE)
    target_compile_definitions(vulkan_tests PRIVATE VK_USE_PLATFORM_METAL_EXT)
endif()

target_link_libraries(vulkan_tests PRIVATE
    Vulkan::Vulkan
    GPUOpen::VulkanMemoryAllocator
    glm::glm
    doctest::doctest
)

if(APPLE)
    target_link_libraries(vulkan_tests PRIVATE "-framework Foundation")
endif()

if(APPLE)
    set_target_properties(vulkan_tests PROPERTIES
        BUILD_RPATH   "${VULKAN_SDK}/lib"
        INSTALL_RPATH "${VULKAN_SDK}/lib"
    )
endif()

# ── CTest 登録 ───────────────────────────────────────────────────────────────
add_test(
    NAME    physics_tests
    COMMAND vulkan_tests --reporters=console --no-intro=true
)
# GPU テストはやや時間がかかるのでタイムアウトを長めに設定
set_tests_properties(physics_tests PROPERTIES TIMEOUT 300)
