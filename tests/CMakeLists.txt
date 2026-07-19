cmake_minimum_required(VERSION 3.22)

# ── テスト実行ファイル ────────────────────────────────────────────────────────
if(APPLE)
    set(PLATFORM_WRAPPERS
        helpers/VkWrapper.mm                      # ObjC++ NSException catcher (vkCreateInstance)
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
    helpers/MPMHarness.cpp
    test_string2d.cpp
    test_string3d.cpp
    test_fluid.cpp
    test_pbf_fluid.cpp
    test_pbf.cpp
    test_mpm.cpp
    test_pyro.cpp
    test_attribute_buffer.cpp
    test_domain.cpp
)

add_dependencies(vulkan_tests shaders)

target_include_directories(vulkan_tests PRIVATE
    ${doctest_SOURCE_DIR}               # doctest/doctest.h はここの下にある
)

target_link_libraries(vulkan_tests PRIVATE
    nebula
    doctest::doctest
)

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
