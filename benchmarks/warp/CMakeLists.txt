# Warp 比較ベンチ用 headless 実行ファイル (NEBULA_BUILD_BENCH_WARP=ON のときのみ追加される)
add_executable(nebula_bench
    nebula_bench.cpp
    ${CMAKE_SOURCE_DIR}/tests/helpers/HeadlessCtx.cpp
    ${PYRO_PLATFORM_WRAPPERS}
)
add_dependencies(nebula_bench shaders)
target_include_directories(nebula_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(nebula_bench PRIVATE nebula)
if(APPLE)
    set_target_properties(nebula_bench PROPERTIES
        BUILD_RPATH   "${VULKAN_SDK}/lib"
        INSTALL_RPATH "${VULKAN_SDK}/lib"
    )
endif()
