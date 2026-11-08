# 高品質デモ (NEBULA_BUILD_DEMOS=ON のときのみ追加)。シミュレーションは headless 実行し、結果は render/*.py (Blender bpy) でレンダリングする。
function(nebula_demo name)
    add_executable(${name} ${name}.cpp ${CMAKE_SOURCE_DIR}/tests/helpers/HeadlessCtx.cpp ${PYRO_PLATFORM_WRAPPERS})
    add_dependencies(${name} shaders)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${name} PRIVATE nebula)
    if(APPLE)
        set_target_properties(${name} PROPERTIES BUILD_RPATH "${VULKAN_SDK}/lib" INSTALL_RPATH "${VULKAN_SDK}/lib")
    endif()
endfunction()

nebula_demo(demo_city_wind)
nebula_demo(demo_avalanche_mesh)
nebula_demo(demo_seashore)
