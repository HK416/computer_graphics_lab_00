cmake_minimum_required(VERSION 3.25)

# vcpkg 를 고정 태그로 클론/부트스트랩해 툴체인으로 사용한다. project() 호출보다 앞서야 한다.
set(VCPKG_TAG "2026.07.29" CACHE STRING "고정할 vcpkg 릴리스 태그")
set(VCPKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg")

if(WIN32)
    set(vcpkg_exe "${VCPKG_DIR}/vcpkg.exe")
    set(vcpkg_bootstrap "${VCPKG_DIR}/bootstrap-vcpkg.bat")
else()
    set(vcpkg_exe "${VCPKG_DIR}/vcpkg")
    set(vcpkg_bootstrap "${VCPKG_DIR}/bootstrap-vcpkg.sh")
endif()

if(NOT EXISTS "${VCPKG_DIR}/.git")
    find_package(Git REQUIRED)
    message(STATUS "vcpkg ${VCPKG_TAG} 클론: ${VCPKG_DIR}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --branch "${VCPKG_TAG}"
                https://github.com/microsoft/vcpkg.git "${VCPKG_DIR}"
        RESULT_VARIABLE vcpkg_clone_result)
    if(NOT vcpkg_clone_result EQUAL 0)
        message(FATAL_ERROR "vcpkg 클론 실패 (${vcpkg_clone_result})")
    endif()
endif()

if(NOT EXISTS "${vcpkg_exe}")
    message(STATUS "vcpkg 부트스트랩")
    execute_process(
        COMMAND "${vcpkg_bootstrap}" -disableMetrics
        WORKING_DIRECTORY "${VCPKG_DIR}"
        RESULT_VARIABLE vcpkg_bootstrap_result)
    if(NOT vcpkg_bootstrap_result EQUAL 0)
        message(FATAL_ERROR "vcpkg 부트스트랩 실패 (${vcpkg_bootstrap_result})")
    endif()
endif()

set(CMAKE_TOOLCHAIN_FILE "${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")

# clangd / clang-tidy 같은 외부 도구가 시스템 헤더를 찾으려면 compile_commands.json 에 -isysroot 가 있어야 한다.
# CMake 는 기본 SDK 일 때 이 플래그를 생략하므로 직접 지정한다.
if(NOT CMAKE_OSX_SYSROOT AND EXISTS /usr/bin/xcrun)
    execute_process(
        COMMAND /usr/bin/xcrun --sdk macosx --show-sdk-path
        OUTPUT_VARIABLE macos_sdk_path
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(macos_sdk_path)
        set(CMAKE_OSX_SYSROOT "${macos_sdk_path}")
    endif()
endif()

project(cg_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)

# ---- 셰이더: GLSL 을 SPIR-V 로 컴파일한다 ----
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc 를 찾지 못했습니다. Vulkan SDK 설치를 확인하십시오.")
endif()

set(shader_output_dir "${CMAKE_BINARY_DIR}/shaders")
set(shader_sources
    shaders/mesh.vert
    shaders/mesh.frag
    shaders/mesh_oit.frag
    shaders/fullscreen.vert
    shaders/oit_composite.frag
    shaders/tonemap.frag
    shaders/bloom_downsample.comp
    shaders/bloom_upsample.comp
    shaders/exposure_histogram.comp
    shaders/exposure_average.comp
    shaders/upscale.frag
    shaders/sky.frag
    shaders/taau.comp
    shaders/mesh.task
    shaders/mesh.mesh
    shaders/depth_only.vert
    shaders/shadow_cutoff.frag
    shaders/sky.comp
    shaders/irradiance.comp
    shaders/prefilter.comp
    shaders/brdf_lut.comp
    shaders/ssao.comp
    shaders/ssao_blur.comp
    shaders/reflect.comp
    shaders/cull_meshlets.comp
    shaders/skin.comp
    shaders/skin_bounds.comp
    shaders/hzb_reduce.comp
    shaders/pathtrace.rgen
    shaders/pathtrace.rmiss
    shaders/pathtrace_shadow.rmiss
    shaders/pathtrace.rchit
    shaders/pathtrace.rahit)

set(shader_binaries)
foreach(shader IN LISTS shader_sources)
    get_filename_component(shader_name "${shader}" NAME)
    set(shader_binary "${shader_output_dir}/${shader_name}.spv")
    add_custom_command(
        OUTPUT "${shader_binary}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_output_dir}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}"
                --target-env=vulkan1.3
                -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                -MD -MF "${shader_binary}.d"
                $<IF:$<CONFIG:Debug>,-g,-O>
                -o "${shader_binary}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
        DEPFILE "${shader_binary}.d"
        COMMENT "셰이더 컴파일: ${shader}"
        VERBATIM)
    list(APPEND shader_binaries "${shader_binary}")
endforeach()

# 광선 질의 그림자는 SPIR-V 능력을 요구해 런타임 분기가 안 된다. 같은 소스를 정의만 바꿔 한 벌 더
# 컴파일하고, 하드웨어가 지원할 때만 그 파이프라인을 만든다.
set(shader_variants
    "shaders/mesh.frag|mesh_rq.frag.spv|RAY_QUERY_SHADOWS"
    "shaders/mesh_oit.frag|mesh_oit_rq.frag.spv|RAY_QUERY_SHADOWS")

foreach(variant IN LISTS shader_variants)
    string(REPLACE "|" ";" variant_parts "${variant}")
    list(GET variant_parts 0 variant_source)
    list(GET variant_parts 1 variant_name)
    list(GET variant_parts 2 variant_define)
    set(shader_binary "${shader_output_dir}/${variant_name}")
    add_custom_command(
        OUTPUT "${shader_binary}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${shader_output_dir}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}"
                --target-env=vulkan1.3
                -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                -D${variant_define}
                -MD -MF "${shader_binary}.d"
                $<IF:$<CONFIG:Debug>,-g,-O>
                -o "${shader_binary}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${variant_source}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${variant_source}"
        DEPFILE "${shader_binary}.d"
        COMMENT "셰이더 컴파일: ${variant_source} (${variant_define})"
        VERBATIM)
    list(APPEND shader_binaries "${shader_binary}")
endforeach()

add_custom_target(cg_lab_shaders DEPENDS ${shader_binaries} SOURCES ${shader_sources})

# CMake 4.4 + Ninja + MSVC 조합에서 구성별 기본 플래그(/O2, /Od, NDEBUG)가 비어 있는 채로 캐시되는
# 일이 있어 release 가 최적화 없이 빌드됐다. 직접 못 박는다. 기본값이 살아 있어도 중복은 무해하다.
if(MSVC)
    add_compile_options($<$<CONFIG:Debug>:/Od> $<$<CONFIG:Debug>:/RTC1>
                        $<$<NOT:$<CONFIG:Debug>>:/O2> $<$<NOT:$<CONFIG:Debug>>:/Ob2>)
    add_compile_definitions($<$<NOT:$<CONFIG:Debug>>:NDEBUG>)
endif()

add_executable(cg_lab
    src/main.cpp
    src/app/application.cpp
    src/asset/animation.cpp
    src/asset/gltf_loader.cpp
    src/asset/ktx2_loader.cpp
    src/asset/meshlet.cpp
    src/core/error.cpp
    src/core/job_system.cpp
    src/editor/editor.cpp
    src/gfx/context.cpp
    src/gfx/environment.cpp
    src/gfx/bindless.cpp
    src/gfx/geometry.cpp
    src/gfx/lod_network.cpp
    src/gfx/profiler.cpp
    src/gfx/profiler_math.cpp
    src/gfx/raytracing.cpp
    src/gfx/renderer.cpp
    src/gfx/shadow_math.cpp
    src/gfx/resources.cpp
    src/gfx/stb_impl.cpp
    src/gfx/texture.cpp
    src/gfx/swapchain.cpp
    src/gfx/upscaler.cpp
    src/gfx/upscaler_dlss.cpp
    src/gfx/upscaler_fsr.cpp
    src/gfx/upscaler_math.cpp
    src/gfx/upscaler_taau.cpp
    src/gfx/uploader.cpp
    src/gfx/vma_impl.cpp
    src/scene/camera.cpp
    src/scene/scene.cpp
    src/scene/scene_io.cpp)

add_dependencies(cg_lab cg_lab_shaders)

# 한글이 섞인 경로에서 좁은 문자열 변환이 깨지지 않도록 프로세스 코드 페이지를 UTF-8 로 고정한다.
if(WIN32)
    target_sources(cg_lab PRIVATE platform/windows/cg_lab.manifest)
endif()

# 서드파티 단일 헤더 구현부는 경고 검사에서 제외한다.
set_source_files_properties(src/gfx/vma_impl.cpp src/gfx/stb_impl.cpp src/asset/gltf_loader.cpp
    PROPERTIES COMPILE_OPTIONS "$<IF:$<BOOL:${MSVC}>,/w,-w>")

# VMA 헤더 설치 위치가 배포판마다 <vk_mem_alloc.h> 또는 <vma/vk_mem_alloc.h> 로 갈리므로,
# 실제 헤더를 찾아 전달 헤더를 만들고 프로젝트에서는 <vma/vk_mem_alloc.h> 하나만 쓴다.
get_target_property(vma_include_dirs GPUOpen::VulkanMemoryAllocator INTERFACE_INCLUDE_DIRECTORIES)
find_file(VMA_HEADER_PATH
    NAMES vma/vk_mem_alloc.h vk_mem_alloc.h
    HINTS ${vma_include_dirs}
    NO_DEFAULT_PATH
    REQUIRED)
file(WRITE "${CMAKE_BINARY_DIR}/compat/vma/vk_mem_alloc.h" "#pragma once\n#include \"${VMA_HEADER_PATH}\"\n")

# FSR 3.1 은 FidelityFX API 의 서명된 DLL 로 붙인다. SDK 를 통째로 빌드하면 셰이더 순열을
# 수천 개 컴파일해야 하는데, 우리가 쓰는 것은 헤더와 amd_fidelityfx_vk.dll 뿐이다. DLL 은 실행
# 시점에 찾으므로 없으면 편집기에서 FSR 만 비활성으로 나온다. 링크는 걸지 않는다.
option(CG_LAB_FSR "FidelityFX FSR 3.1 업스케일러를 포함한다" ${WIN32})
set(FIDELITYFX_TAG "v1.1.4" CACHE STRING "고정할 FidelityFX SDK 태그")
set(FIDELITYFX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/fidelityfx")

if(CG_LAB_FSR AND NOT WIN32)
    message(STATUS "FidelityFX API 는 Windows 전용이라 FSR 을 끕니다")
    set(CG_LAB_FSR OFF)
endif()

if(CG_LAB_FSR AND NOT EXISTS "${FIDELITYFX_DIR}/ffx-api/include/ffx_api/ffx_api.h")
    find_package(Git REQUIRED)
    message(STATUS "FidelityFX SDK ${FIDELITYFX_TAG} 클론: ${FIDELITYFX_DIR}")
    # 히스토리와 blob 을 받지 않고 필요한 두 디렉터리만 채운다. 저장소 전체는 수백 MB 다.
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --filter=blob:none --no-checkout --depth 1
                --branch "${FIDELITYFX_TAG}"
                https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.git "${FIDELITYFX_DIR}"
        RESULT_VARIABLE ffx_clone_result)
    if(ffx_clone_result EQUAL 0)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" sparse-checkout set --cone ffx-api/include PrebuiltSignedDLL
            WORKING_DIRECTORY "${FIDELITYFX_DIR}"
            RESULT_VARIABLE ffx_sparse_result)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" checkout "${FIDELITYFX_TAG}"
            WORKING_DIRECTORY "${FIDELITYFX_DIR}"
            RESULT_VARIABLE ffx_checkout_result)
    endif()
    if(NOT EXISTS "${FIDELITYFX_DIR}/ffx-api/include/ffx_api/ffx_api.h")
        message(WARNING "FidelityFX SDK 를 받지 못해 FSR 을 끕니다")
        set(CG_LAB_FSR OFF)
    endif()
endif()

if(CG_LAB_FSR)
    target_compile_definitions(cg_lab PRIVATE CG_LAB_FSR)
    target_include_directories(cg_lab PRIVATE "${FIDELITYFX_DIR}/ffx-api/include")
    # 서명된 DLL 을 실행 파일 옆에 둔다. 링크가 아니라 실행 시점에 찾는다.
    add_custom_command(TARGET cg_lab POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FIDELITYFX_DIR}/PrebuiltSignedDLL/amd_fidelityfx_vk.dll"
                "$<TARGET_FILE_DIR:cg_lab>/amd_fidelityfx_vk.dll"
        COMMENT "amd_fidelityfx_vk.dll 복사")
endif()

# DLSS 는 SDK 를 자동으로 받지 않는다. 재배포 조건이 붙은 SDK 라, 받을지 말지는 쓰는 쪽이 정할
# 일이다. https://github.com/NVIDIA/DLSS 를 받아 그 경로를 넘기면 켜진다.
set(CG_LAB_DLSS_SDK "" CACHE PATH "NVIDIA DLSS(NGX) SDK 경로. 비우면 DLSS 를 빼고 빌드한다")

if(CG_LAB_DLSS_SDK AND NOT WIN32)
    message(STATUS "이 저장소의 DLSS 연결은 Windows 전용입니다")
    set(CG_LAB_DLSS_SDK "")
endif()

if(CG_LAB_DLSS_SDK)
    set(ngx_include "${CG_LAB_DLSS_SDK}/include")
    set(ngx_lib_dir "${CG_LAB_DLSS_SDK}/lib/Windows_x86_64/x64")
    # 디버그 빌드는 디버그 CRT 를 쓰는 변종을 링크해야 런타임이 갈리지 않는다. 가중치 DLL 은
    # 라이브러리 변종과 무관하므로 양쪽 다 rel 을 쓴다. dev 쪽은 화면 한가운데에 배포 금지
    # 워터마크를 그려서, 렌더 결과를 눈으로 보는 이 저장소에는 맞지 않는다. NGX 쪽을 진단할 일이
    # 있으면 lib/Windows_x86_64/dev 의 DLL 로 바꿔 넣으면 된다.
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ngx_lib "${ngx_lib_dir}/nvsdk_ngx_d_dbg.lib")
    else()
        set(ngx_lib "${ngx_lib_dir}/nvsdk_ngx_d.lib")
    endif()
    set(ngx_dll_dir "${CG_LAB_DLSS_SDK}/lib/Windows_x86_64/rel")

    if(NOT EXISTS "${ngx_include}/nvsdk_ngx_vk.h" OR NOT EXISTS "${ngx_lib}")
        message(WARNING "CG_LAB_DLSS_SDK 경로에서 NGX 를 찾지 못해 DLSS 를 뺍니다: ${CG_LAB_DLSS_SDK}")
        set(CG_LAB_DLSS_SDK "")
    else()
        target_compile_definitions(cg_lab PRIVATE CG_LAB_DLSS)
        target_include_directories(cg_lab PRIVATE "${ngx_include}")
        target_link_libraries(cg_lab PRIVATE "${ngx_lib}")
        # 가중치를 담은 DLL. NGX 는 실행 파일 옆이나 초기화 때 넘긴 경로에서 찾는다.
        # 초해상(dlss)과 Ray Reconstruction(dlssd)은 가중치가 따로다.
        foreach(ngx_dll nvngx_dlss.dll nvngx_dlssd.dll)
            add_custom_command(TARGET cg_lab POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${ngx_dll_dir}/${ngx_dll}"
                        "$<TARGET_FILE_DIR:cg_lab>/${ngx_dll}"
                COMMENT "${ngx_dll} 복사")
        endforeach()
    endif()
endif()

target_include_directories(cg_lab PRIVATE src "${CMAKE_BINARY_DIR}/compat")
target_link_libraries(cg_lab PRIVATE Vulkan::Vulkan SDL3::SDL3 glm::glm imgui::imgui imguizmo::imguizmo meshoptimizer::meshoptimizer nlohmann_json::nlohmann_json spdlog::spdlog GPUOpen::VulkanMemoryAllocator)

if(MSVC)
    target_compile_options(cg_lab PRIVATE /W4 /utf-8)
else()
    # Vulkan 구조체는 sType 만 지정하고 나머지를 값 초기화하는 관용을 쓰므로 해당 경고는 끈다.
    target_compile_options(cg_lab PRIVATE -Wall -Wextra -Wno-missing-field-initializers)
endif()

# 런타임에 public/ 의 폰트와 에셋, 그리고 컴파일된 셰이더를 찾기 위한 경로.
target_compile_definitions(cg_lab PRIVATE
    CG_LAB_ASSET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/public"
    CG_LAB_SHADER_ROOT="${shader_output_dir}"
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_DEPTH_ZERO_TO_ONE)

# VS Code 등 편집기가 인클루드 경로를 찾을 수 있도록 컴파일 데이터베이스를 소스 루트에 링크한다.
# 데이터베이스는 생성 단계에서 나오므로 첫 구성에는 아직 없다. 심볼릭 링크를 만들 권한이 없는
# Windows 는 복사로 떨어지는데, 없는 파일을 복사하려 하면 구성 자체가 실패한다. 있을 때만 건다.
if(EXISTS "${CMAKE_BINARY_DIR}/compile_commands.json")
    file(CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json" "${CMAKE_SOURCE_DIR}/compile_commands.json"
         SYMBOLIC COPY_ON_ERROR)
endif()

# LOD 신경망의 역전파가 실제로 손실을 줄이는지 확인하는 자체 점검.
enable_testing()
add_executable(cg_lab_tests tests/lod_network_test.cpp src/gfx/lod_network.cpp)
add_executable(cg_lab_animation_tests tests/animation_test.cpp src/asset/animation.cpp)
add_executable(cg_lab_camera_tests tests/camera_test.cpp src/scene/camera.cpp)
add_executable(cg_lab_scene_tests tests/scene_test.cpp src/scene/scene.cpp src/asset/animation.cpp src/core/error.cpp)
add_executable(cg_lab_scene_io_tests tests/scene_io_test.cpp src/scene/scene_io.cpp src/scene/scene.cpp
    src/scene/camera.cpp
    src/asset/animation.cpp src/core/error.cpp)
add_executable(cg_lab_profiler_tests tests/profiler_test.cpp src/gfx/profiler_math.cpp)
add_executable(cg_lab_shadow_tests tests/shadow_test.cpp src/gfx/shadow_math.cpp)
add_executable(cg_lab_upscaler_tests tests/upscaler_test.cpp src/gfx/upscaler_math.cpp)
add_executable(cg_lab_vertex_pack_tests tests/vertex_pack_test.cpp)
target_include_directories(cg_lab_vertex_pack_tests PRIVATE src)
target_link_libraries(cg_lab_vertex_pack_tests PRIVATE glm::glm)
if(MSVC)
    target_compile_options(cg_lab_vertex_pack_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_vertex_pack_tests PRIVATE -Wall -Wextra)
endif()
add_executable(cg_lab_concurrency_tests tests/lock_free_queue_test.cpp src/core/job_system.cpp)
target_include_directories(cg_lab_concurrency_tests PRIVATE src)
find_package(Threads REQUIRED)
target_link_libraries(cg_lab_concurrency_tests PRIVATE Threads::Threads)
target_include_directories(cg_lab_tests PRIVATE src)
if(MSVC)
    target_compile_options(cg_lab_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_tests PRIVATE -Wall -Wextra)
endif()
target_include_directories(cg_lab_animation_tests PRIVATE src)
target_link_libraries(cg_lab_animation_tests PRIVATE glm::glm)
if(MSVC)
    target_compile_options(cg_lab_animation_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_animation_tests PRIVATE -Wall -Wextra)
endif()

target_include_directories(cg_lab_camera_tests PRIVATE src)
target_link_libraries(cg_lab_camera_tests PRIVATE SDL3::SDL3 glm::glm)
if(MSVC)
    target_compile_options(cg_lab_camera_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_camera_tests PRIVATE -Wall -Wextra)
endif()

target_include_directories(cg_lab_scene_tests PRIVATE src)
target_link_libraries(cg_lab_scene_tests PRIVATE SDL3::SDL3 spdlog::spdlog glm::glm)
target_compile_definitions(cg_lab_scene_tests PRIVATE GLM_ENABLE_EXPERIMENTAL)
if(MSVC)
    target_compile_options(cg_lab_scene_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_scene_tests PRIVATE -Wall -Wextra)
endif()

target_include_directories(cg_lab_scene_io_tests PRIVATE src)
target_link_libraries(cg_lab_scene_io_tests PRIVATE SDL3::SDL3 spdlog::spdlog glm::glm nlohmann_json::nlohmann_json)
target_compile_definitions(cg_lab_scene_io_tests PRIVATE GLM_ENABLE_EXPERIMENTAL)
if(MSVC)
    target_compile_options(cg_lab_scene_io_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_scene_io_tests PRIVATE -Wall -Wextra)
endif()

target_include_directories(cg_lab_profiler_tests PRIVATE src)
if(MSVC)
    target_compile_options(cg_lab_profiler_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_profiler_tests PRIVATE -Wall -Wextra)
endif()

target_include_directories(cg_lab_shadow_tests PRIVATE src)
target_link_libraries(cg_lab_shadow_tests PRIVATE glm::glm)
if(MSVC)
    target_compile_options(cg_lab_shadow_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_shadow_tests PRIVATE -Wall -Wextra)
endif()

target_include_directories(cg_lab_upscaler_tests PRIVATE src)
target_link_libraries(cg_lab_upscaler_tests PRIVATE glm::glm)
if(MSVC)
    target_compile_options(cg_lab_upscaler_tests PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab_upscaler_tests PRIVATE -Wall -Wextra)
endif()

# 테스트는 assert 로 확인한다. RelWithDebInfo 의 NDEBUG 를 그대로 두면 검사가 통째로 사라지고
# 쓰지 않는 변수 경고만 남는다.
foreach(test_target
        cg_lab_tests cg_lab_animation_tests cg_lab_camera_tests cg_lab_scene_tests
        cg_lab_scene_io_tests cg_lab_profiler_tests cg_lab_shadow_tests cg_lab_concurrency_tests
        cg_lab_upscaler_tests cg_lab_vertex_pack_tests)
    if(MSVC)
        target_compile_options(${test_target} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${test_target} PRIVATE -UNDEBUG)
    endif()
    # 한글이 섞인 경로를 std::filesystem::path 로 만들 때 시스템 코드 페이지로 변환하다 예외를
    # 던진다. 실행 파일과 같은 이유로 테스트에도 필요하다.
    if(WIN32)
        target_sources(${test_target} PRIVATE platform/windows/cg_lab.manifest)
    endif()
endforeach()

add_test(NAME lod_network COMMAND cg_lab_tests)
add_test(NAME animation COMMAND cg_lab_animation_tests)
add_test(NAME camera COMMAND cg_lab_camera_tests)
add_test(NAME scene COMMAND cg_lab_scene_tests)
add_test(NAME scene_io COMMAND cg_lab_scene_io_tests)
add_test(NAME profiler COMMAND cg_lab_profiler_tests)
add_test(NAME shadow COMMAND cg_lab_shadow_tests)
add_test(NAME upscaler COMMAND cg_lab_upscaler_tests)
add_test(NAME concurrency COMMAND cg_lab_concurrency_tests)
add_test(NAME vertex_pack COMMAND cg_lab_vertex_pack_tests)

# cmake --build --preset debug --target run 으로 빌드한 뒤 바로 실행한다.
add_custom_target(run
    COMMAND cg_lab
    DEPENDS cg_lab
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    USES_TERMINAL)
