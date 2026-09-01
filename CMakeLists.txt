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
    shaders/tonemap.frag)

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

add_custom_target(cg_lab_shaders DEPENDS ${shader_binaries} SOURCES ${shader_sources})

add_executable(cg_lab
    src/main.cpp
    src/app/application.cpp
    src/asset/gltf_loader.cpp
    src/asset/meshlet.cpp
    src/core/error.cpp
    src/editor/editor.cpp
    src/gfx/context.cpp
    src/gfx/bindless.cpp
    src/gfx/geometry.cpp
    src/gfx/renderer.cpp
    src/gfx/resources.cpp
    src/gfx/stb_impl.cpp
    src/gfx/texture.cpp
    src/gfx/swapchain.cpp
    src/gfx/uploader.cpp
    src/gfx/vma_impl.cpp
    src/scene/camera.cpp
    src/scene/scene.cpp)

add_dependencies(cg_lab cg_lab_shaders)

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

target_include_directories(cg_lab PRIVATE src "${CMAKE_BINARY_DIR}/compat")
target_link_libraries(cg_lab PRIVATE Vulkan::Vulkan SDL3::SDL3 glm::glm imgui::imgui imguizmo::imguizmo meshoptimizer::meshoptimizer spdlog::spdlog GPUOpen::VulkanMemoryAllocator)

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
file(CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json" "${CMAKE_SOURCE_DIR}/compile_commands.json"
     SYMBOLIC COPY_ON_ERROR)
