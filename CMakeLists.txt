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

project(cg_lab VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Vulkan REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(cg_lab
    src/main.cpp
    src/app/application.cpp
    src/core/error.cpp)

target_include_directories(cg_lab PRIVATE src)
target_link_libraries(cg_lab PRIVATE Vulkan::Vulkan SDL3::SDL3 spdlog::spdlog)

if(MSVC)
    target_compile_options(cg_lab PRIVATE /W4 /utf-8)
else()
    target_compile_options(cg_lab PRIVATE -Wall -Wextra)
endif()

# 런타임에 public/ 의 폰트와 에셋을 찾기 위한 경로.
target_compile_definitions(cg_lab PRIVATE CG_LAB_ASSET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/public")
