#pragma once

// editor_*.cpp 가 함께 쓰는 include 와 상수. Editor 의 선언은 editor.h 하나에 있고, 정의만 기능별 번역 단위로 나뉜다
// (renderer_internal.h 와 같은 방식). 다른 곳에서 include 하지 않는다.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include "asset/primitives.h"
#include "core/error.h"
#include "editor/editor.h"
#include "editor/log_sink.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/profiler.h"
#include "gfx/renderer.h"
#include "gfx/rigid_body_gpu.h"
#include "gfx/shadow_math.h"
#include "scene/scene.h"

namespace editor {

// 도킹 창 이름. 도크스페이스가 배치를 짜고 패널들이 같은 이름으로 연다.
inline constexpr const char* WINDOW_HIERARCHY = "계층";
inline constexpr const char* WINDOW_INSPECTOR = "인스펙터";
inline constexpr const char* WINDOW_SCENE = "장면";
inline constexpr const char* WINDOW_CONSOLE = "콘솔";
inline constexpr const char* WINDOW_SETTINGS = "렌더 설정";
// 플러그인이 여는 창. 이름은 ProfilerPlugin::window 의 리터럴과 같아야 기본 배치에 도킹된다.
inline constexpr const char* WINDOW_PROFILER = "프로파일러";

} // namespace editor
