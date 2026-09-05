#pragma once

#include "core/job_system.h"
#include "editor/editor.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/hardware_profile.h"
#include "gfx/renderer.h"
#include "scene/scene.h"

namespace app {

struct Options;

// 플러그인이 만질 수 있는 것 전부. Application 이 소유하는 객체의 참조라 플러그인보다 오래 산다.
//
// ponytail: 헤드리스(창·렌더러 없는 물리 전용 실행)가 오면 gfx 멤버를 포인터로 바꾸고 플러그인이 null 을
// 살핀다. 지금은 모두 있다고 보고 참조로 둔다. settings 는 렌더러 없이도 있으므로 그때도 참조로 남는다.
struct Services {
    scene::SceneManager& scenes;
    core::JobSystem& jobs;
    const Options& options;
    const gfx::HardwareProfile& profile;
    gfx::RenderSettings& settings;
    gfx::Context& context;
    gfx::BindlessTextures& bindless;
    gfx::GeometryStore& geometry;
    gfx::Renderer& renderer;
    editor::Editor& editor;
};

// 기능 하나. Application 이 등록한 순서대로 훅을 부른다(Bevy 의 Plugin 과 같은 자리).
//
// 프레임 흐름에서의 자리: 이벤트 → camera.update → scene.update → **update()** → renderer.prepareFrame →
// editor.build(안에서 **ui()**) → scene.refresh → renderer.drawFrame.
struct Plugin {
    virtual ~Plugin() = default;
    virtual const char* name() const = 0;
    // 기동 시 한 번. 자원 생성, 렌더 패스 등록, 옵션·하드웨어 프로파일 읽기.
    virtual void build(Services& services) { (void)services; }
    // 프레임 앞. scene.update 뒤, renderer.prepareFrame 앞.
    virtual void update(Services& services, float deltaSeconds) {
        (void)services;
        (void)deltaSeconds;
    }
    // 편집기 «렌더 설정» 창 안의 절. editor.settingsSection 으로 접는 머리를 만든다.
    virtual void ui(Services& services) { (void)services; }
};

} // namespace app
