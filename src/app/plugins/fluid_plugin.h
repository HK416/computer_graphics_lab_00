#pragma once

#include "app/plugin.h"

namespace app {

// 유체(SPH). 하드웨어 프로파일의 입자 상한을 적용하고 편집기 «유체» 절을 그린다.
//
// ponytail: gfx::FluidSimulator 는 Renderer 에 남는다. 입자 인스턴스가 오브젝트 인스턴스 뒤에 이어 붙고 TLAS
// 인스턴스 앞에 붙는 배치(buildDrawCommands)와 묶여 있어 떼어 내면 그 배치 규칙을 둘이 나눠 갖게 된다. 그 밖에도
// fluidTlasPrepended 를 TLAS 를 세우는 노드 넷이 읽고, fluidSurfaceTables 를 경로 추적·반사·DDGI 가 푸시 상수로
// 받으며, 표면 래스터 패스가 렌더러의 대상 다섯을 쓴다(결합 지점 13개, 2026-09 조사). 강체 GPU 솔버처럼 아무것도
// 읽지 않는 것이 아니라 플러그인으로 옮기지 않는다.
class FluidPlugin : public Plugin {
public:
    const char* name() const override { return "유체"; }
    void build(Services& services) override;
    void ui(Services& services) override;
};

} // namespace app
