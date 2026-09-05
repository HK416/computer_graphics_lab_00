#pragma once

#include "app/plugin.h"

namespace app {

// 유체(SPH). 하드웨어 프로파일의 입자 상한을 적용하고 편집기 «유체» 절을 그린다.
//
// ponytail: gfx::FluidSimulator 는 Renderer 에 남는다. 입자 인스턴스가 오브젝트 인스턴스 뒤에 이어 붙고 TLAS
// 인스턴스 앞에 붙는 배치(buildDrawCommands)와 묶여 있어 떼어 내면 그 배치 규칙을 둘이 나눠 갖게 된다.
class FluidPlugin : public Plugin {
public:
    const char* name() const override { return "유체"; }
    void build(Services& services) override;
    void ui(Services& services) override;
};

} // namespace app
