#pragma once

#include "app/plugin.h"

namespace app {

// 콜라이더·유체 경계 표시. --no-colliders 를 적용하고 편집기 절을 그린다.
//
// ponytail: 선 파이프라인과 정점 버퍼, 기록(recordDebugLines)은 Renderer 에 남는다. 표시 해상도 깊이 슬롯과
// 후처리 샘플러를 함께 쓰기 때문이다. 플러그인이 갖게 하려면 그 둘을 Renderer 가 공개해야 한다.
class DebugLinesPlugin : public Plugin {
public:
    const char* name() const override { return "디버그 선"; }
    void build(Services& services) override;
    void ui(Services& services) override;
};

} // namespace app
