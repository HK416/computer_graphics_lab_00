#pragma once

#include "app/plugin.h"

namespace app {

// 구간 계측. --profile 로 켜고, 편집기 «프로파일러» 절을 그리며, 종료할 때 결과를 로그로 남긴다(창을 못 보는
// 스크린샷·CI 실행에서도 결과가 남는다). 계측기 자체(gfx::GpuProfiler)는 커맨드 버퍼 기록과 묶여 Renderer 에 있다.
class ProfilerPlugin : public Plugin {
public:
    ~ProfilerPlugin() override;
    const char* name() const override { return "프로파일러"; }
    void build(Services& services) override;
    void ui(Services& services) override;

private:
    gfx::Renderer* renderer = nullptr;
};

} // namespace app
