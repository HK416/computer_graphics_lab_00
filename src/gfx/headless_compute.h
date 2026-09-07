#pragma once

#include <cstdint>
#include <functional>

#include <vulkan/vulkan.h>

namespace gfx {

struct Context;

// 명령을 한 번에 하나씩 제출하고 끝날 때까지 기다리는 자리. 이름은 headless 지만 **큐는 그래픽스**라
// 그리기도 기록할 수 있다(관측 렌더가 그렇게 쓴다). `--headless` 의 GPU 물리가 처음 쓰던 자리다.
// 렌더러의 프레임 루프(여러 프레임을 겹쳐 돌리고 되읽기를 몇 프레임 미루는 것)를 흉내 내지 않는다 —
// 기다릴 화면이 없으니 겹칠 이유가 없고, 기다리는 편이 결과가 곧바로 장면에 들어와 스텝 하나가 곧 한
// 프레임이 된다.
class HeadlessCompute {
public:
    // collectRetired 가 참이면 제출이 끝날 때마다 Context 의 은퇴 자원을 거둔다. **창이 있는 실행에서
    // 이것을 켜면 안 된다** — 이 자리는 자기 제출만 기다리는데 렌더러가 맡긴 자원은 아직 도는 프레임이
    // 읽고 있을 수 있다. 게다가 거둬 버리면 Renderer::drawFrame 의 «맡긴 것이 있으면 전부 기다린다»
    // 보호 장치가 hasRetired() 거짓을 보고 통째로 건너뛴다.
    explicit HeadlessCompute(Context& context, bool collectRetired = true);
    ~HeadlessCompute();
    HeadlessCompute(const HeadlessCompute&) = delete;
    HeadlessCompute& operator=(const HeadlessCompute&) = delete;

    // record 를 명령 버퍼에 기록해 제출하고 끝날 때까지 기다린다. 이번 제출의 번호(0 부터 하나씩
    // 올라간다)를 record 에 함께 넘기고 그대로 돌려준다. 되읽기 슬롯을 고르는 «프레임 번호» 로 쓴다.
    // 번호를 밖에서 따로 세지 않게 인자로 주는 것이라, 부르는 쪽과 한 칸 어긋날 자리가 없다.
    uint64_t submit(const std::function<void(VkCommandBuffer, uint64_t)>& record);

private:
    Context& context;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint64_t submitted = 0;
    bool collectRetired = true;
};

} // namespace gfx
