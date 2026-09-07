#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"

struct SDL_Window;

namespace gfx {

// 렌더 경로 선택에 쓰이는 선택적 하드웨어 기능. 미지원 기능에 대한 소프트웨어 폴백은 만들지 않고,
// 해당 경로 자체를 비활성화한다.
struct Capabilities {
    bool meshShader = false;
    bool taskShader = false;
    bool accelerationStructure = false;
    bool rayTracingPipeline = false;
    bool rayQuery = false;
    // 하위 가속 구조의 삼각형 수를 장치 버퍼에서 읽어 세울 수 있는지(간접 구축). GPU 가 만든 물 표면을
    // 광선 경로에 올리는 데 쓴다. 없으면 물 표면은 Path Tracing 에 보이지 않는다.
    bool accelerationStructureIndirectBuild = false;
    // 간접 그리기 개수를 GPU 버퍼에서 읽어 압축 드로우가 가능한지. 없으면 고정 개수로 디스패치하고
    // 컬링된 드로우는 instanceCount 를 0 으로 기록한다.
    bool drawIndirectCount = false;
    bool pipelineStatistics = false;
    bool shaderFloat16 = false;
    // FSR 셰이더가 SPIR-V Int16 능력을 선언한다.
    bool shaderInt16 = false;
    bool shaderInt8 = false;
    bool subgroupSizeControl = false;
    bool textureCompressionBc = false;
    bool textureCompressionAstc = false;
    bool memoryBudget = false;
    bool memoryPriority = false;
    // 타임스탬프 쿼리로 GPU 구간 시간을 잴 수 있는지. 주기와 유효 비트가 모두 있어야 한다.
    bool timestamps = false;
    // 그림자 절두체 근평면 앞의 캐스터를 잘라 내지 않고 눌러 담을 수 있는지.
    bool depthClamp = false;
    // 1 화소보다 굵은 선을 그릴 수 있는지. 콜라이더 표시가 굵게 그리는 데 쓴다.
    bool wideLines = false;
    // 정점 셰이더가 gl_DrawID 를 읽을 수 있는지. MoltenVK 는 shaderDrawParameters 를 지원한다고
    // 보고하지만 MSL 에 DrawIndex 가 없어 SPIR-V 변환에서 죽는다. 없으면 meshlet 디버그 뷰가 메쉬
    // 단위로 뭉개진다.
    bool shaderDrawIndex = false;
    // Vulkan 메모리 모델. 협력 행렬 셰이더가 GL_KHR_memory_scope_semantics 를 통해 이 능력을 선언하므로
    // 켜 두지 않으면 셰이더 모듈 생성이 규정 밖이다(검증 레이어가 VUID-...-pCode-08740 로 잡는다).
    // 다른 셰이더는 그대로 GLSL450 모델을 쓴다 — 모델은 모듈마다 선언한다.
    //
    // **vulkanMemoryModelDeviceScope 까지 있어야 참이다.** 이 기능을 켜는 순간 규격의 제약이 장치 전체에
    // 걸려, 장치 범위 원자 연산을 쓰는 **모든** 셰이더가 그 짝 기능을 요구한다(VUID-...-06265). 컬링·유체·
    // 강체가 전부 device scope 원자 합을 쓰므로 둘을 떼어 켤 수 없다.
    bool vulkanMemoryModel = false;
    // 협력 행렬(텐서 코어). 신경망 선형 층의 **가속 변종**만 쓴다. 기본 경로는 언제나 FMA 커널이다.
    // 미리 컴파일해 둔 모양 후보와 장치가 광고하는 모양의 교집합이 비면 꺼진다.
    bool cooperativeMatrix = false;
    // 고른 모양의 셰이더 변종 이름. **모양과 셰이더를 한 자리에서 고른다** — 이름을 부르는 쪽에서
    // 다시 조합하면 후보를 더할 때 모양은 바뀌고 셰이더는 안 바뀌는 어긋남이 생긴다(그러면 디스패치는
    // 32 행 타일을 세는데 셰이더는 16 행만 채워 출력 아래 절반이 비고, 컴파일러도 검증 레이어도 잡지
    // 못한다). 정적 수명 리터럴을 가리킨다.
    const char* coopShader = nullptr;
    // 고른 모양. cooperativeMatrix 가 false 면 전부 0 이다.
    uint32_t coopM = 0;
    uint32_t coopN = 0;
    uint32_t coopK = 0;
    // A/B 가 fp32 인가. 그러면 캐스팅이 없어 FMA 경로와 아주 가깝고, false 면 fp16 A/B 에 fp32 누산기라
    // 오차가 두 자릿수 커진다. 자기 검사의 허용치가 이 값에 따라 갈린다.
    bool coopFloat32 = false;
    // 협력 행렬 커널을 못 박아 만들 서브그룹 크기. 협력 행렬은 서브그룹 전체가 함께 도는 연산이라
    // 작업 그룹이 정확히 한 서브그룹이어야 한다. 드라이버가 컴퓨트 단계의 크기 고정을 허용하지 않으면
    // 모양이 있어도 가속을 켜지 않는다.
    uint32_t coopSubgroupSize = 0;
    uint32_t subgroupSize = 0;
};

// 용도별 큐 패밀리. 전용 패밀리가 없으면 그래픽스 패밀리로 접힌다.
struct QueueFamilies {
    uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
    uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
    // 그래픽스 큐가 돌려주는 타임스탬프의 유효 비트 수. 0 이면 그 큐에서 타임스탬프를 쓸 수 없다.
    uint32_t graphicsTimestampBits = 0;

    bool hasAsyncCompute() const { return compute != graphics; }
    bool hasSeparateTransfer() const { return transfer != graphics && transfer != compute; }
};

struct Context {
    // window 가 null 이면 서피스도 스왑체인도 없는 헤드리스 장치를 만든다(--headless 의 GPU 물리).
    explicit Context(SDL_Window* window);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    QueueFamilies queueFamilies;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;

    Capabilities caps;

    // 장치 전용 힙의 예산과 사용량(바이트). VK_EXT_memory_budget 이 있으면 드라이버가 알려 주는 예산이고,
    // 없으면 VMA 가 힙 크기의 80% 를 예산으로, 자기가 할당한 만큼만 사용량으로 본다. 큰 모델을 올리기
    // 전에 들어갈지 미리 재는 데 쓴다.
    struct MemoryBudget {
        VkDeviceSize budget = 0;
        VkDeviceSize usage = 0;
    };
    MemoryBudget deviceMemoryBudget() const;
    // 장치 전용 힙의 «크기» 합. 예산과 달리 다른 프로세스나 이미 만든 자원에 영향받지 않아 기기의
    // 성격을 재는 데 쓴다. 자동 튜닝의 등급 판정이 이 값을 본다.
    VkDeviceSize deviceLocalMemoryBytes() const;
    // 0 이 아니면 드라이버 예산 대신 이 값을 예산으로 본다. --gpu-budget 이 채우며, 모델과 가속 구조의
    // 예산 검사가 모두 같은 값을 보게 여기 둔다.
    VkDeviceSize memoryBudgetOverride = 0;

    // 진행 중인 프레임이 아직 읽고 있을 수 있는 자원을 바로 지우지 않고 모아 둔다. 편집기가 유체
    // 부품을 떼는 것처럼 자원이 사라지는 일은 프레임 한가운데 일어나는데, 그때마다 장치를 세우면
    // 편집할 때마다 화면이 끊긴다. 대신 다음 프레임 첫머리에서 지운다.
    void retireBuffer(Buffer& buffer);
    // 파괴 함수가 여기 없는 자원(가속 구조 등)을 같은 규칙으로 맡긴다. 맡긴 순서대로 지우므로,
    // 다른 자원을 붙들고 있는 것을 먼저 맡기면 수명 순서도 지켜진다.
    void retireDeferred(std::function<void()> destroy);
    bool hasRetired() const { return !retiredResources.empty(); }
    // 제출이 하나도 남아 있지 않을 때 부른다. 맡긴 것을 전부 지운다. «FRAMES_IN_FLIGHT 뒤» 같은
    // 프레임 번호 판정으로는 모자란다. MoltenVK 는 장치 주소 버퍼를 제출마다 전부 상주시키므로,
    // 살아 있던 동안 제출된 명령이 하나라도 돌고 있으면 지우는 순간 장치가 죽는다.
    void collectRetired();

private:
    std::vector<std::function<void()>> retiredResources;
};

} // namespace gfx
