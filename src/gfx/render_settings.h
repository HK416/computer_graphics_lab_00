#pragma once

#include <cstdint>

#include "gfx/fluid.h"
#include "gfx/raytracing.h"
#include "gfx/upscaler.h"

namespace gfx {

// 사용자가 만지는 렌더 설정 전부. 앱 전역이며 장면 JSON 에는 넣지 않는다(renderScale·upscaler 가
// 장치 의존이라 다른 기계에서 열면 어긋난다). 명령줄·자동 튜닝·편집기·플러그인이 같은 인스턴스를
// 고치고 렌더러가 읽는다. 장치가 못 하는 항목은 렌더러가 여기서 직접 끈다(BLAS 실패, 업스케일러
// 폴백, mesh shader 가용).
struct RenderSettings {
    // 업스케일 설정. 배율을 낮추면 장면을 작게 그린 뒤 확대한다.
    float renderScale = 1.0F;
    Upscaler upscaler = Upscaler::SPATIAL;
    float upscaleSharpness = 0.25F;

    float exposure = 1.0F;
    bool wireframe = false;
    // 강체 콜라이더와 유체 경계를 선으로 덧그린다. Unity 의 기즈모처럼 물체 뒤로 숨는다.
    bool showColliders = true;
    bool colliderOcclusion = true;
    // shaders/scene_data.glsl 의 DEBUG_MODE_* 값.
    uint32_t debugMode = 0;
    // 자동 LOD 선정을 끄면 이 단계를 강제한다.
    bool automaticLod = true;
    uint32_t lodLevel = 0;
    // 허용할 화면 공간 오차(픽셀). 클수록 낮은 단계를 고른다.
    float lodErrorThreshold = 1.0F;

    // 그림자. 방향광과 스폿광은 시점 하나, 점광은 여섯 면을 아틀라스 타일에 담는다.
    bool shadowsEnabled = true;
    // 시점별 절두체 컬링과, 그림자가 화면에 닿을 수 없는 캐스터를 버리는 스윕 컬링.
    bool shadowViewCulling = true;
    bool shadowCasterCulling = true;
    // 방향광 캐스케이드. 층이 모자라면 자동으로 줄어든다.
    uint32_t shadowCascades = 4;
    float shadowSplitLambda = 0.85F;
    // 0 이면 장면 크기에서 자동으로 정한다.
    float shadowDistance = 0.0F;
    // 광원과 캐스터가 그대로인 시점은 다시 그리지 않는다.
    bool shadowCaching = true;

    // 화면 공간 주변광 차폐.
    bool useSsao = true;
    // 환경광을 IBL 로 계산한다. 끄면 균일 환경광만 남는다.
    bool useIbl = true;
    // 하이브리드 그림자: 카메라에서 이 거리 안쪽은 광선으로 가시성을 판정하고 나머지는 그림자 맵을
    // 그대로 쓴다. 광선 질의를 지원하는 장치에서만 켤 수 있다.
    bool useRayQueryShadows = false;
    float rayShadowDistance = 12.0F;
    // 광선 반사: 거칠기가 상한 이하인 불투명 표면의 스페큘러 IBL 을 추적한 반사로 바꾼다. 광선
    // 질의와 IBL 이 있어야 하고, 경로 추적 중에는 그쪽이 반사를 직접 계산하므로 꺼진다.
    bool useReflections = false;
    float reflectionRoughnessCutoff = 0.6F;
    float reflectionIntensity = 1.0F;
    uint32_t reflectionMaxSamples = 16;
    // 장면 반지름에 대한 비율. 장면 크기가 제각각이라 절대 길이로 두지 않는다.
    float ssaoRadius = 0.04F;
    float ssaoIntensity = 1.0F;
    float ssaoBias = 0.002F;
    uint32_t ssaoSamples = 16;

    // GPU 컴퓨트가 meshlet 단위로 컬링하고 간접 그리기 명령을 만든다.
    bool useComputeCulling = true;
    bool frustumCulling = true;
    bool coneCulling = true;
    bool occlusionCulling = true;

    // 신경망이 LOD 임계값을 보정해 삼각형 예산을 맞춘다.
    bool useNeuralLod = false;
    bool trainLodNetwork = true;
    float triangleBudget = 60000.0F;

    // 경로 추적. 하드웨어가 지원하고 가속 구조가 예산에 들어갈 때만 켤 수 있다.
    bool usePathTracing = false;
    PathTraceOptions pathTrace;

    // 유체 부품이 요청해도 이보다 많은 입자는 뿌리지 않는다. 하드웨어 프로파일이 정한다.
    uint32_t fluidParticleLimit = FLUID_MAX_PARTICLES;

    // mesh shader 미지원 장치에서는 켤 수 없다. 파이프라인을 만들면 렌더러가 켠다.
    bool useMeshShader = false;
};

} // namespace gfx
