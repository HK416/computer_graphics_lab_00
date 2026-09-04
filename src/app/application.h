#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "asset/load_progress.h"
#include "asset/model.h"
#include "core/job_system.h"
#include "editor/editor.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/renderer.h"
#include "gfx/texture.h"
#include "scene/scene.h"

struct SDL_Window;

namespace app {

inline constexpr uint32_t AUTOMATIC_LOD = 0xFFFFFFFFU;

struct Options {
    // 지정하면 몇 프레임 뒤에 화면을 PNG 로 저장하고 종료한다. 렌더 결과 검증용이다.
    std::filesystem::path screenshotPath;
    // 몇 프레임째를 저장할지. 시간축 업스케일처럼 여러 프레임을 쌓는 기능은 뒤쪽을 봐야 한다.
    uint64_t screenshotFrame = 8;
    // 지정하면 시작할 때 이 장면 파일을 연다.
    std::filesystem::path scenePath;
    // 시작할 때 불러올 glTF 모델들. 편집기의 "모델" 단추와 같은 경로를 탄다.
    std::vector<std::filesystem::path> modelPaths;
    // shaders/scene_data.glsl 의 DEBUG_MODE_* 값.
    uint32_t debugMode = 0;
    // 0 이면 하드웨어 동시성에 맞춰 정한다.
    unsigned threadCount = 0;
    // AUTOMATIC_LOD 면 오차 기반 자동 선정을 쓴다.
    uint32_t lodLevel = AUTOMATIC_LOD;
    float lodErrorThreshold = 1.0F;
    bool neuralLod = false;
    // 시작하자마자 구간 계측을 켠다. 스크린샷으로 확인할 때 쓴다.
    bool profile = false;
    float renderScale = 1.0F;
    // 0 통과, 1 내장 공간 업스케일
    uint32_t upscaler = 1;
    // 시작할 때 경로 추적을 켠다. 하드웨어가 지원하지 않으면 사유를 남기고 무시한다.
    bool pathTracing = false;
    // 스크린샷 비교용. 두 패스 오클루전 컬링을 끄거나 mesh shader 경로 대신 컴퓨트 컬링 경로를 쓴다.
    bool occlusionCulling = true;
    bool meshShader = true;
    // 시작할 때 광선 반사를 켠다. 광선 질의가 없으면 렌더러가 스스로 끈다.
    bool reflections = false;
    // 프레임마다 카메라를 이만큼(도) 궤도 회전한다. 정지 화면에서는 드러나지 않는 팝인을 재현한다.
    float orbitDegreesPerFrame = 0.0F;
    float triangleBudget = 0.0F;
    // 0 이면 드라이버가 알려 주는 장치 메모리 예산을 쓴다. 메가바이트. 여유를 두거나 게이트를 시험할 때 준다.
    uint64_t gpuBudgetMegabytes = 0;
    // 위치·UV 가 같은 정점을 합칠 노멀 스무딩 각도. 0 이면 속성이 완전히 같을 때만 합친다.
    float weldAngleDegrees = 30.0F;
    // 시작하자마자 재생(물리 시뮬레이션)한다. 스크린샷으로 물리를 확인할 때 쓴다.
    bool play = false;
};

class Application {
public:
    explicit Application(const Options& options);
    ~Application();
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    void run();

private:
    // 한 번 적재한 모델. 같은 파일을 두 번 올리지 않고, 장면 파일이 가리킬 대상이 된다.
    struct LoadedModel {
        std::filesystem::path path;
        uint32_t meshBase = 0;
        uint32_t meshCount = 0;
        asset::Skeleton skeleton;
        std::vector<asset::Instance> instances;
        // 지오메트리 저장소와 텍스처 캐시에서 이 모델이 차지한 자리. 해제할 때 돌려준다.
        gfx::GeometryStore::ModelRange range;
        std::vector<uint32_t> textureSlots;
        // 코드로 만든 내장 모델(구). 파일이 없고 해제 대상도 아니다.
        bool builtin = false;
        // 해제된 항목. 애니메이터가 번호로 가리키므로 지우지 않고 자리만 비워 두었다가 다음 모델이 쓴다.
        bool unloaded = false;
    };

    // 어느 장면도(그리고 force 가 아니면 편집기 되돌리기 기록도) 가리키지 않는 모델을 GPU 에서 내린다.
    // 장면의 오브젝트 수나 부모 관계가 바뀔 때, 장면을 옮길 때, 메뉴에서 부른다.
    void collectUnusedModels(bool force);

    // 기본 장면 «GameScene» 하나를 만든다. 에셋은 편집기나 --model 로 올린다.
    void loadScenes();
    // 파일 없이 코드로 만드는 모델. 유체 입자와 «메쉬 › 구» 프리미티브가 쓰는 구 하나다.
    void registerBuiltinModels();

    // 해석된 모델을 GPU 에 올리기 전에 하는 CPU 작업 전부. 텍스처 디코딩과 LOD 계층 구축이다.
    // 어느 스레드에서든 부를 수 있고 안에서 워커를 나눠 쓴다. 단계별 시간을 ms 로 돌려준다.
    struct PrepareTimings {
        double textureMs = 0.0;
        double lodMs = 0.0;
    };
    PrepareTimings prepareAssets(std::vector<asset::Texture*>& allTextures,
                                 std::vector<asset::Mesh*>& allMeshes,
                                 asset::LoadProgress* progress);
    PrepareTimings prepareModel(asset::Model& model, asset::LoadProgress* progress);

    // 이 모델이 GPU 예산에 들어가는지. 지오메트리와 텍스처에 재할당 중 겹치는 옛 버퍼까지 더해 남은
    // 예산과 견준다. 안 들어가면 사유를 로그에 남기고 false. 장치를 잃는 것보다 여기서 거부하는 편이 낫다.
    bool fitsGpuBudget(const asset::Model& model) const;
    // 실행 인자에서 나온 적재 방식.
    asset::LoadSettings loadSettings() const;
    // 이미 해석해 둔 모델을 GPU 에 올리고 등록 번호를 돌려준다. 지오메트리 재구축은 부르는 쪽 몫이다.
    uint32_t registerModel(const std::filesystem::path& path, asset::Model& model, bool builtin = false);
    // 같은 파일이 이미 올라가 있으면 그 등록 번호를, 아니면 loadedModels.size() 를 돌려준다.
    uint32_t findModel(const std::filesystem::path& path) const;
    // 모델을 아직 올리지 않았으면 해석해서 올리고 등록 번호를 돌려준다. 부르는 스레드에서 끝까지 한다.
    // 해석에 실패하면 loadedModels.size() 를 돌려준다.
    uint32_t ensureModel(const std::filesystem::path& path);
    // 등록된 모델로 장면에 뿌리 오브젝트와 자식들을 만든다.
    void instantiateModel(uint32_t modelIndex, scene::Scene& scene);

    // 편집기가 부르는 런타임 적재. 해석은 백그라운드 스레드가 하고, 끝나면 pumpLoads 가 GPU 에 올려
    // 활성 장면에 붙인다. 창이 멈추지 않는다. 적재 중에 또 부르면 줄을 서서 차례로 처리한다.
    void requestModel(const std::filesystem::path& path);
    // 줄 선 요청 가운데 하나를 백그라운드로 시작한다. 이미 올라간 모델이면 바로 붙인다.
    void startNextLoad();
    // 프레임마다 한 번. 백그라운드 해석이 끝났으면 GPU 에 올리고 장면에 붙인다.
    void pumpLoads();
    // 해석이 끝나 스레드까지 합류한 pendingLoad 를 GPU 에 올리고 장면에 붙인 뒤 비운다.
    void completeLoad();
    // 시작 인자처럼 결과가 바로 필요한 자리에서 쓴다. 요청하고 큐가 빌 때까지 기다린다.
    void loadModel(const std::filesystem::path& path);
    // 편집기가 진행 막대에 보여줄 상태.
    editor::LoadStatus loadStatus() const;
    // 장면을 커스텀 JSON 으로 저장하고 읽는다. 읽은 장면은 새 장면으로 추가한 뒤 전환한다.
    void saveScene(const std::filesystem::path& path);
    void openScene(const std::filesystem::path& path);

    SDL_Window* window = nullptr;
    std::unique_ptr<gfx::Context> context;
    std::unique_ptr<gfx::BindlessTextures> bindless;
    std::unique_ptr<gfx::TextureCache> textures;
    std::unique_ptr<gfx::GeometryStore> geometry;
    std::unique_ptr<gfx::Renderer> renderer;
    std::unique_ptr<editor::Editor> editorUi;
    core::JobSystem jobs;

    // 백그라운드에서 해석 중인 모델 하나. 스레드는 model 과 progress 만 만지고, GPU 자원과 장면은
    // 끝난 뒤 메인 스레드가 완성한다. jobs 보다 뒤에 있어야 먼저 파괴되어 워커를 쓰는 채로 남지 않는다.
    struct PendingLoad {
        std::filesystem::path path;
        asset::Model model;
        asset::LoadProgress progress;
        std::atomic<bool> finished{false};
        // 해석에 실패했다. finished 와 함께 세워지고, 메인 스레드가 로그만 남기고 버린다.
        std::atomic<bool> failed{false};
        // 업로드 단계를 한 프레임 보여준 뒤 올린다. 큰 모델은 업로드만 몇 초라 표시가 있어야 한다.
        bool uploadShown = false;
        // 요청한 순간의 활성 장면. 적재 중에 장면을 옮겨도 요청한 곳에 붙는다.
        size_t sceneIndex = 0;
        uint64_t startTicks = 0;
        PrepareTimings timings;
        std::thread worker;
        ~PendingLoad() {
            if (worker.joinable()) {
                worker.join();
            }
        }
    };
    std::unique_ptr<PendingLoad> pendingLoad;
    std::deque<std::filesystem::path> loadQueue;

    scene::SceneManager scenes;
    std::filesystem::path assetRoot;
    std::filesystem::path sceneRoot;
    std::vector<LoadedModel> loadedModels;
    // 기본 도형마다의 전역 메쉬 번호. asset::Primitive 순서와 같다.
    std::vector<uint32_t> primitiveMeshes;
    // 마지막으로 미사용 모델을 살핀 때의 장면 번호와 그 장면의 구조 리비전.
    size_t collectedScene = SIZE_MAX;
    uint64_t collectedTopology = 0;
    // 강체 물리의 고정 간격 누적기. 프레임이 길어도 정해진 스텝 수까지만 따라잡는다.
    float physicsAccumulator = 0.0F;
    float orbitDegreesPerFrame = 0.0F;
    Options options;
};

} // namespace app
