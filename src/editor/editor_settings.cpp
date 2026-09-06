// 렌더 설정 패널과 플러그인 설정 절 훅.
// Editor 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 editor.h 하나에 있다.

#include "editor/editor_internal.h"

namespace editor {

void Editor::buildRenderSettings(scene::Scene& active, float deltaSeconds) {
    if (!showRenderSettings) {
        return;
    }
    // 도킹되지 않는 떠 있는 창. 열어 둔 채 장면을 돌려 보며 값을 만질 수 있다.
    ImGui::SetNextWindowSize(ImVec2{440.0F, 680.0F}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(WINDOW_SETTINGS, &showRenderSettings, ImGuiWindowFlags_NoDocking)) {
        ImGui::End();
        return;
    }

    frameTimeMilliseconds = frameTimeMilliseconds * 0.9F + deltaSeconds * 1000.0F * 0.1F;
    ImGui::Text("프레임 %.2f ms (%.0f FPS)",
                frameTimeMilliseconds,
                frameTimeMilliseconds > 0.0F ? 1000.0F / frameTimeMilliseconds : 0.0F);
    ImGui::Text("렌더 해상도 %ux%u", renderer.renderExtent().width, renderer.renderExtent().height);
    ImGui::Text("작업 워커 %u", workerCount);

    // 설정이 길어 그룹을 접을 수 있게 하고, 검색어가 있으면 이름에 그 말이 든 그룹만 펼친다.
    ImGui::SetNextItemWidth(200.0F);
    ImGui::InputTextWithHint("##settingsFilter", "그룹 검색", settingsFilter.data(), settingsFilter.size());
    auto section = [this](const char* name) { return settingsSection(name); };

    // Path Tracing은 래스터 패스를 통째로 건너뛴다. 거기 딸린 설정은 눌러도 아무 일이 없으므로
    // 디버그 뷰와 같은 이유로 잠근다.
    bool rasterOnly = renderer.settings.usePathTracing;
    // 광선 그림자와 반사는 래스터 안에서 도는 것이라 Path Tracing과는 무관하다.
    bool rayQueryReady = renderer.rayQueryShadowsAvailable() && !rasterOnly;
    scene::PostProcess& post = active.post;

    if (section("하드웨어")) {
        if (autoTune == gfx::AutoTune::OFF) {
            ImGui::Text("자동 튜닝: %s", gfx::autoTuneName(autoTune));
        } else {
            ImGui::Text("자동 튜닝: %s · 등급 %s", gfx::autoTuneName(autoTune), gfx::tierName(hardwareProfile.tier));
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("실행 인자 --auto-tune off|safe|aggressive 로 고른다");
        }
        // 왜 그렇게 골랐는지 그대로 보여 준다. 설정을 되돌리기 전에 근거를 알 수 있어야 한다.
        for (const std::string& reason : hardwareProfile.reasons) {
            ImGui::BulletText("%s", reason.c_str());
        }
        // 기동 시의 판정이라 그 뒤에 바뀐 것은 여기 나오지 않는다. 사용자가 고쳤거나, 가속 구조가
        // 예산을 넘어 렌더러가 스스로 광선 기능을 끈 경우가 그렇다.
        ImGui::TextDisabled("기동 시 판정이다. 지금 값은 아래 절들이 보여 준다");
    }

    if (section("후처리")) {
        ImGui::SliderFloat("노출", &renderer.settings.exposure, 0.05F, 8.0F, "%.2f");
        ImGui::SliderFloat("Bloom 세기", &post.bloomIntensity, 0.0F, 2.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("임계값을 넘은 부분을 흐려서 «더할» 세기");
        }
        ImGui::BeginDisabled(post.bloomIntensity <= 0.0F);
        ImGui::SliderFloat("Bloom 임계값", &post.bloomThreshold, 0.0F, 8.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("이 밝기를 넘는 곳만 번진다. 0 이면 화면 전체가 흐려진다");
        }
        ImGui::SliderFloat("Bloom 무릎", &post.bloomKnee, 0.0F, 1.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("임계값 언저리를 부드럽게 넘기는 폭. 0 이면 경계에서 깜빡인다");
        }
        ImGui::SliderFloat("Bloom 번짐", &post.bloomScatter, 0.0F, 1.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("클수록 넓은 밉 쪽에 무게를 두어 멀리 퍼진다");
        }
        ImGui::EndDisabled();
        ImGui::Checkbox("Auto Exposure", &post.autoExposure);
        ImGui::BeginDisabled(!post.autoExposure);
        ImGui::SliderFloat("적응 속도", &post.adaptationSpeed, 0.1F, 10.0F, "%.1f /s");
        ImGui::DragFloatRange2("EV 범위", &post.exposureMinEv, &post.exposureMaxEv, 0.1F, -10.0F, 20.0F, "%.1f");
        ImGui::EndDisabled();
    }

    if (section("높이 안개")) {
        ImGui::SliderFloat("안개 밀도", &post.fogDensity, 0.0F, 2.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::BeginDisabled(post.fogDensity <= 0.0F);
        ImGui::ColorEdit3("안개 색", glm::value_ptr(post.fogColor));
        ImGui::DragFloat("안개 높이", &post.fogHeight, 0.05F, -100.0F, 100.0F, "%.2f");
        ImGui::SliderFloat("높이 감쇠", &post.fogFalloff, 0.0F, 5.0F, "%.2f");
        ImGui::SliderFloat("태양 산란", &post.fogSunScatter, 0.0F, 2.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("첫 방향광 쪽을 볼수록 안개가 밝아진다(Henyey-Greenstein). 방향광이 없으면 효과가 없다");
        }
        ImGui::EndDisabled();
    }

    if (section("조명")) {
        ImGui::Text("장면 조명 %zu개", active.lights.size());
        ImGui::ColorEdit3("환경광", glm::value_ptr(active.ambientColor));
        ImGui::SliderFloat("환경광 세기", &active.ambientIntensity, 0.0F, 4.0F, "%.2f");
        ImGui::BeginDisabled(rasterOnly);
        ImGui::Checkbox("그림자", &renderer.settings.shadowsEnabled);
        ImGui::BeginDisabled(!renderer.settings.shadowsEnabled);
        ImGui::Checkbox("시점 Frustum Culling", &renderer.settings.shadowViewCulling);
        ImGui::SameLine();
        ImGui::Checkbox("Caster Culling", &renderer.settings.shadowCasterCulling);
        ImGui::Checkbox("시점 캐싱", &renderer.settings.shadowCaching);
        int cascades = static_cast<int>(renderer.settings.shadowCascades);
        if (ImGui::SliderInt("캐스케이드", &cascades, 1, static_cast<int>(gfx::MAX_SHADOW_CASCADES))) {
            renderer.settings.shadowCascades = static_cast<uint32_t>(cascades);
        }
        ImGui::SliderFloat("분할 혼합", &renderer.settings.shadowSplitLambda, 0.0F, 1.0F, "%.2f");
        ImGui::DragFloat("그림자 거리", &renderer.settings.shadowDistance, 1.0F, 0.0F, 10000.0F, "%.0f (0 이면 자동)");
        ImGui::Text("드로우 %u / %u, 다시 그린 층 %u",
                    renderer.shadowDrawCount(),
                    renderer.shadowDrawCandidates(),
                    renderer.shadowLayersDrawn());
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::TextDisabled("그림자 시점 %u개까지 (방향광/스폿광 1, 점광 6)", gfx::MAX_SHADOW_VIEWS);

        ImGui::BeginDisabled(!rayQueryReady);
        ImGui::Checkbox("Ray Traced Shadows (하이브리드)", &renderer.settings.useRayQueryShadows);
        ImGui::BeginDisabled(!renderer.settings.useRayQueryShadows);
        ImGui::SliderFloat("광선 거리", &renderer.settings.rayShadowDistance, 1.0F, 200.0F, "%.0f");
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing 중에는 래스터 패스를 건너뛰므로 적용되지 않는다");
        } else if (!rayQueryReady) {
            if (!renderer.rayTracingBlocked().empty()) {
                ImGui::TextWrapped("%s", renderer.rayTracingBlocked().c_str());
            } else {
                ImGui::TextDisabled("이 장치는 Ray Query를 지원하지 않는다");
            }
        } else {
            ImGui::TextDisabled("이 거리 안쪽만 광선으로 판정하고 바깥은 그림자 맵을 쓴다");
        }
    }

    if (section("Ray Traced Reflections")) {
        bool reflectionReady = rayQueryReady && renderer.settings.useIbl;
        ImGui::BeginDisabled(!reflectionReady);
        ImGui::Checkbox("Ray Traced Reflections", &renderer.settings.useReflections);
        ImGui::BeginDisabled(!renderer.settings.useReflections);
        ImGui::SliderFloat("거칠기 상한", &renderer.settings.reflectionRoughnessCutoff, 0.03F, 1.0F, "%.2f");
        ImGui::SliderFloat("반사 세기", &renderer.settings.reflectionIntensity, 0.0F, 2.0F, "%.2f");
        int reflectionSamples = static_cast<int>(renderer.settings.reflectionMaxSamples);
        if (ImGui::SliderInt("누적 상한", &reflectionSamples, 1, 64)) {
            renderer.settings.reflectionMaxSamples = static_cast<uint32_t>(reflectionSamples);
        }
        ImGui::Checkbox("Denoiser", &renderer.settings.reflectionDenoise);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "깊이·노멀로 히스토리를 검증하고, 매끈한 면은 히트 거리의 가상점으로 되짚고, 휘도 분산으로 "
                "가중한 à-trous 필터를 세 번 돈다. 끄면 누적만 한다");
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing이 반사를 직접 계산한다");
        } else if (!rayQueryReady) {
            ImGui::TextDisabled("이 장치는 Ray Query를 지원하지 않는다");
        } else if (!renderer.settings.useIbl) {
            ImGui::TextDisabled("IBL 이 꺼져 있으면 스페큘러 항이 없어 반사도 쉰다");
        } else {
            ImGui::TextDisabled("거칠기 상한 이하의 불투명 표면만 추적한다. 픽셀당 광선 하나, Temporal 누적");
        }
    }

    if (section("ReSTIR Direct Lighting")) {
        ImGui::BeginDisabled(!rayQueryReady);
        ImGui::Checkbox("ReSTIR Direct Lighting", &renderer.settings.useRestir);
        ImGui::BeginDisabled(!renderer.settings.useRestir);
        int candidates = static_cast<int>(renderer.settings.restirCandidates);
        if (ImGui::SliderInt("광원 후보 수", &candidates, 1, 32)) {
            renderer.settings.restirCandidates = static_cast<uint32_t>(candidates);
        }
        ImGui::Checkbox("Temporal 재사용", &renderer.settings.restirTemporal);
        ImGui::SameLine();
        ImGui::Checkbox("Spatial 재사용", &renderer.settings.restirSpatial);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing이 직접광을 직접 계산한다");
        } else if (!rayQueryReady) {
            ImGui::TextDisabled("이 장치는 Ray Query를 지원하지 않는다");
        } else {
            ImGui::TextDisabled(
                "불투명 표면의 직접광을 광원 후보 재추출과 시간·공간 재사용으로 고른 광원 하나에 광선 하나로 "
                "계산한다. 광원 수와 무관하게 픽셀당 광선 하나");
        }
    }

    if (section("DDGI")) {
        ImGui::BeginDisabled(!rayQueryReady);
        ImGui::Checkbox("DDGI", &renderer.settings.useDdgi);
        ImGui::BeginDisabled(!renderer.settings.useDdgi);
        int probes = static_cast<int>(renderer.settings.ddgiProbes);
        if (ImGui::SliderInt("축당 프로브", &probes, 2, 16)) {
            renderer.settings.ddgiProbes = static_cast<uint32_t>(probes);
        }
        int rays = static_cast<int>(renderer.settings.ddgiRays);
        if (ImGui::SliderInt("프로브당 광선", &rays, 8, 256)) {
            renderer.settings.ddgiRays = static_cast<uint32_t>(rays);
        }
        ImGui::SliderFloat("히스테리시스", &renderer.settings.ddgiHysteresis, 0.5F, 0.99F, "%.2f");
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing이 간접광을 직접 계산한다");
        } else if (!rayQueryReady) {
            ImGui::TextDisabled("이 장치는 Ray Query를 지원하지 않는다");
        } else {
            ImGui::TextDisabled(
                "장면 경계 상자의 프로브 격자가 프레임마다 광선을 쏘아 조도·가시성 아틀라스를 갱신한다. "
                "확산 조도가 IBL 큐브 대신 이것을 읽는다. 디버그 뷰 «Probe Irradiance»");
        }
    }

    if (section("환경 (IBL)")) {
        scene::Environment& env = active.environment;
        ImGui::Checkbox("IBL 사용", &renderer.settings.useIbl);
        int skySource = env.useHdr ? 1 : 0;
        if (ImGui::Combo("하늘", &skySource, "절차적\0HDR 파일\0")) {
            env.useHdr = skySource == 1;
        }
        if (env.useHdr) {
            ImGui::TextDisabled("%s", env.hdrPath.empty() ? "파일 없음" : env.hdrPath.filename().string().c_str());
            if (ImGui::Button("HDR 파일 고르기")) {
                hdrFiles.clear();
                std::error_code error;
                for (const auto& entry : std::filesystem::directory_iterator(modelRoot, error)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".hdr") {
                        hdrFiles.push_back(entry.path());
                    }
                }
                std::ranges::sort(hdrFiles);
                ImGui::OpenPopup("HDR 선택");
            }
            if (ImGui::BeginPopup("HDR 선택")) {
                if (hdrFiles.empty()) {
                    ImGui::TextDisabled("%s 에 .hdr 파일이 없습니다", modelRoot.string().c_str());
                }
                for (const std::filesystem::path& file : hdrFiles) {
                    if (ImGui::Selectable(file.filename().string().c_str())) {
                        env.hdrPath = file;
                    }
                }
                ImGui::Separator();
                ImGui::SetNextItemWidth(320.0F);
                ImGui::InputText("경로", hdrPathInput.data(), hdrPathInput.size());
                ImGui::SameLine();
                if (ImGui::Button("열기") && hdrPathInput[0] != '\0') {
                    env.hdrPath = std::filesystem::path(hdrPathInput.data());
                }
                ImGui::EndPopup();
            }
        } else {
            ImGui::ColorEdit3("천정", glm::value_ptr(env.zenithColor));
            ImGui::ColorEdit3("지평", glm::value_ptr(env.horizonColor));
            ImGui::ColorEdit3("지면", glm::value_ptr(env.groundColor));
            ImGui::ColorEdit3("태양색", glm::value_ptr(env.sunColor));
            ImGui::SliderFloat("태양 세기", &env.sunIntensity, 0.0F, 8.0F, "%.2f");
        }
        ImGui::SliderFloat("환경 세기", &env.intensity, 0.0F, 4.0F, "%.2f");
        ImGui::SliderFloat("환경 회전", &env.yawDegrees, -180.0F, 180.0F, "%.0f°");
        if (ImGui::Button("다시 굽기")) {
            renderer.invalidateEnvironment();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("태양 방향은 첫 방향광을 따라간다");
    }

    if (section("SSAO")) {
        ImGui::BeginDisabled(rasterOnly);
        ImGui::Checkbox("사용", &renderer.settings.useSsao);
        ImGui::BeginDisabled(!renderer.settings.useSsao);
        ImGui::SliderFloat("반지름", &renderer.settings.ssaoRadius, 0.005F, 0.3F, "장면의 %.3f배");
        ImGui::SliderFloat("세기", &renderer.settings.ssaoIntensity, 0.0F, 3.0F, "%.2f");
        ImGui::SliderFloat("편향", &renderer.settings.ssaoBias, 0.0F, 0.02F, "%.4f");
        int samples = static_cast<int>(renderer.settings.ssaoSamples);
        if (ImGui::SliderInt("표본", &samples, 4, 64)) {
            renderer.settings.ssaoSamples = static_cast<uint32_t>(samples);
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing 중에는 래스터 패스를 건너뛰므로 적용되지 않는다");
        }

        // 컬 컴퓨트도 태스크 셰이더도 래스터 분기 안에서만 돈다. Path Tracing의 가속 구조는 0단계 LOD
        // 삼각형으로 세우므로 meshlet 컬링도 LOD 선정도 관여하지 않는다.
    }

    if (section("컬링과 LOD")) {
        ImGui::BeginDisabled(rasterOnly);
        ImGui::Checkbox("Compute Culling", &renderer.settings.useComputeCulling);
        // 오클루전은 Mesh Shader 경로의 태스크 셰이더에도 적용되므로 Compute Culling 잠금 밖에 둔다.
        ImGui::Checkbox("HZB Occlusion Culling (두 패스)", &renderer.settings.occlusionCulling);
        ImGui::BeginDisabled(!renderer.settings.useComputeCulling);
        ImGui::Checkbox("Frustum Culling", &renderer.settings.frustumCulling);
        ImGui::SameLine();
        ImGui::Checkbox("Normal Cone Culling", &renderer.settings.coneCulling);
        ImGui::EndDisabled();

        ImGui::Checkbox("자동 LOD 선정", &renderer.settings.automaticLod);
        if (renderer.settings.automaticLod) {
            ImGui::SliderFloat("허용 화면 오차", &renderer.settings.lodErrorThreshold, 0.1F, 32.0F, "%.2f px");

            ImGui::Checkbox("Neural LOD", &renderer.settings.useNeuralLod);
            if (renderer.settings.useNeuralLod) {
                ImGui::Checkbox("학습", &renderer.settings.trainLodNetwork);
                ImGui::SameLine();
                if (ImGui::Button("가중치 초기화")) {
                    renderer.lodNetwork.reset();
                }
                ImGui::SliderFloat("삼각형 예산",
                                   &renderer.settings.triangleBudget,
                                   1000.0F,
                                   500000.0F,
                                   "%.0f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat(
                    "학습률", &renderer.lodNetwork.learningRate, 0.001F, 0.5F, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::Text("손실 %.5f, 기대 삼각형 %.0f",
                            static_cast<double>(renderer.lodNetwork.lastLoss()),
                            static_cast<double>(renderer.lodNetwork.lastSoftTriangleCount()));
            }
        } else {
            int lodLevel = static_cast<int>(renderer.settings.lodLevel);
            int maxLod = std::max(static_cast<int>(geometryStore != nullptr ? geometryStore->maxLodCount() : 1) - 1, 0);
            if (ImGui::SliderInt("LOD 단계", &lodLevel, 0, std::max(maxLod, 0))) {
                renderer.settings.lodLevel = static_cast<uint32_t>(lodLevel);
            }
        }

        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing 중에는 래스터 패스를 건너뛰므로 적용되지 않는다");
        }
    }

    if (section("해상도와 Upscaling")) {
        if (ImGui::SliderFloat("렌더 배율", &renderer.settings.renderScale, 0.25F, 2.0F, "%.2f")) {
            // 배율은 다음 프레임의 표시 크기 갱신에서 반영된다.
        }
        ImGui::Text("장면 %ux%u -> 표시 %ux%u",
                    renderer.renderExtent().width,
                    renderer.renderExtent().height,
                    renderer.displayExtent().width,
                    renderer.displayExtent().height);

        std::vector<gfx::UpscalerInfo> upscalers = renderer.upscalers();
        bool dlssSelected =
            renderer.settings.upscaler == gfx::Upscaler::DLSS || renderer.settings.upscaler == gfx::Upscaler::DLSS_RR;
        for (const gfx::UpscalerInfo& info : upscalers) {
            // Ray Reconstruction 은 DLSS 의 한 모드다. 목록에 따로 두면 초해상과 무관한 별개 기법처럼
            // 보이고, Path Tracing을 켜야 한다는 조건도 드러나지 않는다. 아래에서 체크박스로 다룬다.
            if (info.kind == gfx::Upscaler::DLSS_RR) {
                continue;
            }
            bool selected = info.kind == gfx::Upscaler::DLSS ? dlssSelected : renderer.settings.upscaler == info.kind;
            ImGui::BeginDisabled(!info.available);
            if (ImGui::RadioButton(info.name, selected)) {
                renderer.settings.upscaler = info.kind;
            }
            ImGui::EndDisabled();
            if (!info.available) {
                ImGui::SameLine();
                ImGui::TextDisabled("(%s)", info.reason);
            }
        }
        if (renderer.settings.upscaler == gfx::Upscaler::SPATIAL) {
            ImGui::SliderFloat("Sharpening", &renderer.settings.upscaleSharpness, 0.0F, 1.0F, "%.2f");
        }
        if (dlssSelected) {
            ImGui::Indent();

            // 사전 설정은 곧 렌더 배율이다. NGX 에 넘기는 값도 배율에서 되돌리므로 둘이 어긋날 수 없다.
            gfx::DlssQuality quality = gfx::dlssQualityForScale(renderer.settings.renderScale);
            if (ImGui::BeginCombo("품질", gfx::dlssQualityName(quality))) {
                for (uint32_t index = 0; index < gfx::DLSS_QUALITY_COUNT; ++index) {
                    auto candidate = static_cast<gfx::DlssQuality>(index);
                    if (ImGui::Selectable(gfx::dlssQualityName(candidate), candidate == quality)) {
                        renderer.settings.renderScale = gfx::dlssQualityScale(candidate);
                    }
                }
                ImGui::EndCombo();
            }

            gfx::UpscalerInfo reconstruction{};
            for (const gfx::UpscalerInfo& info : upscalers) {
                if (info.kind == gfx::Upscaler::DLSS_RR) {
                    reconstruction = info;
                }
            }
            bool useReconstruction = renderer.settings.upscaler == gfx::Upscaler::DLSS_RR;
            ImGui::BeginDisabled(!reconstruction.available);
            if (ImGui::Checkbox("Ray Reconstruction", &useReconstruction)) {
                renderer.settings.upscaler = useReconstruction ? gfx::Upscaler::DLSS_RR : gfx::Upscaler::DLSS;
            }
            ImGui::EndDisabled();
            if (!reconstruction.available) {
                ImGui::SameLine();
                ImGui::TextDisabled("(%s)", reconstruction.reason);
            } else if (useReconstruction && !renderer.settings.usePathTracing) {
                ImGui::TextDisabled("Path Tracing을 켜야 동작한다. 그전까지는 Super Resolution 으로 돌아간다");
            } else if (useReconstruction) {
                ImGui::TextDisabled("Path Tracing 1표본을 Denoise 하면서 확대한다");
            }

            ImGui::Unindent();
        }
    }

    if (section("Path Tracing")) {
        ImGui::BeginDisabled(!renderer.pathTracingAvailable());
        ImGui::Checkbox("Path Tracing", &renderer.settings.usePathTracing);
        ImGui::EndDisabled();
        if (!renderer.pathTracingAvailable()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(미지원)");
            if (!renderer.rayTracingBlocked().empty()) {
                ImGui::TextWrapped("%s", renderer.rayTracingBlocked().c_str());
            }
        } else if (renderer.settings.usePathTracing) {
            gfx::PathTraceOptions& options = renderer.settings.pathTrace;
            int bounces = static_cast<int>(options.maxBounces);
            if (ImGui::SliderInt("반사 횟수", &bounces, 1, 16)) {
                options.maxBounces = static_cast<uint32_t>(bounces);
            }
            int perFrame = static_cast<int>(options.samplesPerFrame);
            if (ImGui::SliderInt("프레임당 표본", &perFrame, 1, 16)) {
                options.samplesPerFrame = static_cast<uint32_t>(perFrame);
            }
            int lightCandidates = static_cast<int>(options.lightCandidates);
            if (ImGui::SliderInt("광원 후보 수", &lightCandidates, 1, 32)) {
                options.lightCandidates = static_cast<uint32_t>(lightCandidates);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(
                    "다음 사건 추정이 재추출(RIS)할 광원 후보 수. 광원이 많을수록 잡음이 준다. 1 이면 균등 선택");
            }
            int maxSamples = static_cast<int>(options.maxSamples);
            if (ImGui::SliderInt("표본 상한", &maxSamples, 0, 4096, maxSamples == 0 ? "무제한" : "%d")) {
                options.maxSamples = static_cast<uint32_t>(maxSamples);
            }
            ImGui::Checkbox("다음 사건 추정", &options.nextEventEstimation);
            ImGui::SameLine();
            ImGui::Checkbox("Russian Roulette", &options.russianRoulette);
            ImGui::SliderFloat("복사휘도 상한", &options.radianceClamp, 1.0F, 64.0F, "%.1f");
            ImGui::SliderFloat("하늘 밝기", &options.skyIntensity, 0.0F, 4.0F, "%.2f");
            ImGui::Text("누적 표본 %u", renderer.pathTraceSamples());
            ImGui::SameLine();
            if (ImGui::Button("누적 초기화")) {
                renderer.resetPathAccumulation();
            }
        }
    }

    if (section("파이프라인")) {
        ImGui::Checkbox("Wireframe", &renderer.settings.wireframe);
        ImGui::BeginDisabled(!renderer.meshShaderAvailable() || rasterOnly);
        ImGui::Checkbox("Mesh Shader 경로", &renderer.settings.useMeshShader);
        ImGui::EndDisabled();
        if (!renderer.meshShaderAvailable()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(미지원)");
        } else if (rasterOnly) {
            // 광선 순회는 가속 구조를 타지 mesh 셰이더를 실행하지 않는다. 둘은 아예 다른 파이프라인이다.
            ImGui::TextDisabled("Path Tracing은 래스터 파이프라인을 타지 않는다");
        }

        static constexpr const char* DEBUG_MODE_NAMES[] = {"Shading",
                                                           "Meshlet",
                                                           "Normal",
                                                           "UV",
                                                           "Depth",
                                                           "LOD",
                                                           "Shadow Cascade",
                                                           "Shadow",
                                                           "Motion Vector",
                                                           "Cull Pass",
                                                           "Reflection Raw",
                                                           "Reflection Accumulated",
                                                           "Reflection Filtered",
                                                           "ReSTIR Light",
                                                           "Probe Irradiance"};
        // Path Tracing이나 이 장치가 못 만드는 값은 개별로 잠그고 사유를 보인다.
        if (ImGui::BeginCombo("디버그 뷰", DEBUG_MODE_NAMES[renderer.settings.debugMode])) {
            for (uint32_t mode = 0; mode < IM_ARRAYSIZE(DEBUG_MODE_NAMES); ++mode) {
                const char* blocked = renderer.debugModeBlockedReason(mode);
                ImGui::BeginDisabled(blocked != nullptr);
                if (ImGui::Selectable(DEBUG_MODE_NAMES[mode], renderer.settings.debugMode == mode)) {
                    renderer.settings.debugMode = mode;
                }
                ImGui::EndDisabled();
                if (blocked != nullptr && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("%s", blocked);
                }
            }
            ImGui::EndCombo();
        }
        if (const char* blocked = renderer.debugModeBlockedReason(renderer.settings.debugMode); blocked != nullptr) {
            ImGui::TextDisabled("%s", blocked);
        }

        bool vsync = renderer.vsyncEnabled();
        if (ImGui::Checkbox("수직 동기화", &vsync)) {
            renderer.setVsync(vsync);
        }

        ImGui::Separator();
        ImGui::TextDisabled("하드웨어 기능");
        const gfx::Capabilities& caps = context.caps;
        ImGui::BeginDisabled();
        bool meshShader = caps.meshShader;
        bool rayTracing = caps.rayTracingPipeline;
        bool drawIndirectCount = caps.drawIndirectCount;
        bool drawIndex = caps.shaderDrawIndex;
        ImGui::Checkbox("mesh shader", &meshShader);
        ImGui::Checkbox("Ray Tracing Pipeline", &rayTracing);
        ImGui::Checkbox("drawIndirectCount", &drawIndirectCount);
        ImGui::Checkbox("gl_DrawID (없으면 meshlet 디버그 뷰가 메쉬 단위)", &drawIndex);
        ImGui::EndDisabled();
    }
    // 플러그인의 절(물리·유체·프로파일러·콜라이더 표시). 렌더러 필드만 만지는 절은 위에 남아 있다.
    if (pluginSettings) {
        pluginSettings();
    }
    ImGui::End();
}

bool Editor::settingsSection(const char* name) {
    if (settingsFilter[0] != '\0') {
        if (!containsNoCase(name, settingsFilter.data())) {
            return false;
        }
        ImGui::SeparatorText(name);
        return true;
    }
    // 헤더 이름이 안의 체크박스 이름("Ray Traced Reflections", "Path Tracing")과 같으면 ID 가 겹쳐 체크박스가
    // 눌리지 않는다. 숨은 접미사로 헤더 ID 를 따로 둔다.
    std::string header = std::string(name) + "##section";
    return ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
}

} // namespace editor
