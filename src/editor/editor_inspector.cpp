// 인스펙터 패널(변환·부품 절·부품 추가).
// Editor 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 editor.h 하나에 있다.

#include "editor/editor_internal.h"

namespace editor {

namespace {

const char* alphaModeName(asset::AlphaMode mode) {
    switch (mode) {
    case asset::AlphaMode::CUTOFF:
        return "컷오프";
    case asset::AlphaMode::TRANSLUCENT:
        return "반투명";
    default:
        return "불투명";
    }
}

} // namespace

void Editor::buildInspector(scene::Scene& active, const gfx::GeometryStore& geometry) {
    if (!ImGui::Begin(WINDOW_INSPECTOR)) {
        ImGui::End();
        return;
    }

    int selectedObject = primarySelection();
    if (selectedObject < 0 || selectedObject >= static_cast<int>(active.objects.size())) {
        ImGui::TextDisabled("선택된 오브젝트가 없습니다");
        ImGui::End();
        return;
    }

    scene::Object& object = active.objects[static_cast<size_t>(selectedObject)];
    auto objectIndex = static_cast<uint32_t>(selectedObject);
    {
        std::array<char, 256> nameInput{};
        std::copy_n(object.name.c_str(), std::min(object.name.size() + 1, nameInput.size() - 1), nameInput.begin());
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::InputText("##name", nameInput.data(), nameInput.size())) {
            object.name = nameInput.data();
        }
    }
    ImGui::Separator();

    // 부품 헤더. 오른쪽 끝에 «제거» 단추를 겹쳐 둔다. 떼는 일은 배열을 압축하므로 패널을 다 그린 뒤 한다.
    auto componentHeader = [this, &active, objectIndex](const char* label, int32_t scene::Object::* handle) {
        ImGui::PushID(label);
        bool open = ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap);
        float buttonWidth = ImGui::CalcTextSize("제거").x + ImGui::GetStyle().FramePadding.x * 2.0F;
        ImGui::SameLine(ImGui::GetContentRegionMax().x - buttonWidth);
        if (ImGui::SmallButton("제거")) {
            deferred = [&active, objectIndex, handle] { active.detachComponent(objectIndex, handle); };
        }
        ImGui::PopID();
        return open;
    };

    if (ImGui::CollapsingHeader("변환", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("위치", glm::value_ptr(object.transform.position), 0.01F);
        glm::vec3 euler = glm::degrees(glm::eulerAngles(object.transform.rotation));
        if (ImGui::DragFloat3("회전", glm::value_ptr(euler), 0.5F)) {
            object.transform.rotation = glm::quat(glm::radians(euler));
        }
        ImGui::DragFloat3("크기", glm::value_ptr(object.transform.scale), 0.01F, 0.001F, 1000.0F);
        if (object.parent >= 0) {
            ImGui::TextDisabled("부모: %s", active.objects[static_cast<size_t>(object.parent)].name.c_str());
        }
    }

    // 조명 속성. 위치와 방향은 변환에서 오므로 여기서는 나머지만 다룬다.
    if (object.light >= 0 && object.light < static_cast<int>(active.lights.size()) &&
        componentHeader("조명", &scene::Object::light)) {
        scene::Light& light = active.lights[static_cast<size_t>(object.light)];
        constexpr std::array<const char*, 4> LIGHT_NAMES{"방향광", "점광", "스폿광", "영역광"};
        auto typeIndex = static_cast<int>(light.type);
        if (ImGui::Combo("종류", &typeIndex, LIGHT_NAMES.data(), static_cast<int>(LIGHT_NAMES.size()))) {
            light.type = static_cast<scene::LightType>(typeIndex);
        }
        ImGui::ColorEdit3("색", glm::value_ptr(light.color));
        ImGui::DragFloat("세기", &light.intensity, 0.05F, 0.0F, 1000.0F);
        if (light.type != scene::LightType::DIRECTIONAL) {
            ImGui::DragFloat("거리", &light.range, 0.1F, 0.01F, 10000.0F);
        }
        if (light.type == scene::LightType::SPOT) {
            ImGui::DragFloat("안쪽 각", &light.innerConeDegrees, 0.5F, 0.0F, 89.0F);
            ImGui::DragFloat("바깥 각", &light.outerConeDegrees, 0.5F, 0.0F, 89.0F);
            light.innerConeDegrees = std::min(light.innerConeDegrees, light.outerConeDegrees);
        }
        if (light.type == scene::LightType::AREA) {
            ImGui::DragFloat2("크기", glm::value_ptr(light.size), 0.05F, 0.01F, 1000.0F);
            ImGui::TextDisabled("영역광은 그림자를 만들지 않습니다");
        } else {
            ImGui::Checkbox("그림자", &light.castsShadow);
        }
    }

    // 애니메이션 컨트롤러. 스켈레톤을 가진 오브젝트에만 나온다.
    if (object.animator >= 0 && object.animator < static_cast<int>(active.animators.size())) {
        scene::Animator& animator = active.animators[static_cast<size_t>(object.animator)];
        if (ImGui::CollapsingHeader("애니메이션", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (animator.skeleton.animations.empty()) {
                ImGui::TextDisabled("클립이 없는 스켈레톤입니다");
            } else {
                animator.clip = std::min(animator.clip, static_cast<uint32_t>(animator.skeleton.animations.size()) - 1);
                const asset::Animation& clip = animator.skeleton.animations[animator.clip];
                if (ImGui::BeginCombo("클립", clip.name.c_str())) {
                    for (uint32_t i = 0; i < animator.skeleton.animations.size(); ++i) {
                        if (ImGui::Selectable(animator.skeleton.animations[i].name.c_str(), i == animator.clip)) {
                            animator.clip = i;
                            animator.clipTime = 0.0F;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::Checkbox("재생", &animator.playing);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90.0F);
                ImGui::DragFloat("속도", &animator.speed, 0.01F, -4.0F, 4.0F);
                ImGui::SliderFloat("시간", &animator.clipTime, 0.0F, std::max(clip.duration, 0.001F), "%.2f s");
            }
            ImGui::Text("조인트 %zu, 스킨 %zu", animator.skeleton.nodes.size(), animator.skeleton.skins.size());
        }
    }

    // Mesh Renderer. 메쉬는 바꿀 수 있고 재질은 메쉬에 딸려 온다.
    if (object.meshRenderer >= 0 && componentHeader("Mesh Renderer", &scene::Object::meshRenderer)) {
        uint32_t selectedMesh = active.meshOf(objectIndex);
        bool live = geometry.meshLive(selectedMesh);
        std::string current = live ? std::to_string(selectedMesh) + ": " + geometry.meshName(selectedMesh) : "(없음)";
        if (ImGui::BeginCombo("메쉬", current.c_str())) {
            for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
                if (!geometry.meshLive(meshIndex)) {
                    continue;
                }
                std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex);
                if (ImGui::Selectable(label.c_str(), meshIndex == selectedMesh)) {
                    active.attachMeshRenderer(objectIndex, meshIndex, active.skinOf(objectIndex));
                }
            }
            ImGui::EndCombo();
        }
        if (live) {
            const gfx::GpuMesh& mesh = geometry.mesh(selectedMesh);
            ImGui::Text("삼각형 %u", mesh.indexCount / 3);
            ImGui::Text("바운딩 반지름 %.3f", mesh.boundingSphere.w);

            const asset::Material& material = geometry.material(mesh.materialIndex);
            ImGui::Text("재질: %s", material.name.c_str());
            ImGui::Text("알파 경로: %s%s", alphaModeName(material.alphaMode), material.doubleSided ? " (양면)" : "");
            ImGui::Text("금속성 %.2f / 거칠기 %.2f", material.metallicFactor, material.roughnessFactor);
            ImGui::ColorButton("기저 색",
                               ImVec4{material.baseColorFactor.r,
                                      material.baseColorFactor.g,
                                      material.baseColorFactor.b,
                                      material.baseColorFactor.a});
        } else {
            ImGui::TextDisabled("메쉬가 해제되었거나 아직 올라오지 않았다");
        }
    }

    if (object.rigidBody >= 0 && object.rigidBody < static_cast<int>(active.rigidBodies.size()) &&
        componentHeader("강체", &scene::Object::rigidBody)) {
        scene::RigidBody& body = active.rigidBodies[static_cast<size_t>(object.rigidBody)];
        constexpr std::array<const char*, scene::COLLIDER_SHAPE_COUNT> SHAPE_NAMES{
            "구", "상자", "평면", "원기둥", "캡슐", "메쉬"};
        auto shapeIndex = static_cast<int>(body.shape);
        if (ImGui::Combo("모양", &shapeIndex, SHAPE_NAMES.data(), static_cast<int>(SHAPE_NAMES.size()))) {
            body.shape = static_cast<scene::ColliderShape>(shapeIndex);
        }
        bool alwaysKinematic = body.shape == scene::ColliderShape::PLANE || body.shape == scene::ColliderShape::MESH;
        switch (body.shape) {
        case scene::ColliderShape::SPHERE:
            ImGui::DragFloat("반지름", &body.radius, 0.01F, 0.01F, 100.0F);
            break;
        case scene::ColliderShape::BOX:
            ImGui::DragFloat3("반쪽 크기", glm::value_ptr(body.halfExtents), 0.01F, 0.01F, 100.0F);
            break;
        case scene::ColliderShape::CYLINDER:
        case scene::ColliderShape::CAPSULE:
            ImGui::DragFloat("반지름", &body.radius, 0.01F, 0.01F, 100.0F);
            ImGui::DragFloat("반높이", &body.halfExtents.y, 0.01F, 0.01F, 100.0F);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(body.shape == scene::ColliderShape::CAPSULE
                                      ? "축은 오브젝트의 +Y. 반구를 뺀 몸통의 반높이다"
                                      : "축은 오브젝트의 +Y");
            }
            break;
        case scene::ColliderShape::MESH:
            if (active.colliderMesh(objectIndex) != nullptr) {
                ImGui::TextDisabled("Mesh Renderer 의 메쉬(삼각형 %zu개, 굵은 LOD)를 그대로 쓴다. 늘 운동학이다",
                                    active.colliderMesh(objectIndex)->indices.size() / 3);
            } else {
                ImGui::TextDisabled("Mesh Renderer 가 없어 부딪히지 않는다. 늘 운동학이다");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("바닥·지형용. 움직이는 물체가 이 삼각형에 닿는다. 유체는 통과한다");
            }
            break;
        case scene::ColliderShape::PLANE:
            ImGui::TextDisabled("오브젝트의 +Y 가 법선인 무한 평면. 늘 운동학이다");
            break;
        }
        ImGui::BeginDisabled(alwaysKinematic);
        ImGui::Checkbox("운동학", &body.kinematic);
        ImGui::SameLine();
        ImGui::Checkbox("중력", &body.useGravity);
        ImGui::DragFloat("질량", &body.mass, 0.1F, 0.01F, 10000.0F);
        ImGui::EndDisabled();
        ImGui::SliderFloat("반발", &body.restitution, 0.0F, 1.0F, "%.2f");
        ImGui::SliderFloat("마찰", &body.friction, 0.0F, 1.0F, "%.2f");

        constexpr std::array<const char*, 3> RIGID_BACKENDS{"자동", "CPU", "GPU"};
        // 컴퓨트 파이프라인을 만들지 못한 장치에서는 GPU 를 고를 수 없다.
        bool rigidGpuUsable = rigidStatus.gpuAvailable;
        if (ImGui::BeginCombo("백엔드", RIGID_BACKENDS[static_cast<size_t>(body.backend)])) {
            for (uint32_t i = 0; i < RIGID_BACKENDS.size(); ++i) {
                bool usable = rigidGpuUsable || i != static_cast<uint32_t>(scene::SimulationBackend::GPU);
                ImGui::BeginDisabled(!usable);
                if (ImGui::Selectable(RIGID_BACKENDS[i], static_cast<uint32_t>(body.backend) == i)) {
                    body.backend = static_cast<scene::SimulationBackend>(i);
                }
                ImGui::EndDisabled();
                if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("이 장치에서는 강체 컴퓨트 파이프라인을 만들지 못했다");
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("자동은 CPU 다. GPU 는 접촉을 Jacobi 로 풀어 CPU 와 수치가 다르고,\n"
                              "결과가 몇 프레임 늦게 반영된다.\n"
                              "두 백엔드는 서로 부딪히지 않으므로 한 장면에서 섞지 않는다");
        }
        if (body.backend == scene::SimulationBackend::GPU) {
            ImGui::TextDisabled("Jacobi %u 회, GPU 강체 %u 개. 쌓인 물체가 CPU 보다 물렁하다",
                                gfx::RIGID_SOLVER_ITERATIONS,
                                rigidStatus.gpuBodies);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("CPU 백엔드 강체와는 서로 부딪히지 않는다. 바닥도 GPU 로 맞춰야 한다");
            }
        }
        if (active.simulating) {
            ImGui::Text("속도 (%.2f, %.2f, %.2f)", body.velocity.x, body.velocity.y, body.velocity.z);
        }
    }

    if (object.fluid >= 0 && object.fluid < static_cast<int>(active.fluids.size()) &&
        componentHeader("유체", &scene::Object::fluid)) {
        scene::Fluid& fluid = active.fluids[static_cast<size_t>(object.fluid)];
        constexpr std::array<const char*, 3> BACKEND_NAMES{"자동", "CPU", "GPU"};
        // 컴퓨트 파이프라인을 만들지 못한 장치에서는 GPU 를 고를 수 없다. 하드웨어 게이트 규약대로
        // 항목을 잠그고 사유를 보여 준다.
        bool gpuUsable = renderer.fluidGpuAvailable();
        if (ImGui::BeginCombo("백엔드", BACKEND_NAMES[static_cast<size_t>(fluid.backend)])) {
            for (uint32_t i = 0; i < BACKEND_NAMES.size(); ++i) {
                bool usable = gpuUsable || i != static_cast<uint32_t>(scene::SimulationBackend::GPU);
                ImGui::BeginDisabled(!usable);
                if (ImGui::Selectable(BACKEND_NAMES[i], static_cast<uint32_t>(fluid.backend) == i)) {
                    fluid.backend = static_cast<scene::SimulationBackend>(i);
                }
                ImGui::EndDisabled();
                if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("이 장치에서는 유체 컴퓨트 파이프라인을 만들지 못했다");
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("자동은 GPU 를 쓰되 만들지 못하면 CPU 로 내려간다");
        }
        // 자동이 무엇을 골랐는지는 렌더러만 안다. 그것을 그대로 보여 준다.
        ImGui::SameLine();
        bool onCpu = renderer.fluidOnCpu(static_cast<uint32_t>(object.fluid));
        ImGui::TextDisabled("(지금 %s)", onCpu ? "CPU" : "GPU");

        int particles = static_cast<int>(fluid.particleCount);
        if (ImGui::SliderInt("입자 수", &particles, 64, gfx::FLUID_MAX_PARTICLES)) {
            fluid.particleCount = static_cast<uint32_t>(particles);
        }
        // 하드웨어 프로파일이 상한을 낮췄으면 슬라이더 값보다 적게 뿌린다. 그 사실을 여기서 말한다.
        if (fluid.particleCount > renderer.settings.fluidParticleLimit) {
            ImGui::SameLine();
            ImGui::TextDisabled("(상한 %u)", renderer.settings.fluidParticleLimit);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("자동 튜닝이 이 기기의 상한을 정했다. --auto-tune off 로 풀 수 있다");
            }
        }
        constexpr std::array<const char*, 2> DISPLAY_NAMES{"입자", "표면"};
        bool surfaceUsable = renderer.fluidSurfaceAvailable() || onCpu;
        ImGui::BeginDisabled(!surfaceUsable);
        auto displayIndex = static_cast<int>(fluid.display);
        if (ImGui::Combo("표시", &displayIndex, DISPLAY_NAMES.data(), static_cast<int>(DISPLAY_NAMES.size()))) {
            fluid.display = static_cast<scene::FluidDisplay>(displayIndex);
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip(surfaceUsable
                                  ? "표면은 마칭 큐브로 등치면을 뽑아 물처럼 그린다.\n"
                                    "Path Tracing 은 이 표면을 굴절·흡수로 추적하고 Ray Traced Reflections 는\n"
                                    "«반사 + 물빛» 으로 본다. 그림자는 받지만 던지지는 않는다"
                                  : "이 장치에서는 표면 컴퓨트를 만들지 못했다. CPU 백엔드로는 쓸 수 있다");
        }
        if (fluid.display == scene::FluidDisplay::SURFACE) {
            uint32_t ceiling = onCpu ? gfx::FLUID_MAX_CPU_SURFACE_RESOLUTION : gfx::FLUID_MAX_SURFACE_RESOLUTION;
            auto resolution = static_cast<int>(fluid.surfaceResolution);
            if (ImGui::SliderInt("격자 해상도", &resolution, 8, static_cast<int>(gfx::FLUID_MAX_SURFACE_RESOLUTION))) {
                fluid.surfaceResolution = static_cast<uint32_t>(resolution);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("축마다의 셀 수다. 표본 수는 세제곱으로 는다");
            }
            // CPU 백엔드는 표본마다 입자 전부를 훑으므로 렌더러가 더 낮게 묶는다. 그 사실을 말해 준다.
            if (fluid.surfaceResolution > ceiling) {
                ImGui::SameLine();
                ImGui::TextDisabled("(CPU 상한 %u)", ceiling);
            }
            ImGui::SliderFloat("등치값", &fluid.surfaceIso, 0.05F, 4.0F, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("작으면 표면이 부풀고 크면 물이 얇아져 구멍이 뚫린다");
            }
            ImGui::ColorEdit3("물 색", glm::value_ptr(fluid.waterColor));
            ImGui::SliderFloat("표면 거칠기", &fluid.surfaceRoughness, 0.01F, 0.5F, "%.3f");
            ImGui::DragFloat3("흡수 계수", glm::value_ptr(fluid.absorption), 0.05F, 0.0F, 20.0F);
            ImGui::DragFloat("두께 배율", &fluid.thicknessScale, 0.05F, 0.05F, 20.0F);
        }
        ImGui::DragFloat3("방출 반쪽 크기", glm::value_ptr(fluid.emitterHalfExtents), 0.01F, 0.01F, 50.0F);
        ImGui::DragFloat("입자 반지름", &fluid.particleRadius, 0.001F, 0.005F, 1.0F, "%.3f");
        ImGui::DragFloat("커널 반지름", &fluid.smoothingRadius, 0.005F, 0.01F, 2.0F, "%.3f");
        ImGui::DragFloat("기준 밀도", &fluid.restDensity, 1.0F, 1.0F, 10000.0F);
        ImGui::DragFloat("강성", &fluid.stiffness, 1.0F, 1.0F, 5000.0F);
        ImGui::DragFloat("점성", &fluid.viscosity, 0.01F, 0.0F, 50.0F);
        ImGui::DragFloat3("용기 최소", glm::value_ptr(fluid.containerMin), 0.05F, -100.0F, 100.0F);
        ImGui::DragFloat3("용기 최대", glm::value_ptr(fluid.containerMax), 0.05F, -100.0F, 100.0F);
        ImGui::DragFloat3("중력", glm::value_ptr(fluid.gravity), 0.1F, -100.0F, 100.0F);
        ImGui::TextDisabled("입자는 GPU 에서 계산해 내장 구로 그린다. Path Tracing에도 보인다");
    }

    if (object.particleSystem >= 0 && object.particleSystem < static_cast<int>(active.particleSystems.size()) &&
        componentHeader("입자", &scene::Object::particleSystem)) {
        scene::ParticleSystem& system = active.particleSystems[static_cast<size_t>(object.particleSystem)];
        if (!renderer.particleGpuAvailable()) {
            ImGui::TextDisabled("이 장치에서는 입자 컴퓨트를 만들지 못했다");
        }
        int maxParticles = static_cast<int>(system.maxParticles);
        if (ImGui::SliderInt("최대 입자", &maxParticles, 64, 65536, "%d", ImGuiSliderFlags_Logarithmic)) {
            system.maxParticles = static_cast<uint32_t>(maxParticles);
        }
        ImGui::DragFloat("초당 방출", &system.emitRate, 1.0F, 0.0F, 100000.0F, "%.0f");
        ImGui::DragFloat("수명", &system.lifetime, 0.05F, 0.05F, 60.0F, "%.2f");
        ImGui::DragFloat("초기 속력", &system.initialSpeed, 0.05F, 0.0F, 100.0F, "%.2f");
        ImGui::SliderFloat("퍼짐 각", &system.spreadAngleDegrees, 0.0F, 180.0F, "%.0f°");
        ImGui::DragFloat("중력 배율", &system.gravityScale, 0.01F, -5.0F, 5.0F, "%.2f");
        ImGui::DragFloat("항력", &system.drag, 0.01F, 0.0F, 20.0F, "%.2f");
        ImGui::DragFloat("시작 지름", &system.sizeStart, 0.005F, 0.0F, 10.0F, "%.3f");
        ImGui::DragFloat("끝 지름", &system.sizeEnd, 0.005F, 0.0F, 10.0F, "%.3f");
        ImGui::ColorEdit4("색", glm::value_ptr(system.color));
        ImGui::ColorEdit3("발광", glm::value_ptr(system.emissive), ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
        ImGui::DragFloat("반발", &system.restitution, 0.01F, 0.0F, 1.0F, "%.2f");
        bool collisionUsable = renderer.particleCollisionAvailable();
        ImGui::BeginDisabled(!collisionUsable);
        ImGui::Checkbox("충돌", &system.collide);
        ImGui::EndDisabled();
        if (!collisionUsable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("광선 질의가 없어 장면과 부딪히지 않는다");
        }
        ImGui::TextDisabled("GPU 전용. 카메라를 향한 스프라이트로 그리며 Path Tracing 에는 보이지 않는다.\n"
                            "충돌은 상위 가속 구조에 광선 질의로 판정한다");
    }

    if (object.cloth >= 0 && object.cloth < static_cast<int>(active.cloths.size()) &&
        componentHeader("천", &scene::Object::cloth)) {
        scene::Cloth& cloth = active.cloths[static_cast<size_t>(object.cloth)];
        auto clothIndex = static_cast<uint32_t>(object.cloth);
        constexpr std::array<const char*, 3> BACKEND_NAMES{"자동", "CPU", "GPU"};
        bool gpuUsable = renderer.clothGpuAvailable();
        int backendIndex = static_cast<int>(cloth.backend);
        if (ImGui::BeginCombo("백엔드", BACKEND_NAMES[static_cast<size_t>(backendIndex)])) {
            for (int i = 0; i < static_cast<int>(BACKEND_NAMES.size()); ++i) {
                ImGui::BeginDisabled(i == static_cast<int>(scene::SimulationBackend::GPU) && !gpuUsable);
                if (ImGui::Selectable(BACKEND_NAMES[static_cast<size_t>(i)], i == backendIndex)) {
                    cloth.backend = static_cast<scene::SimulationBackend>(i);
                }
                ImGui::EndDisabled();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(지금 %s)", renderer.clothOnCpu(clothIndex) ? "CPU" : "GPU");
        constexpr std::array<uint32_t, 3> RESOLUTIONS{16U, 32U, 64U};
        constexpr std::array<const char*, 3> RESOLUTION_NAMES{"16×16", "32×32", "64×64"};
        size_t resolutionIndex = cloth.resolution <= 16 ? 0 : (cloth.resolution <= 32 ? 1 : 2);
        if (ImGui::BeginCombo("해상도", RESOLUTION_NAMES[resolutionIndex])) {
            for (size_t i = 0; i < RESOLUTIONS.size(); ++i) {
                if (ImGui::Selectable(RESOLUTION_NAMES[i], i == resolutionIndex)) {
                    cloth.resolution = RESOLUTIONS[i];
                    // 메쉬도 그 격자로 바꾼다. 정점 수가 맞지 않으면 천이 돌지 않는다.
                    auto primitive = static_cast<size_t>(asset::clothPrimitiveFor(cloth.resolution));
                    if (primitive < primitiveMeshes.size()) {
                        active.attachMeshRenderer(objectIndex, primitiveMeshes[primitive]);
                    }
                }
            }
            ImGui::EndCombo();
        }
        ImGui::DragFloat("질량", &cloth.mass, 0.01F, 0.01F, 100.0F, "%.2f kg");
        ImGui::DragFloat("신장 컴플라이언스", &cloth.stretchCompliance, 1e-4F, 0.0F, 1.0F, "%.4f");
        ImGui::DragFloat("전단 컴플라이언스", &cloth.shearCompliance, 1e-4F, 0.0F, 1.0F, "%.4f");
        ImGui::DragFloat("굽힘 컴플라이언스", &cloth.bendCompliance, 1e-4F, 0.0F, 1.0F, "%.4f");
        ImGui::DragFloat("감쇠", &cloth.damping, 0.01F, 0.0F, 20.0F, "%.2f");
        int substeps = static_cast<int>(cloth.substeps);
        if (ImGui::SliderInt("서브스텝", &substeps, 1, 16)) {
            cloth.substeps = static_cast<uint32_t>(substeps);
        }
        int iterations = static_cast<int>(cloth.iterations);
        if (ImGui::SliderInt("반복", &iterations, 1, 32)) {
            cloth.iterations = static_cast<uint32_t>(iterations);
        }
        constexpr std::array<const char*, 3> PIN_NAMES{"없음", "윗변", "위 모서리 둘"};
        int pinIndex = static_cast<int>(cloth.pin);
        if (ImGui::Combo("고정", &pinIndex, PIN_NAMES.data(), static_cast<int>(PIN_NAMES.size()))) {
            cloth.pin = static_cast<scene::ClothPin>(pinIndex);
        }
        ImGui::DragFloat("두께", &cloth.thickness, 0.001F, 0.0F, 1.0F, "%.3f");
        ImGui::SliderFloat("마찰", &cloth.friction, 0.0F, 1.0F, "%.2f");
        ImGui::DragFloat3("중력", glm::value_ptr(cloth.gravity), 0.1F, -100.0F, 100.0F);
        ImGui::DragFloat3("바람", glm::value_ptr(cloth.wind), 0.1F, -100.0F, 100.0F);
        if (!renderer.clothActive(clothIndex)) {
            ImGui::TextColored(ImVec4{1.0F, 0.6F, 0.3F, 1.0F},
                               "메쉬가 천 격자가 아니다. 해상도를 다시 골라 격자를 붙인다");
        }
        ImGui::TextDisabled(
            "크기는 오브젝트 배율, 정지 자세는 오브젝트 변환이다. 윗변은 월드에서 가장 높은 변이다.\n"
            "콜라이더 도형과 부딪히고, 메쉬 콜라이더는 GPU 가 광선 질의(TLAS), CPU 가 삼각형으로 본다.\n"
            "Path Tracing 과 광선 반사·그림자에 변형 그대로 보인다");
        if (!renderer.clothMeshCollisionAvailable()) {
            ImGui::TextDisabled("광선 질의가 없어 GPU 백엔드는 메쉬 콜라이더를 지나간다");
        }
    }

    ImGui::Separator();
    if (ImGui::Button("컴포넌트 추가", ImVec2{-1.0F, 0.0F})) {
        ImGui::OpenPopup("컴포넌트 추가");
    }
    if (ImGui::BeginPopup("컴포넌트 추가")) {
        ImGui::BeginDisabled(object.meshRenderer >= 0);
        if (ImGui::BeginMenu("Mesh Renderer")) {
            for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
                if (!geometry.meshLive(meshIndex)) {
                    continue;
                }
                std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex);
                if (ImGui::MenuItem(label.c_str())) {
                    active.attachMeshRenderer(objectIndex, meshIndex);
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.light >= 0);
        if (ImGui::MenuItem("조명")) {
            active.attachLight(objectIndex);
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.rigidBody >= 0);
        if (ImGui::MenuItem("강체")) {
            scene::RigidBody body;
            // 메쉬가 있으면 경계 구를 콜라이더 크기로 삼는다.
            uint32_t mesh = active.meshOf(objectIndex);
            if (geometry.meshLive(mesh)) {
                body.radius = std::max(geometry.mesh(mesh).boundingSphere.w, 0.01F);
                body.halfExtents = glm::vec3{body.radius * 0.7F};
            }
            active.attachRigidBody(objectIndex, body);
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.fluid >= 0);
        if (ImGui::MenuItem("유체")) {
            active.attachFluid(objectIndex);
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.particleSystem >= 0);
        if (ImGui::MenuItem("입자")) {
            active.attachParticleSystem(objectIndex);
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.cloth >= 0);
        if (ImGui::MenuItem("천")) {
            scene::Cloth cloth;
            active.attachCloth(objectIndex, cloth);
            // 메쉬를 32×32 격자로 붙인다. 이미 있던 메쉬는 천 격자로 바뀐다.
            auto primitive = static_cast<size_t>(asset::clothPrimitiveFor(cloth.resolution));
            if (primitive < primitiveMeshes.size()) {
                active.attachMeshRenderer(objectIndex, primitiveMeshes[primitive]);
            }
        }
        ImGui::EndDisabled();
        ImGui::EndPopup();
    }
    ImGui::End();
}

} // namespace editor
