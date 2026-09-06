// 장면 뷰(렌더 결과 표시, 기즈모, 클릭 선택).
// Editor 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 editor.h 하나에 있다.

#include "editor/editor_internal.h"

namespace editor {

namespace {

// 광선과 구가 만나는 가장 가까운 거리. 만나지 않거나 뒤쪽이면 음수.
float raySphere(const scene::Ray& ray, const glm::vec4& sphere) {
    glm::vec3 toCenter = glm::vec3(sphere) - ray.origin;
    float alongRay = glm::dot(toCenter, ray.direction);
    float centerDistanceSq = glm::dot(toCenter, toCenter) - alongRay * alongRay;
    float radiusSq = sphere.w * sphere.w;
    if (centerDistanceSq > radiusSq) {
        return -1.0F;
    }
    float halfChord = std::sqrt(radiusSq - centerDistanceSq);
    float near = alongRay - halfChord;
    float far = alongRay + halfChord;
    // 구 안에서 쏘면 near 가 음수다. 그때는 반대쪽 교차를 쓴다.
    return near >= 0.0F ? near : (far >= 0.0F ? far : -1.0F);
}

// 광선에 걸리는 가장 가까운 오브젝트. 없으면 -1. 경계 구로 거른 뒤 메쉬 콜라이더 표(강체 메쉬 콜라이더가 쓰는
// 상한 아래의 가장 고운 LOD, 위치만)의 삼각형과 교차시킨다. 지역 공간에서 정규화하지 않은 방향으로 재면 t 가
// 세계 공간 배율 그대로라 오브젝트끼리 견줄 수 있다.
//
// ponytail: 스킨·천 오브젝트는 CPU 정점이 바인드 포즈·초기 격자라 구로만 판정한다. 변형 정점을 되읽으면 정확해진다.
int pickObject(const scene::Scene& scene, const gfx::GeometryStore& geometry, const scene::Ray& ray) {
    int best = -1;
    float bestDistance = std::numeric_limits<float>::max();
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        uint32_t mesh = scene.meshOf(index);
        if (!geometry.meshLive(mesh) || !scene.visibleInTree(index)) {
            continue;
        }
        const glm::mat4& world = scene.worldMatrix(index);
        glm::vec4 local = geometry.mesh(mesh).boundingSphere;
        glm::vec3 center = glm::vec3(world * glm::vec4{glm::vec3(local), 1.0F});
        // 비균등 스케일은 가장 긴 축으로 보수적으로 키운다. focusSelected 와 같은 계산이다.
        float scale = std::sqrt(std::max({glm::dot(glm::vec3(world[0]), glm::vec3(world[0])),
                                          glm::dot(glm::vec3(world[1]), glm::vec3(world[1])),
                                          glm::dot(glm::vec3(world[2]), glm::vec3(world[2]))}));
        float distance = raySphere(ray, glm::vec4{center, local.w * scale});
        if (distance < 0.0F || distance >= bestDistance) {
            continue;
        }
        const scene::Object& object = scene.objects[index];
        const scene::ColliderMesh* collider = scene.colliderMesh(index);
        // 스케일이 0 인 축이 있으면 역행렬이 없다. 그때는 구 판정으로 남긴다.
        if (collider != nullptr && object.animator < 0 && object.cloth < 0 &&
            std::abs(glm::determinant(world)) > 1.0e-12F) {
            glm::mat4 inverse = glm::inverse(world);
            scene::Ray localRay{glm::vec3(inverse * glm::vec4{ray.origin, 1.0F}),
                                glm::vec3(inverse * glm::vec4{ray.direction, 0.0F})};
            distance = -1.0F;
            for (size_t i = 0; i + 2 < collider->indices.size(); i += 3) {
                float hit = scene::rayTriangle(localRay,
                                               collider->positions[collider->indices[i]],
                                               collider->positions[collider->indices[i + 1]],
                                               collider->positions[collider->indices[i + 2]]);
                if (hit >= 0.0F && (distance < 0.0F || hit < distance)) {
                    distance = hit;
                }
            }
            if (distance < 0.0F || distance >= bestDistance) {
                continue;
            }
        }
        bestDistance = distance;
        best = static_cast<int>(index);
    }
    return best;
}

} // namespace

void Editor::buildSceneView(scene::Scene& active) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0F, 0.0F});
    bool open = ImGui::Begin(WINDOW_SCENE);
    ImGui::PopStyleVar();
    if (!open) {
        sceneHovered = false;
        ImGui::End();
        return;
    }

    // ImGui 좌표는 논리 단위이므로, 실제 픽셀 해상도로 그리려면 프레임버퍼 배율을 곱해야 한다.
    ImVec2 available = ImGui::GetContentRegionAvail();
    ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
    if (available.x >= 1.0F && available.y >= 1.0F) {
        viewportExtent = {static_cast<uint32_t>(available.x * scale.x), static_cast<uint32_t>(available.y * scale.y)};
    }

    // 툴바에서 고른 렌더 타깃을 장면 자리에 보여 준다. 기본은 표시 이미지고, 다른 대상은 자기 비율로
    // 가운데에 맞춰 넣는다. 기즈모와 클릭 선택은 표시 이미지일 때만 받는다.
    std::vector<gfx::Renderer::TargetView> targets = renderer.targetViews();
    int targetCount = static_cast<int>(targets.size());
    int presentIndex = 0;
    for (int i = 0; i < targetCount; ++i) {
        if (targets[static_cast<size_t>(i)].views[0] == renderer.presentView()) {
            presentIndex = i;
        }
    }
    // 렌더 모드가 바뀌어 고른 대상이 더는 채워지지 않으면 표시 이미지로 돌아간다.
    if (selectedTarget < 0 || selectedTarget >= targetCount ||
        !targets[static_cast<size_t>(selectedTarget)].available) {
        selectedTarget = presentIndex;
    }
    const gfx::Renderer::TargetView& target = targets[static_cast<size_t>(selectedTarget)];
    bool presentView = selectedTarget == presentIndex;
    int sliceCount = static_cast<int>(target.views.size());
    selectedSlice = std::clamp(selectedSlice, 0, sliceCount - 1);

    ImVec2 imageArea = available;
    ImVec2 imageOrigin = ImGui::GetCursorPos();
    if (!presentView) {
        float aspect = static_cast<float>(target.extent.width) / static_cast<float>(std::max(target.extent.height, 1U));
        float height = std::min(available.y, available.x / aspect);
        imageArea = ImVec2{height * aspect, height};
        ImGui::SetCursorPos(ImVec2{imageOrigin.x + (available.x - imageArea.x) * 0.5F,
                                   imageOrigin.y + (available.y - imageArea.y) * 0.5F});
    }
    ImGui::Image(ImTextureRef{static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(
                     textureFor(target.views[static_cast<size_t>(selectedSlice)], target.layout)))},
                 imageArea);
    sceneHovered = ImGui::IsItemHovered();
    bool imageClicked = presentView && ImGui::IsItemClicked(ImGuiMouseButton_Left);

    ImVec2 imagePosition = ImGui::GetItemRectMin();
    ImVec2 imageSize = ImGui::GetItemRectSize();
    // 툴바는 대상 크기와 무관하게 장면 창 왼쪽 위에 둔다.
    ImVec2 toolbarPosition = ImVec2{ImGui::GetWindowPos().x + imageOrigin.x + 8.0F,
                                    ImGui::GetWindowPos().y + imageOrigin.y - ImGui::GetScrollY() + 8.0F};

    gizmoUsing = false;
    bool anySelected = hasSelection() && presentView;
    if (anySelected && imageSize.x > 1.0F && imageSize.y > 1.0F) {
        // 텍스트 입력 중에는 W/E/R 이 글자다.
        if (!active.camera.isLooking() && !ImGui::GetIO().WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_W)) {
                gizmoOperation = ImGuizmo::TRANSLATE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_E)) {
                gizmoOperation = ImGuizmo::ROTATE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                gizmoOperation = ImGuizmo::SCALE;
            }
        }

        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(imagePosition.x, imagePosition.y, imageSize.x, imageSize.y);

        float aspect = imageSize.x / imageSize.y;
        glm::mat4 view = active.camera.viewMatrix();
        glm::mat4 projection = active.camera.gizmoProjectionMatrix(aspect);

        // 기즈모는 기준 오브젝트 자리에 서고, 끌어서 생긴 차이를 선택 전체에 똑같이 먹인다.
        auto primary = static_cast<uint32_t>(primarySelection());
        glm::mat4 pivotBefore = active.worldMatrix(primary);
        glm::mat4 model = pivotBefore;

        // Ctrl 을 누르면 스냅을 건다. 값은 툴바에서 고치고, «스냅» 체크는 늘 걸리게 한다.
        glm::vec3 snapValue{snapTranslate};
        if (gizmoOperation == ImGuizmo::ROTATE) {
            snapValue = glm::vec3{snapRotate};
        } else if (gizmoOperation == ImGuizmo::SCALE) {
            snapValue = glm::vec3{snapScale};
        }
        const float* snap = snapAlways || ImGui::GetIO().KeyCtrl ? glm::value_ptr(snapValue) : nullptr;

        if (ImGuizmo::Manipulate(glm::value_ptr(view),
                                 glm::value_ptr(projection),
                                 static_cast<ImGuizmo::OPERATION>(gizmoOperation),
                                 static_cast<ImGuizmo::MODE>(gizmoMode),
                                 glm::value_ptr(model),
                                 nullptr,
                                 snap)) {
            glm::mat4 delta = model * glm::inverse(pivotBefore);

            // 조상이 함께 선택돼 있으면 그쪽에서 이미 옮겨진다. 여기서 또 먹이면 두 배로 움직인다.
            auto ancestorSelected = [this, &active](int index) {
                int parent = active.objects[static_cast<size_t>(index)].parent;
                while (parent >= 0) {
                    if (isSelected(parent)) {
                        return true;
                    }
                    parent = active.objects[static_cast<size_t>(parent)].parent;
                }
                return false;
            };

            // 옮기기 전에 세계 변환을 모두 재어 둔다. 하나를 고치면 그 자손의 세계 변환이 따라
            // 움직여, 순회 도중에 재면 뒤쪽이 어긋난다.
            std::vector<std::pair<uint32_t, glm::mat4>> targets;
            for (int selected : selection) {
                if (selected < 0 || selected >= static_cast<int>(active.objects.size()) || ancestorSelected(selected)) {
                    continue;
                }
                auto index = static_cast<uint32_t>(selected);
                targets.emplace_back(index, delta * active.worldMatrix(index));
            }
            for (const auto& [index, world] : targets) {
                int parent = active.objects[static_cast<size_t>(index)].parent;
                glm::mat4 parentWorld =
                    parent >= 0 ? active.worldMatrix(static_cast<uint32_t>(parent)) : glm::mat4{1.0F};
                active.objects[static_cast<size_t>(index)].transform =
                    scene::Transform::fromMatrix(glm::inverse(parentWorld) * world);
            }
        }
        gizmoUsing = ImGuizmo::IsUsing();
    }

    // 장면 뷰를 왼쪽 단추로 누르면 그 아래 오브젝트를 고른다. 기즈모를 잡고 있거나 시선을 돌리는
    // 중에는 받지 않는다. 빈 곳을 누르면 선택이 풀린다.
    if (imageClicked && !gizmoUsing && !ImGuizmo::IsOver() && !active.camera.isLooking() && geometryStore != nullptr &&
        imageSize.x > 1.0F && imageSize.y > 1.0F) {
        ImVec2 mouse = ImGui::GetMousePos();
        glm::vec2 uv{(mouse.x - imagePosition.x) / imageSize.x, (mouse.y - imagePosition.y) / imageSize.y};
        scene::Ray ray = active.camera.screenToRay(uv, imageSize.x / imageSize.y);
        int picked = pickObject(active, *geometryStore, ray);
        if (ImGui::GetIO().KeyCtrl) {
            toggleSelect(picked);
        } else {
            selectOnly(picked);
        }
    }

    // 조작 도구 선택은 장면 뷰 위에 겹쳐 둔다. 맨 앞에서 무엇을 볼지 고른다. 위젯을 채널 1 에 그리고
    // 그 뒤(채널 0)에 배경을 깐다. 장면 위에 바로 놓으면 밝은 하늘에서 글자가 묻힌다.
    ImGui::SetCursorScreenPos(toolbarPosition);
    ImDrawList* toolbarDrawList = ImGui::GetWindowDrawList();
    toolbarDrawList->ChannelsSplit(2);
    toolbarDrawList->ChannelsSetCurrent(1);
    // 기본 프레임 색은 배경과 거의 같아 라디오·체크박스 테두리가 묻힌다. 툴바 안에서만 밝게 둔다.
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4{0.34F, 0.34F, 0.36F, 1.0F});
    ImGui::BeginGroup();
    ImGui::SetNextItemWidth(160.0F);
    if (ImGui::BeginCombo("##target", target.name)) {
        for (int i = 0; i < targetCount; ++i) {
            const gfx::Renderer::TargetView& candidate = targets[static_cast<size_t>(i)];
            ImGui::BeginDisabled(!candidate.available);
            if (ImGui::Selectable(candidate.name, selectedTarget == i)) {
                selectedTarget = i;
                selectedSlice = 0;
            }
            ImGui::EndDisabled();
            if (!candidate.available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("지금 렌더 모드에서는 채워지지 않는 대상이다");
            }
        }
        ImGui::EndCombo();
    }
    if (sliceCount > 1) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0F);
        ImGui::SliderInt(target.sliceLabel, &selectedSlice, 0, sliceCount - 1);
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("|");
    ImGui::SameLine();
    if (ImGui::RadioButton("이동", gizmoOperation == ImGuizmo::TRANSLATE)) {
        gizmoOperation = ImGuizmo::TRANSLATE;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("회전", gizmoOperation == ImGuizmo::ROTATE)) {
        gizmoOperation = ImGuizmo::ROTATE;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("크기", gizmoOperation == ImGuizmo::SCALE)) {
        gizmoOperation = ImGuizmo::SCALE;
    }
    ImGui::SameLine();
    if (ImGui::Button(gizmoMode == ImGuizmo::LOCAL ? "로컬" : "월드")) {
        gizmoMode = gizmoMode == ImGuizmo::LOCAL ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
    }
    // 스냅 값은 지금 고른 조작 종류의 것을 보여 준다.
    ImGui::SameLine();
    ImGui::Checkbox("스냅", &snapAlways);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Ctrl 을 누르는 동안 스냅이 걸린다. 체크하면 늘 건다");
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(72.0F);
    if (gizmoOperation == ImGuizmo::ROTATE) {
        ImGui::DragFloat("##snapRotate", &snapRotate, 1.0F, 1.0F, 90.0F, "%.0f°");
    } else if (gizmoOperation == ImGuizmo::SCALE) {
        ImGui::DragFloat("##snapScale", &snapScale, 0.01F, 0.01F, 2.0F, "x%.2f");
    } else {
        ImGui::DragFloat("##snapTranslate", &snapTranslate, 0.05F, 0.01F, 100.0F, "%.2f");
    }

    // 카메라 조작 방식. 기본은 궤도이고, 자유 모드는 1인칭처럼 날아다닌다.
    ImGui::SameLine();
    ImGui::TextUnformatted("|");
    ImGui::SameLine();
    bool orbit = active.camera.mode == scene::CameraMode::ORBIT;
    if (ImGui::RadioButton("궤도", orbit)) {
        active.camera.setMode(scene::CameraMode::ORBIT);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("자유", !orbit)) {
        active.camera.setMode(scene::CameraMode::FLY);
    }
    ImGui::SameLine();
    ImGui::TextDisabled(orbit ? "(?) 우클릭 회전, 휠 확대, 가운데 단추 이동"
                              : "(?) 우클릭 시선, WASD/방향키 이동, 휠 속도");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(orbit ? "선택한 오브젝트로 궤도 중심을 옮기려면 계층에서 F 를 누른다"
                                : "Q/E 또는 PageUp/PageDown 으로 오르내리고, Shift 로 4배 빨라진다");
    }
    ImGui::EndGroup();
    ImGui::PopStyleColor();
    {
        const ImVec2 PAD{6.0F, 4.0F};
        ImVec2 minimum = ImGui::GetItemRectMin();
        ImVec2 maximum = ImGui::GetItemRectMax();
        toolbarDrawList->ChannelsSetCurrent(0);
        toolbarDrawList->AddRectFilled(ImVec2{minimum.x - PAD.x, minimum.y - PAD.y},
                                       ImVec2{maximum.x + PAD.x, maximum.y + PAD.y},
                                       IM_COL32(18, 18, 18, 210),
                                       4.0F);
        toolbarDrawList->ChannelsMerge();
    }

    ImGui::End();
}

} // namespace editor
