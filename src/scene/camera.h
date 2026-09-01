#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <SDL3/SDL_events.h>

namespace scene {

// 궤도(orbit)는 대상 주위를 도는 기본 조작이고, 자유(fly)는 1인칭처럼 직접 날아다닌다.
enum class CameraMode {
    ORBIT,
    FLY,
};

class Camera {
public:
    void handleEvent(const SDL_Event& event);
    void update(float deltaSeconds);

    glm::mat4 viewMatrix() const;
    // 무한 원거리 reverse-Z 투영. 깊이 비교는 GREATER 를 쓰고 깊이 버퍼는 0 으로 지운다.
    glm::mat4 projectionMatrix(float aspect) const;
    // ImGuizmo 는 OpenGL 규약(NDC +Y 위)을 가정하므로 Y 뒤집기를 되돌린 투영을 따로 준다.
    glm::mat4 gizmoProjectionMatrix(float aspect) const;
    glm::vec3 forward() const;
    bool isLooking() const { return dragging; }

    // 두 모드가 같은 화면에서 이어지도록 전환 시점에 대상이나 위치를 맞춰 준다.
    void setMode(CameraMode next);
    // 대상과 거리로 위치를 다시 잡는다. 궤도 모드에서 대상을 옮긴 뒤 부른다.
    void applyOrbit();
    // 이 지점을 중심으로 돌도록 대상을 옮긴다. 거리와 방향은 그대로 둔다.
    void focusOn(const glm::vec3& point, float radius);

    glm::vec3 position{0.0F, 0.5F, 2.5F};
    float yawDegrees = -90.0F;
    float pitchDegrees = 0.0F;
    float fovYDegrees = 60.0F;
    float nearPlane = 0.05F;
    float moveSpeed = 2.5F;
    float lookSensitivity = 0.1F;

    CameraMode mode = CameraMode::ORBIT;
    // 궤도 회전의 중심과 거리. 자유 모드에서는 쓰지 않는다.
    glm::vec3 target{0.0F, 0.5F, 0.0F};
    float distance = 2.5F;
    // 자유 모드에서 오른쪽 단추를 누르지 않아도 키보드 이동을 받을지. 편집기가 장면 뷰 위에
    // 마우스가 있을 때만 켠다. 그렇지 않으면 인스펙터에 값을 입력하다 카메라가 움직인다.
    bool keyboardEnabled = false;

private:
    void orbitRotate(float deltaYaw, float deltaPitch);
    void orbitPan(float deltaX, float deltaY);
    void orbitZoom(float steps);

    bool dragging = false;
    bool panning = false;
};

} // namespace scene
