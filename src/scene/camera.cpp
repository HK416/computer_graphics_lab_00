#include "scene/camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>
#include <SDL3/SDL.h>

namespace scene {
namespace {
constexpr float MAX_PITCH_DEGREES = 89.0F;
constexpr glm::vec3 WORLD_UP{0.0F, 1.0F, 0.0F};
// 기즈모 전용 투영의 원평면. 광선 계산의 수치 조건만 결정하므로 장면 크기에 맞는 값이면 된다.
constexpr float GIZMO_FAR_PLANE = 1000.0F;
} // namespace

glm::vec3 Camera::forward() const {
    float yaw = glm::radians(yawDegrees);
    float pitch = glm::radians(pitchDegrees);
    return glm::normalize(glm::vec3{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)});
}

void Camera::handleEvent(const SDL_Event& event) {
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_RIGHT) {
        looking = true;
        SDL_SetWindowRelativeMouseMode(SDL_GetWindowFromID(event.button.windowID), true);
    } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_RIGHT) {
        looking = false;
        SDL_SetWindowRelativeMouseMode(SDL_GetWindowFromID(event.button.windowID), false);
    } else if (event.type == SDL_EVENT_MOUSE_MOTION && looking) {
        yawDegrees += event.motion.xrel * lookSensitivity;
        pitchDegrees =
            std::clamp(pitchDegrees - event.motion.yrel * lookSensitivity, -MAX_PITCH_DEGREES, MAX_PITCH_DEGREES);
    }
}

void Camera::update(float deltaSeconds) {
    if (!looking) {
        return;
    }
    const bool* keys = SDL_GetKeyboardState(nullptr);
    glm::vec3 front = forward();
    glm::vec3 right = glm::normalize(glm::cross(front, WORLD_UP));

    glm::vec3 movement{0.0F};
    movement += front * static_cast<float>(keys[SDL_SCANCODE_W] - keys[SDL_SCANCODE_S]);
    movement += right * static_cast<float>(keys[SDL_SCANCODE_D] - keys[SDL_SCANCODE_A]);
    movement += WORLD_UP * static_cast<float>(keys[SDL_SCANCODE_E] - keys[SDL_SCANCODE_Q]);
    if (glm::dot(movement, movement) <= 0.0F) {
        return;
    }

    float speed = moveSpeed * (keys[SDL_SCANCODE_LSHIFT] ? 4.0F : 1.0F);
    position += glm::normalize(movement) * speed * deltaSeconds;
}

glm::mat4 Camera::viewMatrix() const {
    return glm::lookAt(position, position + forward(), WORLD_UP);
}

glm::mat4 Camera::projectionMatrix(float aspect) const {
    float focal = 1.0F / std::tan(glm::radians(fovYDegrees) * 0.5F);
    glm::mat4 projection(0.0F);
    projection[0][0] = focal / aspect;
    projection[1][1] = -focal; // Vulkan 클립 공간은 Y 가 아래로 향한다.
    projection[2][3] = -1.0F;
    projection[3][2] = nearPlane;
    return projection;
}

// ImGuizmo 는 마우스 광선을 만들 때 NDC 의 근평면 점과 원평면 점을 역변환해 잇는다. 무한 원거리
// 투영에서는 원평면 점의 w 가 0 이라 좌표가 무한대가 되고, 광선 방향이 NaN 이 되어 끌기 시작하는
// 순간 변환 행렬이 통째로 NaN 이 된다. 그래서 기즈모에는 유한한 원평면을 가진 투영을 따로 준다.
// Y 를 뒤집지 않는 것은 ImGuizmo 가 Y 축이 위로 향하는 클립 공간을 가정하기 때문이다.
glm::mat4 Camera::gizmoProjectionMatrix(float aspect) const {
    return glm::perspectiveRH_ZO(glm::radians(fovYDegrees), aspect, nearPlane, GIZMO_FAR_PLANE);
}

} // namespace scene
