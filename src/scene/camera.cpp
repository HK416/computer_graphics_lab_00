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

glm::mat4 Camera::gizmoProjectionMatrix(float aspect) const {
    glm::mat4 projection = projectionMatrix(aspect);
    projection[1][1] = -projection[1][1];
    return projection;
}

} // namespace scene
