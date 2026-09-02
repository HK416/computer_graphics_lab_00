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
// 궤도 거리는 0 이 되면 회전 중심이 눈 안으로 들어와 조작이 뒤집힌다.
constexpr float MIN_ORBIT_DISTANCE = 0.05F;
constexpr float ORBIT_ZOOM_PER_STEP = 0.12F;
} // namespace

glm::vec3 Camera::forward() const {
    float yaw = glm::radians(yawDegrees);
    float pitch = glm::radians(pitchDegrees);
    return glm::normalize(glm::vec3{std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)});
}

// 투영 행렬을 뒤집는 대신 시야각으로 직접 만든다. 무한 원거리 투영은 역변환이 성립하지 않고,
// 기즈모용 투영은 Y 부호가 반대라 어느 쪽을 써도 한 번 더 손봐야 하기 때문이다.
Ray Camera::screenToRay(glm::vec2 uv, float aspect) const {
    float ndcX = uv.x * 2.0F - 1.0F;
    // 화면은 위가 0 이라 뒤집어야 위쪽이 +가 된다.
    float ndcY = 1.0F - uv.y * 2.0F;
    float tanHalf = std::tan(glm::radians(fovYDegrees) * 0.5F);

    glm::vec3 ahead = forward();
    glm::vec3 right = glm::normalize(glm::cross(ahead, WORLD_UP));
    glm::vec3 up = glm::cross(right, ahead);

    Ray ray;
    ray.origin = position;
    ray.direction = glm::normalize(ahead + right * (ndcX * tanHalf * aspect) + up * (ndcY * tanHalf));
    return ray;
}

void Camera::applyOrbit() {
    distance = std::max(distance, MIN_ORBIT_DISTANCE);
    position = target - forward() * distance;
}

void Camera::setMode(CameraMode next) {
    if (next == mode) {
        return;
    }
    mode = next;
    // 화면이 튀지 않도록 전환 시점의 시선을 그대로 유지한다. 궤도로 갈 때는 지금 보고 있는
    // 앞쪽 한 점을 중심으로 삼는다.
    if (mode == CameraMode::ORBIT) {
        target = position + forward() * distance;
    }
}

void Camera::focusOn(const glm::vec3& point, float radius) {
    target = point;
    distance = std::max(radius, MIN_ORBIT_DISTANCE);
    if (mode == CameraMode::ORBIT) {
        applyOrbit();
    }
}

void Camera::orbitRotate(float deltaYaw, float deltaPitch) {
    yawDegrees += deltaYaw;
    pitchDegrees = std::clamp(pitchDegrees - deltaPitch, -MAX_PITCH_DEGREES, MAX_PITCH_DEGREES);
    applyOrbit();
}

void Camera::orbitPan(float deltaX, float deltaY) {
    glm::vec3 front = forward();
    glm::vec3 right = glm::normalize(glm::cross(front, WORLD_UP));
    glm::vec3 up = glm::cross(right, front);
    // 멀리서 볼수록 화면 한 픽셀이 덮는 거리가 커진다. 그만큼 이동도 커야 손맛이 같다.
    float scale = distance * 0.002F;
    target += (-right * deltaX + up * deltaY) * scale;
    applyOrbit();
}

void Camera::orbitZoom(float steps) {
    // 곱셈으로 줄여야 가까이서는 촘촘하고 멀리서는 성큼 움직인다.
    distance = std::max(distance * std::exp(-steps * ORBIT_ZOOM_PER_STEP), MIN_ORBIT_DISTANCE);
    applyOrbit();
}

void Camera::handleEvent(const SDL_Event& event) {
    SDL_Window* window = nullptr;
    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN || event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
        window = SDL_GetWindowFromID(event.button.windowID);
    }

    if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN &&
        (event.button.button == SDL_BUTTON_RIGHT || event.button.button == SDL_BUTTON_MIDDLE)) {
        dragging = true;
        // 궤도 모드에서 가운데 단추는 대상을 끌고 다니는 이동이다.
        panning = mode == CameraMode::ORBIT && event.button.button == SDL_BUTTON_MIDDLE;
        SDL_SetWindowRelativeMouseMode(window, true);
    } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP &&
               (event.button.button == SDL_BUTTON_RIGHT || event.button.button == SDL_BUTTON_MIDDLE)) {
        dragging = false;
        panning = false;
        SDL_SetWindowRelativeMouseMode(window, false);
    } else if (event.type == SDL_EVENT_MOUSE_MOTION && dragging) {
        if (mode == CameraMode::FLY) {
            yawDegrees += event.motion.xrel * lookSensitivity;
            pitchDegrees =
                std::clamp(pitchDegrees - event.motion.yrel * lookSensitivity, -MAX_PITCH_DEGREES, MAX_PITCH_DEGREES);
        } else if (panning) {
            orbitPan(event.motion.xrel, event.motion.yrel);
        } else {
            orbitRotate(event.motion.xrel * lookSensitivity, event.motion.yrel * lookSensitivity);
        }
    } else if (event.type == SDL_EVENT_MOUSE_WHEEL) {
        if (mode == CameraMode::ORBIT) {
            orbitZoom(event.wheel.y);
        } else {
            // 자유 모드에서 휠은 이동 속도를 조절한다. 위치를 직접 바꾸면 조작이 두 갈래가 된다.
            moveSpeed = std::clamp(moveSpeed * std::exp(event.wheel.y * ORBIT_ZOOM_PER_STEP), 0.01F, 10000.0F);
        }
    }
}

void Camera::update(float deltaSeconds) {
    if (mode == CameraMode::ORBIT) {
        // 궤도는 이벤트로만 움직인다. 위치는 대상과 거리에서 나오므로 여기서 다시 맞춰 둔다.
        applyOrbit();
        return;
    }
    if (!dragging && !keyboardEnabled) {
        return;
    }

    const bool* keys = SDL_GetKeyboardState(nullptr);
    glm::vec3 front = forward();
    glm::vec3 right = glm::normalize(glm::cross(front, WORLD_UP));

    // 방향키는 WASD 와 같은 뜻으로 둔다. 어느 쪽에 손을 얹어도 날아다닐 수 있다.
    auto axis = [keys](SDL_Scancode positive, SDL_Scancode negative) {
        return static_cast<float>(keys[positive]) - static_cast<float>(keys[negative]);
    };
    glm::vec3 movement{0.0F};
    movement += front * (axis(SDL_SCANCODE_W, SDL_SCANCODE_S) + axis(SDL_SCANCODE_UP, SDL_SCANCODE_DOWN));
    movement += right * (axis(SDL_SCANCODE_D, SDL_SCANCODE_A) + axis(SDL_SCANCODE_RIGHT, SDL_SCANCODE_LEFT));
    movement += WORLD_UP * (axis(SDL_SCANCODE_E, SDL_SCANCODE_Q) + axis(SDL_SCANCODE_PAGEUP, SDL_SCANCODE_PAGEDOWN));
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
