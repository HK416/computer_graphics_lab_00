#pragma once

#include <SDL3/SDL_events.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace scene {

class Camera {
public:
    void handleEvent(const SDL_Event& event);
    void update(float deltaSeconds);

    glm::mat4 viewMatrix() const;
    // 무한 원거리 reverse-Z 투영. 깊이 비교는 GREATER 를 쓰고 깊이 버퍼는 0 으로 지운다.
    glm::mat4 projectionMatrix(float aspect) const;
    glm::vec3 forward() const;

    glm::vec3 position{0.0F, 0.5F, 2.5F};
    float yawDegrees = -90.0F;
    float pitchDegrees = 0.0F;
    float fovYDegrees = 60.0F;
    float nearPlane = 0.05F;
    float moveSpeed = 2.5F;
    float lookSensitivity = 0.1F;

private:
    bool looking = false;
};

} // namespace scene
