#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "asset/model.h"

namespace {

constexpr float TOLERANCE = 1e-4F;

// 자식이 배열에서 부모보다 앞서는 계층. 뿌리 노드만 Z 축으로 1 초 동안 90 도 돈다.
asset::Skeleton makeSkeleton() {
    asset::Skeleton skeleton;
    skeleton.nodes.resize(2);
    skeleton.nodes[0].parent = 1;
    skeleton.nodes[0].translation = glm::vec3{0.0F, 1.0F, 0.0F};
    skeleton.nodes[1].parent = -1;

    asset::Skin skin;
    skin.joints = {1, 0};
    skin.inverseBind = {glm::mat4{1.0F}, glm::translate(glm::mat4{1.0F}, glm::vec3{0.0F, -1.0F, 0.0F})};
    skeleton.skins.push_back(std::move(skin));

    glm::quat turned = glm::angleAxis(glm::radians(90.0F), glm::vec3{0.0F, 0.0F, 1.0F});
    asset::AnimationSampler sampler;
    sampler.times = {0.0F, 1.0F};
    sampler.values = {glm::vec4{0.0F, 0.0F, 0.0F, 1.0F}, glm::vec4{turned.x, turned.y, turned.z, turned.w}};

    asset::Animation animation;
    animation.name = "회전";
    animation.duration = 1.0F;
    animation.samplers.push_back(std::move(sampler));
    animation.channels.push_back({0, 1, asset::AnimationPath::ROTATION});
    skeleton.animations.push_back(std::move(animation));
    return skeleton;
}

// 자식 조인트에 완전히 묶인 정점 하나를 스키닝한다.
glm::vec3 skinnedPoint(const asset::Skeleton& skeleton, float time, glm::vec3 position) {
    std::vector<glm::mat4> worlds;
    std::vector<glm::mat4> matrices;
    asset::poseNodes(skeleton, 0, time, worlds);
    asset::skinMatrices(skeleton, worlds, 0, matrices);
    assert(matrices.size() == 2);
    return glm::vec3{matrices[1] * glm::vec4{position, 1.0F}};
}

void expectNear(glm::vec3 value, glm::vec3 expected, const char* label) {
    if (glm::length(value - expected) > TOLERANCE) {
        std::printf("%s: (%.4f, %.4f, %.4f) != (%.4f, %.4f, %.4f)\n",
                    label,
                    static_cast<double>(value.x),
                    static_cast<double>(value.y),
                    static_cast<double>(value.z),
                    static_cast<double>(expected.x),
                    static_cast<double>(expected.y),
                    static_cast<double>(expected.z));
        assert(false && "스키닝 결과가 기대값과 다릅니다");
    }
}

} // namespace

int main() {
    asset::Skeleton skeleton = makeSkeleton();
    glm::vec3 bound{0.0F, 1.0F, 0.0F};

    // 바인드 포즈에서는 조인트 행렬이 정점을 움직이지 않아야 한다.
    expectNear(skinnedPoint(skeleton, 0.0F, bound), bound, "바인드 포즈");

    // 90 도 회전이 끝나면 정점은 (0, 1, 0) 에서 (-1, 0, 0) 으로 간다.
    expectNear(skinnedPoint(skeleton, 1.0F, bound), glm::vec3{-1.0F, 0.0F, 0.0F}, "90 도");

    // 사원수 구면 보간이라 절반 시각에서는 정확히 45 도여야 한다.
    float half = std::sqrt(0.5F);
    expectNear(skinnedPoint(skeleton, 0.5F, bound), glm::vec3{-half, half, 0.0F}, "45 도");

    // 마지막 표본을 넘어선 시각은 마지막 값으로 고정된다.
    expectNear(skinnedPoint(skeleton, 4.0F, bound), glm::vec3{-1.0F, 0.0F, 0.0F}, "표본 범위 밖");

    // 애니메이션이 없는 클립 번호는 바인드 포즈로 되돌아간다.
    std::vector<glm::mat4> worlds;
    std::vector<glm::mat4> matrices;
    asset::poseNodes(skeleton, 7, 0.5F, worlds);
    asset::skinMatrices(skeleton, worlds, 0, matrices);
    expectNear(glm::vec3{matrices[1] * glm::vec4{bound, 1.0F}}, bound, "없는 클립");

    std::printf("애니메이션 자체 점검 통과\n");
    return 0;
}
