#include <algorithm>
#include <cstddef>

#include <glm/ext/quaternion_common.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "asset/model.h"

namespace asset {
namespace {

glm::vec4 sampleChannel(const AnimationSampler& sampler, float time, bool rotation) {
    if (sampler.times.empty()) {
        return glm::vec4{0.0F};
    }
    if (time <= sampler.times.front()) {
        return sampler.values.front();
    }
    if (time >= sampler.times.back()) {
        return sampler.values.back();
    }

    size_t next = static_cast<size_t>(std::ranges::upper_bound(sampler.times, time) - sampler.times.begin());
    size_t previous = next - 1;
    if (sampler.step) {
        return sampler.values[previous];
    }

    float span = sampler.times[next] - sampler.times[previous];
    float factor = span > 0.0F ? (time - sampler.times[previous]) / span : 0.0F;
    if (rotation) {
        // 사원수는 선형 보간하면 각속도가 흔들리므로 구면 보간한다.
        const glm::vec4& a = sampler.values[previous];
        const glm::vec4& b = sampler.values[next];
        glm::quat blended = glm::slerp(glm::quat{a.w, a.x, a.y, a.z}, glm::quat{b.w, b.x, b.y, b.z}, factor);
        return glm::vec4{blended.x, blended.y, blended.z, blended.w};
    }
    return glm::mix(sampler.values[previous], sampler.values[next], factor);
}

glm::mat4 localMatrix(const Node& node) {
    return glm::translate(glm::mat4{1.0F}, node.translation) * glm::mat4_cast(node.rotation) *
           glm::scale(glm::mat4{1.0F}, node.scale);
}

} // namespace

void poseNodes(const Skeleton& skeleton, uint32_t clip, float time, std::vector<glm::mat4>& worlds) {
    // 노드 수가 수십 개 수준이라 매 프레임 복사해도 부담이 없다.
    std::vector<Node> posed = skeleton.nodes;
    if (clip < skeleton.animations.size()) {
        const Animation& animation = skeleton.animations[clip];
        for (const AnimationChannel& channel : animation.channels) {
            if (channel.sampler >= animation.samplers.size() || channel.node >= posed.size()) {
                continue;
            }
            glm::vec4 value =
                sampleChannel(animation.samplers[channel.sampler], time, channel.path == AnimationPath::ROTATION);
            Node& node = posed[channel.node];
            switch (channel.path) {
            case AnimationPath::TRANSLATION:
                node.translation = glm::vec3{value};
                break;
            case AnimationPath::ROTATION:
                node.rotation = glm::normalize(glm::quat{value.w, value.x, value.y, value.z});
                break;
            case AnimationPath::SCALE:
                node.scale = glm::vec3{value};
                break;
            }
        }
    }

    worlds.assign(posed.size(), glm::mat4{1.0F});
    std::vector<bool> resolved(posed.size(), false);
    std::vector<uint32_t> chain;
    for (uint32_t i = 0; i < posed.size(); ++i) {
        // 부모가 배열에서 항상 앞선다는 보장이 없어 뿌리까지 거슬러 올라간 뒤 내려오며 채운다.
        chain.clear();
        int32_t current = static_cast<int32_t>(i);
        while (current >= 0 && !resolved[static_cast<size_t>(current)] && chain.size() <= posed.size()) {
            chain.push_back(static_cast<uint32_t>(current));
            current = posed[static_cast<size_t>(current)].parent;
        }
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            const Node& node = posed[*it];
            glm::mat4 local = localMatrix(node);
            worlds[*it] = node.parent >= 0 ? worlds[static_cast<size_t>(node.parent)] * local : local;
            resolved[*it] = true;
        }
    }
}

void skinMatrices(const Skeleton& skeleton,
                  const std::vector<glm::mat4>& worlds,
                  uint32_t skin,
                  std::vector<glm::mat4>& out) {
    if (skin >= skeleton.skins.size()) {
        out.clear();
        return;
    }
    const Skin& target = skeleton.skins[skin];
    out.resize(target.joints.size());
    for (size_t i = 0; i < target.joints.size(); ++i) {
        uint32_t node = target.joints[i];
        out[i] = node < worlds.size() ? worlds[node] * target.inverseBind[i] : glm::mat4{1.0F};
    }
}

} // namespace asset
