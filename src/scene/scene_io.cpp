#include "scene/scene_io.h"

#include <array>

#include <nlohmann/json.hpp>

#include "core/error.h"

namespace scene {
namespace {

using nlohmann::json;

json toJson(const glm::vec3& value) {
    return json::array({value.x, value.y, value.z});
}
json toJson(const glm::vec2& value) {
    return json::array({value.x, value.y});
}
json toJson(const glm::quat& value) {
    return json::array({value.x, value.y, value.z, value.w});
}

glm::vec3 toVec3(const json& value, const glm::vec3& fallback) {
    if (!value.is_array() || value.size() != 3) {
        return fallback;
    }
    return glm::vec3{value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
}

glm::vec2 toVec2(const json& value, const glm::vec2& fallback) {
    if (!value.is_array() || value.size() != 2) {
        return fallback;
    }
    return glm::vec2{value[0].get<float>(), value[1].get<float>()};
}

glm::quat toQuat(const json& value) {
    if (!value.is_array() || value.size() != 4) {
        return glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
    }
    return glm::quat{value[3].get<float>(), value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
}

// 조명 종류는 숫자보다 이름이 파일을 읽기 쉽게 한다.
constexpr std::array<const char*, 4> LIGHT_TYPE_NAMES{"directional", "point", "spot", "area"};

LightType toLightType(const std::string& name) {
    for (uint32_t i = 0; i < LIGHT_TYPE_NAMES.size(); ++i) {
        if (name == LIGHT_TYPE_NAMES[i]) {
            return static_cast<LightType>(i);
        }
    }
    return LightType::DIRECTIONAL;
}

// 전역 메쉬 인덱스가 속한 모델과 그 안에서의 번호를 찾는다.
bool locateMesh(const ModelTable& models, uint32_t meshIndex, int32_t& model, uint32_t& localMesh) {
    for (size_t i = 0; i < models.paths.size(); ++i) {
        if (meshIndex >= models.meshBase[i] && meshIndex < models.meshBase[i] + models.meshCount[i]) {
            model = static_cast<int32_t>(i);
            localMesh = meshIndex - models.meshBase[i];
            return true;
        }
    }
    return false;
}

} // namespace

std::string writeScene(const Scene& scene, const ModelTable& models, const std::filesystem::path& root) {
    json document;
    document["version"] = SCENE_FILE_VERSION;
    document["name"] = scene.name;

    json modelPaths = json::array();
    for (const std::filesystem::path& path : models.paths) {
        std::error_code error;
        std::filesystem::path relative = root.empty() ? path : std::filesystem::relative(path, root, error);
        // 뿌리 밖에 있는 모델은 절대 경로 그대로 둔다.
        bool inside = !relative.empty() && !error && relative.native().rfind("..", 0) != 0;
        modelPaths.push_back((inside ? relative : path).generic_string());
    }
    document["models"] = modelPaths;

    document["camera"] = {{"position", toJson(scene.camera.position)},
                          {"yaw", scene.camera.yawDegrees},
                          {"pitch", scene.camera.pitchDegrees},
                          {"fovY", scene.camera.fovYDegrees},
                          {"near", scene.camera.nearPlane},
                          {"moveSpeed", scene.camera.moveSpeed}};
    document["ambient"] = {{"color", toJson(scene.ambientColor)}, {"intensity", scene.ambientIntensity}};

    json lights = json::array();
    for (const Light& light : scene.lights) {
        lights.push_back({{"type", LIGHT_TYPE_NAMES[static_cast<size_t>(light.type)]},
                          {"color", toJson(light.color)},
                          {"intensity", light.intensity},
                          {"range", light.range},
                          {"innerCone", light.innerConeDegrees},
                          {"outerCone", light.outerConeDegrees},
                          {"size", toJson(light.size)},
                          {"castsShadow", light.castsShadow}});
    }
    document["lights"] = lights;

    json animators = json::array();
    for (const Animator& animator : scene.animators) {
        animators.push_back({{"name", animator.name},
                             {"model", animator.model},
                             {"clip", animator.clip},
                             {"time", animator.clipTime},
                             {"playing", animator.playing},
                             {"speed", animator.speed}});
    }
    document["animators"] = animators;

    json objects = json::array();
    for (const Object& object : scene.objects) {
        json entry{{"name", object.name},
                   {"parent", object.parent},
                   {"position", toJson(object.transform.position)},
                   {"rotation", toJson(object.transform.rotation)},
                   {"scale", toJson(object.transform.scale)},
                   {"visible", object.visible}};
        int32_t model = -1;
        uint32_t localMesh = 0;
        if (object.meshIndex != INVALID_MESH && locateMesh(models, object.meshIndex, model, localMesh)) {
            entry["model"] = model;
            entry["mesh"] = localMesh;
        }
        if (object.animator >= 0) {
            entry["animator"] = object.animator;
            entry["skin"] = object.skin;
        }
        if (object.light >= 0) {
            entry["light"] = object.light;
        }
        objects.push_back(std::move(entry));
    }
    document["objects"] = objects;

    return document.dump(2);
}

SceneFile readScene(const std::string& text) {
    json document = json::parse(text, nullptr, false);
    if (document.is_discarded() || !document.is_object()) {
        core::fatal("장면 파일을 해석할 수 없습니다");
    }
    auto version = document.value("version", 0U);
    if (version != SCENE_FILE_VERSION) {
        core::fatal("지원하지 않는 장면 파일 판입니다: {} (필요: {})", version, SCENE_FILE_VERSION);
    }

    SceneFile file;
    file.scene.name = document.value("name", std::string{"장면"});
    for (const json& path : document.value("models", json::array())) {
        file.models.emplace_back(path.get<std::string>());
    }

    const json& camera = document.value("camera", json::object());
    file.scene.camera.position = toVec3(camera.value("position", json{}), glm::vec3{0.0F});
    file.scene.camera.yawDegrees = camera.value("yaw", 90.0F);
    file.scene.camera.pitchDegrees = camera.value("pitch", 0.0F);
    file.scene.camera.fovYDegrees = camera.value("fovY", 60.0F);
    file.scene.camera.nearPlane = camera.value("near", 0.05F);
    file.scene.camera.moveSpeed = camera.value("moveSpeed", 1.0F);

    const json& ambient = document.value("ambient", json::object());
    file.scene.ambientColor = toVec3(ambient.value("color", json{}), glm::vec3{0.25F});
    file.scene.ambientIntensity = ambient.value("intensity", 1.0F);

    for (const json& entry : document.value("lights", json::array())) {
        Light light;
        light.type = toLightType(entry.value("type", std::string{"directional"}));
        light.color = toVec3(entry.value("color", json{}), glm::vec3{1.0F});
        light.intensity = entry.value("intensity", 3.0F);
        light.range = entry.value("range", 20.0F);
        light.innerConeDegrees = entry.value("innerCone", 20.0F);
        light.outerConeDegrees = entry.value("outerCone", 30.0F);
        light.size = toVec2(entry.value("size", json{}), glm::vec2{2.0F});
        light.castsShadow = entry.value("castsShadow", true);
        file.scene.lights.push_back(light);
    }

    for (const json& entry : document.value("animators", json::array())) {
        Animator animator;
        animator.name = entry.value("name", std::string{});
        animator.clip = entry.value("clip", 0U);
        animator.clipTime = entry.value("time", 0.0F);
        animator.playing = entry.value("playing", true);
        animator.speed = entry.value("speed", 1.0F);
        animator.model = entry.value("model", -1);
        file.animatorModels.push_back(animator.model);
        file.scene.animators.push_back(std::move(animator));
    }

    for (const json& entry : document.value("objects", json::array())) {
        Object object;
        object.name = entry.value("name", std::string{"오브젝트"});
        object.parent = entry.value("parent", -1);
        object.transform.position = toVec3(entry.value("position", json{}), glm::vec3{0.0F});
        object.transform.rotation = toQuat(entry.value("rotation", json{}));
        object.transform.scale = toVec3(entry.value("scale", json{}), glm::vec3{1.0F});
        object.visible = entry.value("visible", true);
        object.animator = entry.value("animator", -1);
        object.skin = entry.value("skin", -1);
        object.light = entry.value("light", -1);
        file.scene.objects.push_back(std::move(object));
        file.objectModels.push_back(entry.value("model", -1));
        file.objectLocalMeshes.push_back(entry.value("mesh", 0U));
    }
    return file;
}

} // namespace scene
