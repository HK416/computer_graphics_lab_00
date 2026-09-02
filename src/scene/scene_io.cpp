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

// 뿌리 안에 있으면 상대 경로로, 밖이면 절대 경로로 적는다. 파일에는 항상 슬래시 형식을 쓴다.
//
// path::native() 는 Windows 에서 std::wstring 이라 좁은 문자열과 섞어 쓸 수 없다. generic_string()
// 으로 한 번 바꿔 두면 비교와 저장이 두 플랫폼에서 같은 코드로 끝난다.
std::string relativeToRoot(const std::filesystem::path& path, const std::filesystem::path& root) {
    if (path.empty()) {
        return {};
    }
    if (root.empty()) {
        return path.generic_string();
    }
    std::error_code error;
    std::filesystem::path relative = std::filesystem::relative(path, root, error);
    if (error || relative.empty()) {
        return path.generic_string();
    }
    std::string generic = relative.generic_string();
    return generic.rfind("../", 0) == 0 || generic == ".." ? path.generic_string() : generic;
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
        modelPaths.push_back(relativeToRoot(path, root));
    }
    document["models"] = modelPaths;

    document["camera"] = {{"position", toJson(scene.camera.position)},
                          {"yaw", scene.camera.yawDegrees},
                          {"pitch", scene.camera.pitchDegrees},
                          {"fovY", scene.camera.fovYDegrees},
                          {"near", scene.camera.nearPlane},
                          {"moveSpeed", scene.camera.moveSpeed},
                          {"mode", scene.camera.mode == CameraMode::FLY ? "fly" : "orbit"},
                          {"target", toJson(scene.camera.target)},
                          {"distance", scene.camera.distance}};
    document["ambient"] = {{"color", toJson(scene.ambientColor)}, {"intensity", scene.ambientIntensity}};
    // HDR 경로도 모델과 같은 규칙으로 적는다.
    std::string hdrPath = relativeToRoot(scene.environment.hdrPath, root);
    document["environment"] = {{"useHdr", scene.environment.useHdr},
                               {"hdr", hdrPath},
                               {"sunColor", toJson(scene.environment.sunColor)},
                               {"sunIntensity", scene.environment.sunIntensity},
                               {"zenith", toJson(scene.environment.zenithColor)},
                               {"horizon", toJson(scene.environment.horizonColor)},
                               {"ground", toJson(scene.environment.groundColor)},
                               {"intensity", scene.environment.intensity},
                               {"yaw", scene.environment.yawDegrees}};
    document["post"] = {{"bloomIntensity", scene.post.bloomIntensity},
                        {"autoExposure", scene.post.autoExposure},
                        {"adaptationSpeed", scene.post.adaptationSpeed},
                        {"exposureMinEv", scene.post.exposureMinEv},
                        {"exposureMaxEv", scene.post.exposureMaxEv},
                        {"fogColor", toJson(scene.post.fogColor)},
                        {"fogDensity", scene.post.fogDensity},
                        {"fogHeight", scene.post.fogHeight},
                        {"fogFalloff", scene.post.fogFalloff}};

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
    for (uint32_t objectIndex = 0; objectIndex < scene.objects.size(); ++objectIndex) {
        const Object& object = scene.objects[objectIndex];
        json entry{{"name", object.name},
                   {"parent", object.parent},
                   {"position", toJson(object.transform.position)},
                   {"rotation", toJson(object.transform.rotation)},
                   {"scale", toJson(object.transform.scale)},
                   {"visible", object.visible}};
        int32_t model = -1;
        uint32_t localMesh = 0;
        uint32_t mesh = scene.meshOf(objectIndex);
        if (mesh != INVALID_MESH && locateMesh(models, mesh, model, localMesh)) {
            entry["model"] = model;
            entry["mesh"] = localMesh;
        }
        if (object.animator >= 0) {
            entry["animator"] = object.animator;
            entry["skin"] = scene.skinOf(objectIndex);
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
    file.scene.camera.mode = camera.value("mode", std::string{"orbit"}) == "fly" ? CameraMode::FLY : CameraMode::ORBIT;
    file.scene.camera.distance = camera.value("distance", 2.5F);
    // 궤도 중심이 없는 옛 파일은 보고 있던 앞쪽 한 점을 중심으로 삼는다. 그냥 기본값을 쓰면
    // 장면을 열자마자 카메라가 뒤로 튄다.
    file.scene.camera.target =
        toVec3(camera.value("target", json{}),
               file.scene.camera.position + file.scene.camera.forward() * file.scene.camera.distance);

    const json& ambient = document.value("ambient", json::object());
    file.scene.ambientColor = toVec3(ambient.value("color", json{}), glm::vec3{0.25F});
    file.scene.ambientIntensity = ambient.value("intensity", 1.0F);

    const json& environment = document.value("environment", json::object());
    Environment& target = file.scene.environment;
    target.useHdr = environment.value("useHdr", false);
    target.hdrPath = environment.value("hdr", std::string{});
    target.sunColor = toVec3(environment.value("sunColor", json{}), target.sunColor);
    target.sunIntensity = environment.value("sunIntensity", target.sunIntensity);
    target.zenithColor = toVec3(environment.value("zenith", json{}), target.zenithColor);
    target.horizonColor = toVec3(environment.value("horizon", json{}), target.horizonColor);
    target.groundColor = toVec3(environment.value("ground", json{}), target.groundColor);
    target.intensity = environment.value("intensity", target.intensity);
    target.yawDegrees = environment.value("yaw", target.yawDegrees);

    // 옛 파일에는 없는 키다. 빠진 값은 기본값을 쓴다.
    const json& post = document.value("post", json::object());
    PostProcess& postTarget = file.scene.post;
    postTarget.bloomIntensity = post.value("bloomIntensity", postTarget.bloomIntensity);
    postTarget.autoExposure = post.value("autoExposure", postTarget.autoExposure);
    postTarget.adaptationSpeed = post.value("adaptationSpeed", postTarget.adaptationSpeed);
    postTarget.exposureMinEv = post.value("exposureMinEv", postTarget.exposureMinEv);
    postTarget.exposureMaxEv = post.value("exposureMaxEv", postTarget.exposureMaxEv);
    postTarget.fogColor = toVec3(post.value("fogColor", json{}), postTarget.fogColor);
    postTarget.fogDensity = post.value("fogDensity", postTarget.fogDensity);
    postTarget.fogHeight = post.value("fogHeight", postTarget.fogHeight);
    postTarget.fogFalloff = post.value("fogFalloff", postTarget.fogFalloff);

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
        object.light = entry.value("light", -1);
        auto skin = entry.value("skin", -1);
        file.scene.objects.push_back(std::move(object));
        file.objectModels.push_back(entry.value("model", -1));
        file.objectLocalMeshes.push_back(entry.value("mesh", 0U));
        // 전역 메쉬 번호는 모델을 올린 뒤에야 정해진다. 부품만 미리 붙여 스킨을 담아 둔다.
        if (file.objectModels.back() >= 0 || skin >= 0) {
            file.scene.attachMeshRenderer(static_cast<uint32_t>(file.scene.objects.size() - 1), INVALID_MESH, skin);
        }
    }
    return file;
}

} // namespace scene
