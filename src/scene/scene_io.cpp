#include "scene/scene_io.h"

#include <algorithm>
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
json toJson(const glm::vec4& value) {
    return json::array({value.x, value.y, value.z, value.w});
}

glm::vec3 toVec3(const json& value, const glm::vec3& fallback) {
    if (!value.is_array() || value.size() != 3) {
        return fallback;
    }
    return glm::vec3{value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
}

glm::vec4 toVec4(const json& value, const glm::vec4& fallback) {
    if (!value.is_array() || value.size() != 4) {
        return fallback;
    }
    return glm::vec4{value[0].get<float>(), value[1].get<float>(), value[2].get<float>(), value[3].get<float>()};
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

constexpr std::array<const char*, COLLIDER_SHAPE_COUNT> SHAPE_NAMES{
    "sphere", "box", "plane", "cylinder", "capsule", "mesh"};
constexpr std::array<const char*, 3> BACKEND_NAMES{"auto", "cpu", "gpu"};

SimulationBackend toBackend(const std::string& name) {
    for (uint32_t i = 0; i < BACKEND_NAMES.size(); ++i) {
        if (name == BACKEND_NAMES[i]) {
            return static_cast<SimulationBackend>(i);
        }
    }
    return SimulationBackend::AUTO;
}

ColliderShape toShape(const std::string& name) {
    for (uint32_t i = 0; i < SHAPE_NAMES.size(); ++i) {
        if (name == SHAPE_NAMES[i]) {
            return static_cast<ColliderShape>(i);
        }
    }
    return ColliderShape::SPHERE;
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
                        {"bloomThreshold", scene.post.bloomThreshold},
                        {"bloomKnee", scene.post.bloomKnee},
                        {"bloomScatter", scene.post.bloomScatter},
                        {"autoExposure", scene.post.autoExposure},
                        {"adaptationSpeed", scene.post.adaptationSpeed},
                        {"exposureMinEv", scene.post.exposureMinEv},
                        {"exposureMaxEv", scene.post.exposureMaxEv},
                        {"fogColor", toJson(scene.post.fogColor)},
                        {"fogDensity", scene.post.fogDensity},
                        {"fogHeight", scene.post.fogHeight},
                        {"fogFalloff", scene.post.fogFalloff},
                        {"fogSunScatter", scene.post.fogSunScatter},
                        {"focusDistance", scene.post.focusDistance},
                        {"aperture", scene.post.aperture},
                        {"motionBlur", scene.post.motionBlur}};

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
                             {"speed", animator.speed},
                             {"blendSeconds", animator.blendSeconds}});
    }
    document["animators"] = animators;

    // 강체는 설정만 적고 속도는 적지 않는다. 재생을 멈추면 어차피 되돌아가는 상태다.
    json rigidBodies = json::array();
    for (const RigidBody& body : scene.rigidBodies) {
        rigidBodies.push_back({{"backend", BACKEND_NAMES[static_cast<size_t>(body.backend)]},
                               {"shape", SHAPE_NAMES[static_cast<size_t>(body.shape)]},
                               {"mass", body.mass},
                               {"useGravity", body.useGravity},
                               {"kinematic", body.kinematic},
                               {"restitution", body.restitution},
                               {"friction", body.friction},
                               {"radius", body.radius},
                               {"halfExtents", toJson(body.halfExtents)}});
    }
    document["rigidBodies"] = rigidBodies;

    json fluids = json::array();
    for (const Fluid& fluid : scene.fluids) {
        fluids.push_back({{"backend", BACKEND_NAMES[static_cast<size_t>(fluid.backend)]},
                          {"emitterHalfExtents", toJson(fluid.emitterHalfExtents)},
                          {"particleCount", fluid.particleCount},
                          {"particleRadius", fluid.particleRadius},
                          {"smoothingRadius", fluid.smoothingRadius},
                          {"restDensity", fluid.restDensity},
                          {"stiffness", fluid.stiffness},
                          {"viscosity", fluid.viscosity},
                          {"containerMin", toJson(fluid.containerMin)},
                          {"containerMax", toJson(fluid.containerMax)},
                          {"gravity", toJson(fluid.gravity)},
                          {"display", fluid.display == FluidDisplay::SURFACE ? "surface" : "particles"},
                          {"surfaceResolution", fluid.surfaceResolution},
                          {"surfaceIso", fluid.surfaceIso},
                          {"surfaceRoughness", fluid.surfaceRoughness},
                          {"waterColor", toJson(fluid.waterColor)},
                          {"absorption", toJson(fluid.absorption)},
                          {"thicknessScale", fluid.thicknessScale}});
    }
    document["fluids"] = fluids;

    json particleSystems = json::array();
    for (const ParticleSystem& system : scene.particleSystems) {
        particleSystems.push_back({{"maxParticles", system.maxParticles},
                                   {"emitRate", system.emitRate},
                                   {"lifetime", system.lifetime},
                                   {"initialSpeed", system.initialSpeed},
                                   {"spreadAngleDegrees", system.spreadAngleDegrees},
                                   {"gravityScale", system.gravityScale},
                                   {"drag", system.drag},
                                   {"sizeStart", system.sizeStart},
                                   {"sizeEnd", system.sizeEnd},
                                   {"color", toJson(system.color)},
                                   {"emissive", toJson(system.emissive)},
                                   {"restitution", system.restitution},
                                   {"collide", system.collide}});
    }
    document["particleSystems"] = particleSystems;

    json cloths = json::array();
    for (const Cloth& cloth : scene.cloths) {
        constexpr std::array<const char*, 3> PIN_NAMES{"none", "topEdge", "twoCorners"};
        cloths.push_back({{"backend", BACKEND_NAMES[static_cast<size_t>(cloth.backend)]},
                          {"resolution", cloth.resolution},
                          {"mass", cloth.mass},
                          {"stretchCompliance", cloth.stretchCompliance},
                          {"shearCompliance", cloth.shearCompliance},
                          {"bendCompliance", cloth.bendCompliance},
                          {"damping", cloth.damping},
                          {"substeps", cloth.substeps},
                          {"iterations", cloth.iterations},
                          {"pin", PIN_NAMES[std::min(static_cast<size_t>(cloth.pin), PIN_NAMES.size() - 1)]},
                          {"thickness", cloth.thickness},
                          {"friction", cloth.friction},
                          {"gravity", toJson(cloth.gravity)},
                          {"wind", toJson(cloth.wind)}});
    }
    document["cloths"] = cloths;

    json forceFields = json::array();
    for (const ForceField& field : scene.forceFields) {
        constexpr std::array<const char*, 3> TYPE_NAMES{"wind", "vortex", "point"};
        forceFields.push_back({{"type", TYPE_NAMES[std::min(static_cast<size_t>(field.type), TYPE_NAMES.size() - 1)]},
                               {"strength", field.strength},
                               {"radius", field.radius},
                               {"falloff", field.falloff}});
    }
    document["forceFields"] = forceFields;

    json ddgiVolumes = json::array();
    for (const DdgiVolume& volume : scene.ddgiVolumes) {
        ddgiVolumes.push_back({{"probes", volume.probes}, {"enabled", volume.enabled}});
    }
    document["ddgiVolumes"] = ddgiVolumes;

    json cameraComponents = json::array();
    for (const CameraComponent& camera : scene.cameraComponents) {
        cameraComponents.push_back({{"fovY", camera.fovYDegrees},
                                    {"near", camera.nearPlane},
                                    {"active", camera.active},
                                    {"observation", camera.observation}});
    }
    document["cameraComponents"] = cameraComponents;

    json cameraPaths = json::array();
    for (const CameraPath& path : scene.cameraPaths) {
        json keys = json::array();
        for (const CameraKey& key : path.keys) {
            keys.push_back({{"position", toJson(key.position)}, {"rotation", toJson(key.rotation)}});
        }
        cameraPaths.push_back({{"duration", path.duration}, {"loop", path.loop}, {"keys", keys}});
    }
    document["cameraPaths"] = cameraPaths;

    json joints = json::array();
    for (const Joint& joint : scene.joints) {
        constexpr std::array<const char*, 3> JOINT_NAMES{"distance", "ball", "hinge"};
        constexpr std::array<const char*, 3> MOTOR_NAMES{"none", "velocity", "position"};
        joints.push_back({{"type", JOINT_NAMES[std::min(static_cast<size_t>(joint.type), JOINT_NAMES.size() - 1)]},
                          {"other", joint.other},
                          {"anchorA", toJson(joint.anchorA)},
                          {"anchorB", toJson(joint.anchorB)},
                          {"axis", toJson(joint.axis)},
                          {"length", joint.length},
                          {"useLimit", joint.useLimit},
                          {"lowerAngle", joint.lowerAngle},
                          {"upperAngle", joint.upperAngle},
                          {"motor", MOTOR_NAMES[std::min(static_cast<size_t>(joint.motor), MOTOR_NAMES.size() - 1)]},
                          {"targetAngle", joint.targetAngle},
                          {"targetSpeed", joint.targetSpeed},
                          {"motorStiffness", joint.motorStiffness},
                          {"maxTorque", joint.maxTorque}});
    }
    document["joints"] = joints;

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
        if (object.rigidBody >= 0) {
            entry["rigidBody"] = object.rigidBody;
        }
        if (object.fluid >= 0) {
            entry["fluid"] = object.fluid;
        }
        if (object.particleSystem >= 0) {
            entry["particleSystem"] = object.particleSystem;
        }
        if (object.cloth >= 0) {
            entry["cloth"] = object.cloth;
        }
        if (object.forceField >= 0) {
            entry["forceField"] = object.forceField;
        }
        if (object.ddgiVolume >= 0) {
            entry["ddgiVolume"] = object.ddgiVolume;
        }
        if (object.cameraComponent >= 0) {
            entry["cameraComponent"] = object.cameraComponent;
        }
        if (object.cameraPath >= 0) {
            entry["cameraPath"] = object.cameraPath;
        }
        if (object.joint >= 0) {
            entry["joint"] = object.joint;
        }
        objects.push_back(std::move(entry));
    }
    document["objects"] = objects;

    return document.dump(2);
}

std::string writeSubtree(const Scene& scene,
                         const std::vector<uint32_t>& roots,
                         const ModelTable& models,
                         const std::filesystem::path& root) {
    // 사본에서 서브트리 밖을 지우면 removeObjects 가 부모·부품·관절 첨자를 알아서 민다. 뿌리를 먼저 떼어 두어야
    // «부모가 지워지면 자식도 지운다» 규칙에 뿌리가 딸려 가지 않는다.
    Scene subtree = scene;
    std::vector<bool> kept(scene.objects.size(), false);
    for (uint32_t rootIndex : roots) {
        if (rootIndex >= scene.objects.size()) {
            continue;
        }
        kept[rootIndex] = true;
        subtree.objects[rootIndex].parent = -1;
        for (uint32_t i = 0; i < scene.objects.size(); ++i) {
            if (scene.isDescendant(i, rootIndex)) {
                kept[i] = true;
            }
        }
    }
    std::vector<uint32_t> doomed;
    for (uint32_t i = 0; i < kept.size(); ++i) {
        if (!kept[i]) {
            doomed.push_back(i);
        }
    }
    subtree.removeObjects(doomed);
    subtree.name = roots.size() == 1 && roots[0] < scene.objects.size() ? scene.objects[roots[0]].name : "프리팹";
    return writeScene(subtree, models, root);
}

uint32_t appendScene(Scene& target, const Scene& source, int32_t parent) {
    target.markStructureDirty();
    auto base = static_cast<uint32_t>(target.objects.size());
    if (parent >= static_cast<int32_t>(base)) {
        parent = -1;
    }
    std::vector<Object> appended = source.objects;
    for (Object& object : appended) {
        object.parent = object.parent >= 0 ? object.parent + static_cast<int32_t>(base) : parent;
    }
    // 부품 종류마다 배열을 이어 붙이고 새 오브젝트의 첨자를 그만큼 민다.
    forEachComponentKind(target, [&](auto& items, int32_t Object::* kind) {
        using Item = typename std::remove_reference_t<decltype(items)>::value_type;
        const auto& sourceItems = ComponentSlot<Item>::items(source);
        auto offset = static_cast<int32_t>(items.size());
        items.insert(items.end(), sourceItems.begin(), sourceItems.end());
        for (Object& object : appended) {
            if (object.*kind >= 0) {
                object.*kind += offset;
            }
        }
    });
    // 관절의 상대 번호는 오브젝트 번호라 함께 민다.
    for (size_t i = target.joints.size() - source.joints.size(); i < target.joints.size(); ++i) {
        Joint& joint = target.joints[i];
        joint.other = joint.other >= 0 ? joint.other + static_cast<int32_t>(base) : -1;
    }
    target.objects.insert(target.objects.end(), appended.begin(), appended.end());
    return base;
}

SceneFile readScene(const std::string& text) {
    json document = json::parse(text, nullptr, false);
    if (document.is_discarded() || !document.is_object()) {
        core::fatal("장면 파일을 해석할 수 없습니다");
    }
    // 옛 판은 빠진 키를 기본값으로 읽는다. 모르는(더 새로운) 판만 거절한다.
    auto version = document.value("version", 0U);
    if (version == 0 || version > SCENE_FILE_VERSION) {
        core::fatal("지원하지 않는 장면 파일 판입니다: {} (필요: {} 이하)", version, SCENE_FILE_VERSION);
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
    // 판 3 부터 bloomIntensity 는 «섞는 비율»이 아니라 «더하는 세기»다. 옛 값을 그대로 읽으면
    // 번짐이 거의 보이지 않으므로 새 기본값을 쓴다. 다만 0 은 뜻이 같은 «끔»이라 그대로 둔다.
    float storedBloom = post.value("bloomIntensity", postTarget.bloomIntensity);
    if (version >= 3 || storedBloom == 0.0F) {
        postTarget.bloomIntensity = storedBloom;
    }
    postTarget.bloomThreshold = post.value("bloomThreshold", postTarget.bloomThreshold);
    postTarget.bloomKnee = post.value("bloomKnee", postTarget.bloomKnee);
    postTarget.bloomScatter = post.value("bloomScatter", postTarget.bloomScatter);
    postTarget.autoExposure = post.value("autoExposure", postTarget.autoExposure);
    postTarget.adaptationSpeed = post.value("adaptationSpeed", postTarget.adaptationSpeed);
    postTarget.exposureMinEv = post.value("exposureMinEv", postTarget.exposureMinEv);
    postTarget.exposureMaxEv = post.value("exposureMaxEv", postTarget.exposureMaxEv);
    postTarget.fogColor = toVec3(post.value("fogColor", json{}), postTarget.fogColor);
    postTarget.fogDensity = post.value("fogDensity", postTarget.fogDensity);
    postTarget.fogHeight = post.value("fogHeight", postTarget.fogHeight);
    postTarget.fogFalloff = post.value("fogFalloff", postTarget.fogFalloff);
    postTarget.fogSunScatter = post.value("fogSunScatter", postTarget.fogSunScatter);
    postTarget.focusDistance = post.value("focusDistance", postTarget.focusDistance);
    postTarget.aperture = post.value("aperture", postTarget.aperture);
    postTarget.motionBlur = post.value("motionBlur", postTarget.motionBlur);

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
        animator.blendSeconds = entry.value("blendSeconds", 0.3F);
        animator.model = entry.value("model", -1);
        file.animatorModels.push_back(animator.model);
        file.scene.animators.push_back(std::move(animator));
    }

    for (const json& entry : document.value("rigidBodies", json::array())) {
        RigidBody body;
        body.backend = toBackend(entry.value("backend", std::string{"auto"}));
        body.shape = toShape(entry.value("shape", std::string{"sphere"}));
        body.mass = entry.value("mass", body.mass);
        body.useGravity = entry.value("useGravity", body.useGravity);
        body.kinematic = entry.value("kinematic", body.kinematic);
        body.restitution = entry.value("restitution", body.restitution);
        body.friction = entry.value("friction", body.friction);
        body.radius = entry.value("radius", body.radius);
        body.halfExtents = toVec3(entry.value("halfExtents", json{}), body.halfExtents);
        file.scene.rigidBodies.push_back(body);
    }

    for (const json& entry : document.value("fluids", json::array())) {
        Fluid fluid;
        fluid.backend = toBackend(entry.value("backend", std::string{"auto"}));
        fluid.emitterHalfExtents = toVec3(entry.value("emitterHalfExtents", json{}), fluid.emitterHalfExtents);
        fluid.particleCount = entry.value("particleCount", fluid.particleCount);
        fluid.particleRadius = entry.value("particleRadius", fluid.particleRadius);
        fluid.smoothingRadius = entry.value("smoothingRadius", fluid.smoothingRadius);
        fluid.restDensity = entry.value("restDensity", fluid.restDensity);
        fluid.stiffness = entry.value("stiffness", fluid.stiffness);
        fluid.viscosity = entry.value("viscosity", fluid.viscosity);
        fluid.containerMin = toVec3(entry.value("containerMin", json{}), fluid.containerMin);
        fluid.containerMax = toVec3(entry.value("containerMax", json{}), fluid.containerMax);
        fluid.gravity = toVec3(entry.value("gravity", json{}), fluid.gravity);
        fluid.display = entry.value("display", std::string{"particles"}) == "surface" ? FluidDisplay::SURFACE
                                                                                      : FluidDisplay::PARTICLES;
        // 상한은 gfx::FLUID_MAX_SURFACE_RESOLUTION 과 같은 값이다. scene 은 gfx 를 보지 못해 손으로
        // 옮겨 적었다. 렌더러가 다시 한 번 묶으므로 여기서는 터무니없는 값만 걸러 낸다.
        fluid.surfaceResolution = std::clamp(entry.value("surfaceResolution", fluid.surfaceResolution), 8U, 128U);
        fluid.surfaceIso = entry.value("surfaceIso", fluid.surfaceIso);
        fluid.surfaceRoughness = entry.value("surfaceRoughness", fluid.surfaceRoughness);
        fluid.waterColor = toVec3(entry.value("waterColor", json{}), fluid.waterColor);
        fluid.absorption = toVec3(entry.value("absorption", json{}), fluid.absorption);
        fluid.thicknessScale = entry.value("thicknessScale", fluid.thicknessScale);
        file.scene.fluids.push_back(fluid);
    }

    for (const json& entry : document.value("particleSystems", json::array())) {
        ParticleSystem system;
        // 상한은 gfx::PARTICLE_MAX_PARTICLES 와 같은 값이다. scene 은 gfx 를 보지 못해 손으로 옮겨 적었다.
        system.maxParticles = std::clamp(entry.value("maxParticles", system.maxParticles), 1U, 65536U);
        system.emitRate = entry.value("emitRate", system.emitRate);
        system.lifetime = entry.value("lifetime", system.lifetime);
        system.initialSpeed = entry.value("initialSpeed", system.initialSpeed);
        system.spreadAngleDegrees = entry.value("spreadAngleDegrees", system.spreadAngleDegrees);
        system.gravityScale = entry.value("gravityScale", system.gravityScale);
        system.drag = entry.value("drag", system.drag);
        system.sizeStart = entry.value("sizeStart", system.sizeStart);
        system.sizeEnd = entry.value("sizeEnd", system.sizeEnd);
        system.color = toVec4(entry.value("color", json{}), system.color);
        system.emissive = toVec3(entry.value("emissive", json{}), system.emissive);
        system.restitution = entry.value("restitution", system.restitution);
        system.collide = entry.value("collide", system.collide);
        file.scene.particleSystems.push_back(system);
    }

    for (const json& entry : document.value("forceFields", json::array())) {
        ForceField field;
        std::string type = entry.value("type", std::string{"wind"});
        field.type = type == "vortex"  ? ForceFieldType::VORTEX
                     : type == "point" ? ForceFieldType::POINT
                                       : ForceFieldType::WIND;
        field.strength = entry.value("strength", field.strength);
        field.radius = std::max(entry.value("radius", field.radius), 0.0F);
        field.falloff = std::max(entry.value("falloff", field.falloff), 0.0F);
        file.scene.forceFields.push_back(field);
    }

    for (const json& entry : document.value("ddgiVolumes", json::array())) {
        DdgiVolume volume;
        volume.probes = std::clamp(entry.value("probes", volume.probes), 2U, 16U);
        volume.enabled = entry.value("enabled", volume.enabled);
        file.scene.ddgiVolumes.push_back(volume);
    }

    for (const json& entry : document.value("cameraComponents", json::array())) {
        CameraComponent camera;
        camera.fovYDegrees = std::clamp(entry.value("fovY", camera.fovYDegrees), 1.0F, 179.0F);
        camera.nearPlane = std::max(entry.value("near", camera.nearPlane), 1.0e-4F);
        camera.active = entry.value("active", camera.active);
        // 옛 장면 파일에는 이 항목이 없다. 없으면 거짓이라 지금까지처럼 화면 카메라로 남는다.
        camera.observation = entry.value("observation", camera.observation);
        file.scene.cameraComponents.push_back(camera);
    }

    for (const json& entry : document.value("cameraPaths", json::array())) {
        CameraPath path;
        path.duration = std::max(entry.value("duration", path.duration), 0.0F);
        path.loop = entry.value("loop", path.loop);
        for (const json& keyEntry : entry.value("keys", json::array())) {
            CameraKey key;
            key.position = toVec3(keyEntry.value("position", json{}), glm::vec3{0.0F});
            key.rotation = toQuat(keyEntry.value("rotation", json{}));
            path.keys.push_back(key);
        }
        file.scene.cameraPaths.push_back(path);
    }

    for (const json& entry : document.value("joints", json::array())) {
        Joint joint;
        std::string type = entry.value("type", std::string{"ball"});
        joint.type = type == "distance" ? JointType::DISTANCE : type == "hinge" ? JointType::HINGE : JointType::BALL;
        joint.other = entry.value("other", -1);
        joint.anchorA = toVec3(entry.value("anchorA", json{}), glm::vec3{0.0F});
        joint.anchorB = toVec3(entry.value("anchorB", json{}), glm::vec3{0.0F});
        joint.axis = toVec3(entry.value("axis", json{}), glm::vec3{0.0F, 1.0F, 0.0F});
        joint.length = std::max(entry.value("length", joint.length), 0.0F);
        joint.useLimit = entry.value("useLimit", joint.useLimit);
        joint.lowerAngle = entry.value("lowerAngle", joint.lowerAngle);
        joint.upperAngle = entry.value("upperAngle", joint.upperAngle);
        std::string motor = entry.value("motor", std::string{"none"});
        joint.motor = motor == "velocity"   ? JointMotor::VELOCITY
                      : motor == "position" ? JointMotor::POSITION
                                            : JointMotor::NONE;
        joint.targetAngle = entry.value("targetAngle", joint.targetAngle);
        joint.targetSpeed = entry.value("targetSpeed", joint.targetSpeed);
        joint.motorStiffness = std::max(entry.value("motorStiffness", joint.motorStiffness), 0.0F);
        joint.maxTorque = std::max(entry.value("maxTorque", joint.maxTorque), 0.0F);
        file.scene.joints.push_back(joint);
    }

    for (const json& entry : document.value("cloths", json::array())) {
        Cloth cloth;
        cloth.backend = toBackend(entry.value("backend", std::string{"auto"}));
        // 내장 격자는 16·32·64 뿐이다. 다른 값은 가장 가까운 것으로 접는다.
        uint32_t resolution = entry.value("resolution", cloth.resolution);
        cloth.resolution = resolution <= 16 ? 16U : (resolution <= 32 ? 32U : 64U);
        cloth.mass = std::max(entry.value("mass", cloth.mass), 1.0e-3F);
        cloth.stretchCompliance = std::max(entry.value("stretchCompliance", cloth.stretchCompliance), 0.0F);
        cloth.shearCompliance = std::max(entry.value("shearCompliance", cloth.shearCompliance), 0.0F);
        cloth.bendCompliance = std::max(entry.value("bendCompliance", cloth.bendCompliance), 0.0F);
        cloth.damping = entry.value("damping", cloth.damping);
        cloth.substeps = std::clamp(entry.value("substeps", cloth.substeps), 1U, 16U);
        cloth.iterations = std::clamp(entry.value("iterations", cloth.iterations), 1U, 32U);
        std::string pin = entry.value("pin", std::string{"topEdge"});
        cloth.pin = pin == "none" ? ClothPin::NONE : (pin == "twoCorners" ? ClothPin::TWO_CORNERS : ClothPin::TOP_EDGE);
        cloth.thickness = std::max(entry.value("thickness", cloth.thickness), 0.0F);
        cloth.friction = std::clamp(entry.value("friction", cloth.friction), 0.0F, 1.0F);
        cloth.gravity = toVec3(entry.value("gravity", json{}), cloth.gravity);
        cloth.wind = toVec3(entry.value("wind", json{}), cloth.wind);
        file.scene.cloths.push_back(cloth);
    }

    // 손으로 고친 파일이나 깨진 파일이 배열 밖을 가리킬 수 있다. 없는 부품은 안 붙은 것으로 본다.
    auto handle = [](int32_t value, size_t size) {
        return value >= 0 && static_cast<size_t>(value) < size ? value : -1;
    };
    for (const json& entry : document.value("objects", json::array())) {
        Object object;
        object.name = entry.value("name", std::string{"오브젝트"});
        object.parent = entry.value("parent", -1);
        object.transform.position = toVec3(entry.value("position", json{}), glm::vec3{0.0F});
        object.transform.rotation = toQuat(entry.value("rotation", json{}));
        object.transform.scale = toVec3(entry.value("scale", json{}), glm::vec3{1.0F});
        object.visible = entry.value("visible", true);
        object.animator = handle(entry.value("animator", -1), file.scene.animators.size());
        object.light = handle(entry.value("light", -1), file.scene.lights.size());
        object.rigidBody = handle(entry.value("rigidBody", -1), file.scene.rigidBodies.size());
        object.fluid = handle(entry.value("fluid", -1), file.scene.fluids.size());
        object.particleSystem = handle(entry.value("particleSystem", -1), file.scene.particleSystems.size());
        object.cloth = handle(entry.value("cloth", -1), file.scene.cloths.size());
        object.forceField = handle(entry.value("forceField", -1), file.scene.forceFields.size());
        object.ddgiVolume = handle(entry.value("ddgiVolume", -1), file.scene.ddgiVolumes.size());
        object.cameraComponent = handle(entry.value("cameraComponent", -1), file.scene.cameraComponents.size());
        object.cameraPath = handle(entry.value("cameraPath", -1), file.scene.cameraPaths.size());
        object.joint = handle(entry.value("joint", -1), file.scene.joints.size());
        auto skin = entry.value("skin", -1);
        file.scene.objects.push_back(std::move(object));
        file.objectModels.push_back(entry.value("model", -1));
        file.objectLocalMeshes.push_back(entry.value("mesh", 0U));
        // 전역 메쉬 번호는 모델을 올린 뒤에야 정해진다. 부품만 미리 붙여 스킨을 담아 둔다.
        if (file.objectModels.back() >= 0 || skin >= 0) {
            file.scene.attachMeshRenderer(static_cast<uint32_t>(file.scene.objects.size() - 1), INVALID_MESH, skin);
        }
    }

    // 부모는 배열에서 뒤에 올 수도 있어 다 읽은 뒤에 자른다. 범위 밖을 가리키면 세계 변환이 배열
    // 밖을 짚는다.
    for (Object& object : file.scene.objects) {
        object.parent = handle(object.parent, file.scene.objects.size());
    }
    return file;
}

} // namespace scene
