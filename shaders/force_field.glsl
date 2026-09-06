#ifndef FORCE_FIELD_GLSL
#define FORCE_FIELD_GLSL

// 힘 마당. src/gfx/fluid.h 의 GpuForceField 와 배치가 같고, 식은 src/physics/force_field.h 와 같아야 한다.
// 천·유체·입자 컴퓨트가 같은 함수를 부른다.

#define FORCE_FIELD_MAX 8u
#define FORCE_FIELD_WIND 0u
#define FORCE_FIELD_VORTEX 1u
#define FORCE_FIELD_POINT 2u

struct ForceField {
    vec4 positionRadius; // xyz 중심, w 반지름(0 이면 무한)
    vec4 axisStrength;   // xyz 방향/축, w 세기
    float falloff;
    uint type;
    uint pad0;
    uint pad1;
};

vec3 forceFieldAcceleration(ForceField field, vec3 position) {
    vec3 offset = position - field.positionRadius.xyz;
    float distanceToCenter = length(offset);
    float weight = 1.0;
    if (field.positionRadius.w > 0.0) {
        float t = clamp(1.0 - distanceToCenter / field.positionRadius.w, 0.0, 1.0);
        weight = t <= 0.0 ? 0.0 : pow(t, field.falloff);
    }
    if (weight <= 0.0) {
        return vec3(0.0);
    }
    vec3 axis = field.axisStrength.xyz;
    float strength = field.axisStrength.w;
    if (field.type == FORCE_FIELD_WIND) {
        return axis * (strength * weight);
    }
    if (field.type == FORCE_FIELD_VORTEX) {
        vec3 radial = offset - axis * dot(offset, axis);
        float r = length(radial);
        if (r < 1.0e-4) {
            return vec3(0.0);
        }
        return cross(axis, radial / r) * (strength * weight);
    }
    if (distanceToCenter < 1.0e-4) {
        return vec3(0.0);
    }
    return (-offset / distanceToCenter) * (strength * weight);
}

#endif
