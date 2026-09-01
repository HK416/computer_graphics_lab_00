#version 460

#include "pathtrace_common.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;

void main() {
    payload.missed = true;
    payload.hitDistance = -1.0;
}
