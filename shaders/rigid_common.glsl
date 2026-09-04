#ifndef RIGID_COMMON_GLSL
#define RIGID_COMMON_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// 강체 GPU 솔버. src/gfx/rigid_body_gpu.h 의 GpuRigidBody / RigidPushConstants 와 배치가 같아야 한다.
// 접촉 생성은 src/physics/rigid_body.cpp 와 같은 규칙이라 두 백엔드가 같은 접촉을 본다.
//
// CPU 솔버는 접촉을 하나씩 순서대로 푸는 순차 임펄스지만 여기서는 Jacobi 다. 물체마다 다른 물체
// 전부와의 접촉을 «같은 속도로» 풀어 한 번에 더한다. 순서 의존이 없어 나눠 풀 수 있는 대신 수렴이
// 느려 반복을 더 돌고, 쌓인 물체가 CPU 보다 조금 더 물렁하다.

#define RIGID_SHAPE_SPHERE 0u
#define RIGID_SHAPE_BOX 1u
#define RIGID_SHAPE_PLANE 2u

#define RIGID_FLAG_GRAVITY 1u

// 한 짝이 낼 수 있는 접촉 점 수. 나란히 놓인 상자의 면 접촉이 네 점이다.
#define RIGID_MAX_MANIFOLD 4

struct RigidBody {
    // xyz 위치, w 질량 역수.
    vec4 position;
    // 쿼터니언 (x, y, z, w).
    vec4 rotation;
    // xyz 속도, w 콜라이더 반지름.
    vec4 velocity;
    // xyz 각속도, w 경계 반지름.
    vec4 angularVelocity;
    // 반복을 시작하기 전의 속도. 반발 목표를 여기서 재야 첫 반복이 튀긴 것을 두 번째가 도로 당기지 않는다.
    vec4 preVelocity;
    vec4 preAngularVelocity;
    // xyz 지역 축 관성 역수, w 반발 계수.
    vec4 inverseInertia;
    // xyz 상자 반쪽 크기, w 마찰 계수.
    vec4 halfExtents;
    uint shape;
    uint flags;
    uint pad0;
    uint pad1;
};

layout(buffer_reference, scalar) buffer RigidBodyBuffer {
    RigidBody items[];
};

layout(push_constant, scalar) uniform RigidPushConstants {
    RigidBodyBuffer bodiesIn;
    RigidBodyBuffer bodiesOut;
    uint bodyCount;
    float dt;
    float gravity;
    // 위치 보정 비율과 허용 침투. CPU 솔버와 같은 값이어야 한다.
    float positionCorrection;
    float penetrationSlop;
    // 이보다 느리게 닿으면 반발을 주지 않는다.
    float restitutionThreshold;
} push;

vec3 rotateByQuat(vec4 q, vec3 v) {
    vec3 axis = q.xyz;
    return v + 2.0 * cross(axis, cross(axis, v) + q.w * v);
}

vec3 rotateByQuatInverse(vec4 q, vec3 v) {
    return rotateByQuat(vec4(-q.xyz, q.w), v);
}

// 쿼터니언의 회전 행렬. 열이 곧 물체의 지역 축이다.
mat3 quatMatrix(vec4 q) {
    return mat3(1.0) + 2.0 * mat3(-(q.y * q.y + q.z * q.z), q.x * q.y + q.z * q.w, q.x * q.z - q.y * q.w,
                                  q.x * q.y - q.z * q.w, -(q.x * q.x + q.z * q.z), q.y * q.z + q.x * q.w,
                                  q.x * q.z + q.y * q.w, q.y * q.z - q.x * q.w, -(q.x * q.x + q.y * q.y));
}

// 지역 축의 관성 역수를 세계 공간으로 감싼다. R * diag(I) * Rᵀ 다.
mat3 worldInverseInertia(RigidBody body) {
    mat3 rotation = quatMatrix(body.rotation);
    mat3 inertia =
        mat3(body.inverseInertia.x, 0.0, 0.0, 0.0, body.inverseInertia.y, 0.0, 0.0, 0.0, body.inverseInertia.z);
    return rotation * inertia * transpose(rotation);
}

bool rigidDynamic(RigidBody body) {
    return body.position.w > 0.0;
}

// 한 짝의 접촉. 법선은 a 에서 b 를 향하고 점마다 침투가 다르다.
struct RigidManifold {
    vec3 normal;
    vec3 points[RIGID_MAX_MANIFOLD];
    float depths[RIGID_MAX_MANIFOLD];
    int count;
};

RigidManifold noContact() {
    RigidManifold manifold;
    manifold.normal = vec3(0.0, 1.0, 0.0);
    for (int i = 0; i < RIGID_MAX_MANIFOLD; ++i) {
        manifold.points[i] = vec3(0.0);
        manifold.depths[i] = 0.0;
    }
    manifold.count = 0;
    return manifold;
}

// 상한을 넘으면 가장 얕은 것을 밀어낸다. CPU 의 addManifoldPoint 와 같은 규칙이어야 두 백엔드가 같은
// 접촉을 본다. 그냥 버리면 상자가 기울어 깊은 꼭짓점이 뒤쪽 번호일 때 침투를 얕게 본다.
void addPoint(inout RigidManifold manifold, vec3 point, float depth) {
    if (manifold.count < RIGID_MAX_MANIFOLD) {
        manifold.points[manifold.count] = point;
        manifold.depths[manifold.count] = depth;
        ++manifold.count;
        return;
    }
    int shallowest = 0;
    for (int i = 1; i < RIGID_MAX_MANIFOLD; ++i) {
        if (manifold.depths[i] < manifold.depths[shallowest]) {
            shallowest = i;
        }
    }
    if (manifold.depths[shallowest] < depth) {
        manifold.points[shallowest] = point;
        manifold.depths[shallowest] = depth;
    }
}

// 상자 지역 공간에서 점까지의 가장 가까운 표면점과 침투. CPU 의 closestOnBox 와 같은 규칙이다.
// 돌려주는 법선은 상자에서 점을 향한다.
RigidManifold closestOnBox(RigidBody box, vec3 worldPoint, float radius) {
    RigidManifold manifold = noContact();
    vec3 extent = box.halfExtents.xyz;
    vec3 local = rotateByQuatInverse(box.rotation, worldPoint - box.position.xyz);
    vec3 clamped = clamp(local, -extent, extent);
    vec3 delta = local - clamped;
    float distanceSq = dot(delta, delta);
    if (distanceSq > 1.0e-10) {
        // 바깥 점.
        float distance = sqrt(distanceSq);
        if (distance >= radius) {
            return manifold;
        }
        manifold.normal = normalize(rotateByQuat(box.rotation, delta / distance));
        addPoint(manifold, box.position.xyz + rotateByQuat(box.rotation, clamped), radius - distance);
        return manifold;
    }
    // 안쪽 점. 가장 얕은 면으로 나간다.
    vec3 gap = extent - abs(local);
    int axis = gap.x < gap.y ? (gap.x < gap.z ? 0 : 2) : (gap.y < gap.z ? 1 : 2);
    vec3 localNormal = vec3(0.0);
    localNormal[axis] = local[axis] >= 0.0 ? 1.0 : -1.0;
    vec3 surface = local;
    surface[axis] = localNormal[axis] * extent[axis];
    manifold.normal = normalize(rotateByQuat(box.rotation, localNormal));
    addPoint(manifold, box.position.xyz + rotateByQuat(box.rotation, surface), gap[axis] + radius);
    return manifold;
}

// 평면의 법선. 오브젝트의 +Y 를 회전한 것이다.
vec3 rigidPlaneNormal(RigidBody body) {
    return normalize(rotateByQuat(body.rotation, vec3(0.0, 1.0, 0.0)));
}

// 축에 투영한 상자의 반지름.
float boxProjection(mat3 rotation, vec3 extent, vec3 axis) {
    return abs(dot(rotation[0], axis)) * extent.x + abs(dot(rotation[1], axis)) * extent.y +
           abs(dot(rotation[2], axis)) * extent.z;
}

// 상자 대 상자. 여섯 면 축의 분리축 검사로 가장 얕게 겹치는 축을 찾고, 그 축에서 겨루는 면 안에 든
// 상대 꼭짓점을 접촉으로 낸다. 꼭짓점 포함 검사만 하면 크기가 같은 상자가 딱 맞게 겹칠 때 겨루는
// 축이 옆면으로 잘못 잡혀 서로를 그대로 통과한다.
//
// ponytail: 면 축 여섯 개만 본다. 모서리끼리 비스듬히 걸치는 경우(축 아홉 개)를 놓친다.
RigidManifold collideBoxBox(RigidBody a, RigidBody b) {
    RigidManifold manifold = noContact();
    mat3 rotationA = quatMatrix(a.rotation);
    mat3 rotationB = quatMatrix(b.rotation);
    vec3 center = b.position.xyz - a.position.xyz;

    float bestOverlap = 1.0e30;
    vec3 bestNormal = vec3(0.0, 1.0, 0.0);
    bool referenceIsA = true;
    int bestAxis = 1;
    for (int i = 0; i < 6; ++i) {
        bool fromA = i < 3;
        vec3 axis = fromA ? rotationA[i] : rotationB[i - 3];
        float distance = dot(center, axis);
        float overlap = boxProjection(rotationA, a.halfExtents.xyz, axis) +
                        boxProjection(rotationB, b.halfExtents.xyz, axis) - abs(distance);
        if (overlap <= 0.0) {
            return manifold;
        }
        if (overlap < bestOverlap) {
            bestOverlap = overlap;
            // 법선은 a 에서 b 를 향해야 한다.
            bestNormal = distance < 0.0 ? -axis : axis;
            referenceIsA = fromA;
            bestAxis = fromA ? i : i - 3;
        }
    }
    manifold.normal = bestNormal;

    // 겨루는 면을 가진 쪽이 기준, 반대쪽이 입사다. 나란히 놓인 상자면 네 점이 나와 넘어지지 않는다.
    RigidBody reference = referenceIsA ? a : b;
    RigidBody incident = referenceIsA ? b : a;
    mat3 referenceRotation = referenceIsA ? rotationA : rotationB;
    mat3 incidentRotation = referenceIsA ? rotationB : rotationA;
    float axisSign = dot(referenceRotation[bestAxis], bestNormal) < 0.0 ? -1.0 : 1.0;
    float direction = referenceIsA ? 1.0 : -1.0;
    vec3 referenceExtent = reference.halfExtents.xyz;

    for (int i = 0; i < 8; ++i) {
        vec3 sides = vec3((i & 1) != 0 ? 1.0 : -1.0, (i & 2) != 0 ? 1.0 : -1.0, (i & 4) != 0 ? 1.0 : -1.0);
        vec3 corner = incident.position.xyz + incidentRotation * (sides * incident.halfExtents.xyz);
        vec3 local = rotateByQuatInverse(reference.rotation, corner - reference.position.xyz);
        float depth = referenceExtent[bestAxis] - direction * axisSign * local[bestAxis];
        if (depth <= 0.0) {
            continue;
        }
        bool inside = true;
        for (int axis = 0; axis < 3; ++axis) {
            if (axis != bestAxis && abs(local[axis]) > referenceExtent[axis] + push.penetrationSlop) {
                inside = false;
            }
        }
        if (inside) {
            addPoint(manifold, corner, min(depth, bestOverlap));
        }
    }
    if (manifold.count > 0) {
        return manifold;
    }
    // 면 안에 든 꼭짓점이 없으면(모서리끼리 걸침) 가장 깊은 지지점 하나로 대신한다.
    vec3 support = b.position.xyz;
    for (int axis = 0; axis < 3; ++axis) {
        float side = dot(rotationB[axis], bestNormal) < 0.0 ? 1.0 : -1.0;
        support += rotationB[axis] * (side * b.halfExtents.xyz[axis]);
    }
    addPoint(manifold, support, bestOverlap);
    return manifold;
}

// a 와 b 사이의 접촉. 법선은 a 에서 b 를 향한다.
RigidManifold rigidCollide(RigidBody a, RigidBody b) {
    RigidManifold manifold = noContact();

    if (a.shape == RIGID_SHAPE_SPHERE && b.shape == RIGID_SHAPE_SPHERE) {
        vec3 delta = b.position.xyz - a.position.xyz;
        float distance = length(delta);
        float reach = a.velocity.w + b.velocity.w;
        if (distance >= reach) {
            return manifold;
        }
        manifold.normal = distance > 1.0e-6 ? delta / distance : vec3(0.0, 1.0, 0.0);
        addPoint(manifold, a.position.xyz + manifold.normal * a.velocity.w, reach - distance);
        return manifold;
    }

    if (a.shape == RIGID_SHAPE_SPHERE && b.shape == RIGID_SHAPE_PLANE) {
        vec3 normal = rigidPlaneNormal(b);
        float distance = dot(a.position.xyz - b.position.xyz, normal);
        if (distance >= a.velocity.w) {
            return manifold;
        }
        // 법선은 a 에서 b 를 향해야 하므로 평면 법선의 반대다.
        manifold.normal = -normal;
        addPoint(manifold, a.position.xyz - normal * a.velocity.w, a.velocity.w - distance);
        return manifold;
    }

    if (a.shape == RIGID_SHAPE_SPHERE && b.shape == RIGID_SHAPE_BOX) {
        manifold = closestOnBox(b, a.position.xyz, a.velocity.w);
        // closestOnBox 의 법선은 상자에서 구를 향한다. a 가 구이므로 뒤집는다.
        manifold.normal = -manifold.normal;
        return manifold;
    }

    if (a.shape == RIGID_SHAPE_BOX && b.shape == RIGID_SHAPE_PLANE) {
        vec3 normal = rigidPlaneNormal(b);
        mat3 rotationA = quatMatrix(a.rotation);
        manifold.normal = -normal;
        for (int i = 0; i < 8; ++i) {
            vec3 sides = vec3((i & 1) != 0 ? 1.0 : -1.0, (i & 2) != 0 ? 1.0 : -1.0, (i & 4) != 0 ? 1.0 : -1.0);
            vec3 corner = a.position.xyz + rotationA * (sides * a.halfExtents.xyz);
            float distance = dot(corner - b.position.xyz, normal);
            if (distance < 0.0) {
                addPoint(manifold, corner, -distance);
            }
        }
        return manifold;
    }

    if (a.shape == RIGID_SHAPE_BOX && b.shape == RIGID_SHAPE_BOX) {
        return collideBoxBox(a, b);
    }

    return manifold;
}

#endif
