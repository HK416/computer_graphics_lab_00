#ifndef RIGID_COMMON_GLSL
#define RIGID_COMMON_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "collider_shapes.glsl"
#include "spatial_hash.glsl"

// 강체 GPU 솔버. src/gfx/rigid_body_gpu.h 의 GpuRigidBody / RigidPushConstants 와 배치가 같아야 한다.
// 접촉 생성은 src/physics/rigid_body.cpp 와 같은 규칙이라 두 백엔드가 같은 접촉을 본다. 광역도 같은 해시
// 격자(spatial_hash.glsl ↔ physics/spatial_hash.h, physics::collectPairs)라 같은 짝을 본다.
//
// CPU 솔버는 접촉을 하나씩 순서대로 푸는 순차 임펄스지만 여기서는 Jacobi 다. 물체마다 다른 물체
// 전부와의 접촉을 «같은 속도로» 풀어 한 번에 더한다. 순서 의존이 없어 나눠 풀 수 있는 대신 수렴이
// 느려 반복을 더 돌고, 쌓인 물체가 CPU 보다 조금 더 물렁하다.

#define RIGID_SHAPE_SPHERE COLLIDER_SHAPE_SPHERE
#define RIGID_SHAPE_BOX COLLIDER_SHAPE_BOX
#define RIGID_SHAPE_PLANE COLLIDER_SHAPE_PLANE
#define RIGID_SHAPE_CYLINDER COLLIDER_SHAPE_CYLINDER
#define RIGID_SHAPE_CAPSULE COLLIDER_SHAPE_CAPSULE
#define RIGID_SHAPE_MESH COLLIDER_SHAPE_MESH

#define RIGID_FLAG_GRAVITY 1u

// 한 짝이 낼 수 있는 접촉 점 수. 나란히 놓인 상자의 면 접촉이 네 점이다.
#define RIGID_MAX_MANIFOLD 4
// 광역 격자 버킷 용량(gfx::RIGID_CELL_CAPACITY 와 같아야 한다)과 물체 하나가 모을 수 있는 이웃 후보 수.
#define RIGID_CELL_CAPACITY 64
#define RIGID_MAX_NEIGHBORS 64

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
    // xyz 상자 반쪽 크기(원기둥·캡슐은 y 가 반높이), w 마찰 계수.
    vec4 halfExtents;
    uint shape;
    uint flags;
    // 메쉬 콜라이더의 세계 공간 삼각형 구간(push.triangles 기준).
    uint triangleOffset;
    uint triangleCount;
};

layout(buffer_reference, scalar) buffer RigidBodyBuffer {
    RigidBody items[];
};

// 메쉬 콜라이더의 세계 공간 삼각형. 앞면은 CCW.
struct RigidTriangle {
    vec3 a;
    vec3 b;
    vec3 c;
};

layout(buffer_reference, scalar) buffer RigidTriangleBuffer {
    RigidTriangle items[];
};

layout(buffer_reference, scalar) buffer RigidIndexBuffer {
    uint items[];
};

layout(push_constant, scalar) uniform RigidPushConstants {
    RigidBodyBuffer bodiesIn;
    RigidBodyBuffer bodiesOut;
    RigidTriangleBuffer triangles;
    // 광역 격자: 버킷마다의 개수와 번호(cell * RIGID_CELL_CAPACITY + slot), 그리고 평면 번호 목록.
    RigidIndexBuffer cellCounts;
    RigidIndexBuffer cellBodies;
    RigidIndexBuffer planes;
    uint bodyCount;
    float dt;
    float gravity;
    // 위치 보정 비율과 허용 침투. CPU 솔버와 같은 값이어야 한다.
    float positionCorrection;
    float penetrationSlop;
    // 이보다 느리게 닿으면 반발을 주지 않는다.
    float restitutionThreshold;
    uint cellCount;
    uint planeCount;
    float cellSize;
} push;

// 광역: self 와 경계 구가 겹치는 물체를 모은다. 이웃 27 셀의 버킷을 훑되, 해시 충돌로 두 셀이 같은 버킷에
// 떨어지면 두 번째는 건너뛴다(한 물체가 두 번 나오지 않게). 버킷에 섞인 먼 셀의 물체는 경계 구 검사(CPU
// collectPairs 와 같은 식)가 거른다. 격자는 적분 직후 위치로 지었고 위치 보정이 셀 경계를 넘길 수 있으므로
// 물체의 «지금 셀» 을 버킷 셀과 견주지 않는다. 평면은 무한이라 목록에서 그대로 이어 붙인다. 마지막에 번호
// 오름차순으로 정렬해야 O(n²) 루프와 같은 순서로 누적되어 부동소수 결과가 그대로다. 상한을 넘는 이웃은
// 버린다(rigid_grid.comp 의 표식).
uint rigidGatherNeighbors(uint self, vec3 position, float boundingRadius, out uint neighbors[RIGID_MAX_NEIGHBORS]) {
    uint count = 0u;
    ivec3 base = spatialCell(position, push.cellSize);
    uint visited[27];
    uint visitedCount = 0u;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint bucket = spatialHash(base + ivec3(dx, dy, dz), push.cellCount);
                bool seen = false;
                for (uint v = 0u; v < visitedCount; ++v) {
                    seen = seen || visited[v] == bucket;
                }
                if (seen) {
                    continue;
                }
                visited[visitedCount++] = bucket;
                uint filled = min(push.cellCounts.items[bucket], uint(RIGID_CELL_CAPACITY));
                for (uint slot = 0u; slot < filled; ++slot) {
                    uint j = push.cellBodies.items[bucket * RIGID_CELL_CAPACITY + slot];
                    if (j == self) {
                        continue;
                    }
                    vec3 otherPosition = push.bodiesIn.items[j].position.xyz;
                    vec3 delta = otherPosition - position;
                    float reach = boundingRadius + push.bodiesIn.items[j].angularVelocity.w;
                    if (dot(delta, delta) > reach * reach) {
                        continue;
                    }
                    if (count < RIGID_MAX_NEIGHBORS) {
                        neighbors[count++] = j;
                    }
                }
            }
        }
    }
    for (uint p = 0u; p < push.planeCount; ++p) {
        uint j = push.planes.items[p];
        if (j != self && count < RIGID_MAX_NEIGHBORS) {
            neighbors[count++] = j;
        }
    }
    // 삽입 정렬. 후보가 수십 개라 이걸로 충분하다.
    for (uint a = 1u; a < count; ++a) {
        uint value = neighbors[a];
        uint b = a;
        while (b > 0u && neighbors[b - 1u] > value) {
            neighbors[b] = neighbors[b - 1u];
            --b;
        }
        neighbors[b] = value;
    }
    return count;
}

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

// 한 짝의 접촉. 법선은 a 에서 b 를 향하고 점마다 침투와 법선이 다를 수 있다(표본 기반 접촉).
struct RigidManifold {
    // 새 점의 기본 법선. 상자·구 짝은 모든 점이 이것을 쓴다.
    vec3 normal;
    vec3 points[RIGID_MAX_MANIFOLD];
    vec3 normals[RIGID_MAX_MANIFOLD];
    float depths[RIGID_MAX_MANIFOLD];
    int count;
};

RigidManifold noContact() {
    RigidManifold manifold;
    manifold.normal = vec3(0.0, 1.0, 0.0);
    for (int i = 0; i < RIGID_MAX_MANIFOLD; ++i) {
        manifold.points[i] = vec3(0.0);
        manifold.normals[i] = vec3(0.0, 1.0, 0.0);
        manifold.depths[i] = 0.0;
    }
    manifold.count = 0;
    return manifold;
}

// 상한을 넘으면 가장 얕은 것을 밀어낸다. CPU 의 addManifoldPoint 와 같은 규칙이어야 두 백엔드가 같은
// 접촉을 본다. 그냥 버리면 상자가 기울어 깊은 꼭짓점이 뒤쪽 번호일 때 침투를 얕게 본다.
void addPointNormal(inout RigidManifold manifold, vec3 point, vec3 normal, float depth) {
    if (manifold.count < RIGID_MAX_MANIFOLD) {
        manifold.points[manifold.count] = point;
        manifold.normals[manifold.count] = normal;
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
        manifold.normals[shallowest] = normal;
        manifold.depths[shallowest] = depth;
    }
}

void addPoint(inout RigidManifold manifold, vec3 point, float depth) {
    addPointNormal(manifold, point, manifold.normal, depth);
}

// 법선을 모두 뒤집는다. a·b 순서를 바꿔 만든 접촉을 되돌릴 때 쓴다.
RigidManifold flipManifold(RigidManifold manifold) {
    manifold.normal = -manifold.normal;
    for (int i = 0; i < RIGID_MAX_MANIFOLD; ++i) {
        manifold.normals[i] = -manifold.normals[i];
    }
    return manifold;
}

// 모든 점의 법선을 하나로 맞춘다. 중심이 겹쳐 법선을 정할 근거가 없을 때 쓴다.
RigidManifold overrideNormal(RigidManifold manifold, vec3 normal) {
    manifold.normal = normal;
    for (int i = 0; i < RIGID_MAX_MANIFOLD; ++i) {
        manifold.normals[i] = normal;
    }
    return manifold;
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

// 세계 점에서 물체 표면의 가장 가까운 점. collider_shapes.glsl 의 지역 함수를 쿼터니언으로 감싼다.
SurfacePoint closestOnBody(RigidBody body, vec3 worldPoint) {
    vec3 local = rotateByQuatInverse(body.rotation, worldPoint - body.position.xyz);
    SurfacePoint result = closestOnColliderLocal(body.shape, body.velocity.w, body.halfExtents.xyz, local);
    result.point = body.position.xyz + rotateByQuat(body.rotation, result.point);
    result.normal = rotateByQuat(body.rotation, result.normal);
    return result;
}

// 상대 표면에서 자기 중심에 가장 가까운 점을 자기 지역 공간으로. probePointLocal 의 힌트다.
vec3 probeHint(RigidBody self, RigidBody other) {
    SurfacePoint near = closestOnBody(other, self.position.xyz);
    return rotateByQuatInverse(self.rotation, near.point - self.position.xyz);
}

// 원기둥·캡슐이 낀 짝. 표본점을 서로에게 찔러 깊이가 양수인 것을 접촉으로 낸다. CPU 의 collideGeneric
// 과 같은 규칙이다.
RigidManifold rigidCollideGeneric(RigidBody a, RigidBody b) {
    RigidManifold manifold = noContact();
    vec3 hintA = probeHint(a, b);
    int countA = probeCount(a.shape);
    for (int i = 0; i < countA; ++i) {
        float radius;
        vec3 local = probePointLocal(a.shape, a.velocity.w, a.halfExtents.xyz, hintA, i, radius);
        vec3 world = a.position.xyz + rotateByQuat(a.rotation, local);
        SurfacePoint surface = closestOnBody(b, world);
        float depth = radius - surface.distance;
        if (depth > 0.0) {
            // b 의 바깥 법선은 b 에서 a 를 향하므로 뒤집는다.
            addPointNormal(manifold, surface.point, -surface.normal, depth);
        }
    }
    vec3 hintB = probeHint(b, a);
    int countB = probeCount(b.shape);
    for (int i = 0; i < countB; ++i) {
        float radius;
        vec3 local = probePointLocal(b.shape, b.velocity.w, b.halfExtents.xyz, hintB, i, radius);
        vec3 world = b.position.xyz + rotateByQuat(b.rotation, local);
        SurfacePoint surface = closestOnBody(a, world);
        float depth = radius - surface.distance;
        if (depth > 0.0) {
            addPointNormal(manifold, surface.point, surface.normal, depth);
        }
    }
    if (manifold.count > 0) {
        manifold.normal = manifold.normals[0];
    }
    return manifold;
}

// a 의 표본점을 메쉬 b 의 삼각형마다 본다. CPU 의 collideMesh 와 같은 규칙이다.
RigidManifold rigidCollideMesh(RigidBody a, RigidBody mesh) {
    RigidManifold manifold = noContact();
    vec3 hint = rotateByQuatInverse(a.rotation, vec3(0.0, -a.angularVelocity.w, 0.0));
    float thickness = 0.5 * a.angularVelocity.w;
    int count = probeCount(a.shape);
    for (int i = 0; i < count; ++i) {
        float radius;
        vec3 local = probePointLocal(a.shape, a.velocity.w, a.halfExtents.xyz, hint, i, radius);
        vec3 world = a.position.xyz + rotateByQuat(a.rotation, local);
        for (uint t = 0u; t < mesh.triangleCount; ++t) {
            RigidTriangle triangle = push.triangles.items[mesh.triangleOffset + t];
            SurfacePoint surface;
            if (closestOnTriangleSurface(triangle.a, triangle.b, triangle.c, world, radius, thickness, surface)) {
                addPointNormal(manifold, surface.point, -surface.normal, radius - surface.distance);
            }
        }
    }
    if (manifold.count > 0) {
        manifold.normal = manifold.normals[0];
    }
    return manifold;
}

bool rigidStatic(uint shape) {
    return shape == RIGID_SHAPE_PLANE || shape == RIGID_SHAPE_MESH;
}

// 구·상자·평면 짝. a 는 구 또는 상자여야 하고, (상자, 구)는 부르는 쪽이 바꿔서 준다.
RigidManifold rigidCollideBasic(RigidBody a, RigidBody b) {
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
        return flipManifold(manifold);
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

// a 와 b 사이의 접촉. 법선은 a 에서 b 를 향한다. 어느 순서로 주어도 된다. CPU 의 collide 와 같은 분기다.
// GLSL 은 재귀가 없어 순서를 바꿀 때 자리를 맞바꾸고 끝에 뒤집는다.
RigidManifold rigidCollide(RigidBody a, RigidBody b) {
    bool flipped = false;
    if (rigidStatic(a.shape)) {
        if (rigidStatic(b.shape)) {
            return noContact();
        }
        RigidBody swap = a;
        a = b;
        b = swap;
        flipped = true;
    }
    RigidManifold manifold;
    if (b.shape == RIGID_SHAPE_MESH) {
        manifold = rigidCollideMesh(a, b);
    } else if (a.shape == RIGID_SHAPE_CYLINDER || a.shape == RIGID_SHAPE_CAPSULE ||
               b.shape == RIGID_SHAPE_CYLINDER || b.shape == RIGID_SHAPE_CAPSULE) {
        manifold = rigidCollideGeneric(a, b);
    } else if (a.shape == RIGID_SHAPE_BOX && b.shape == RIGID_SHAPE_SPHERE) {
        // 구·상자·평면. (상자, 구)는 (구, 상자)로 바꿔 만들고 뒤집는다.
        manifold = flipManifold(rigidCollideBasic(b, a));
    } else {
        manifold = rigidCollideBasic(a, b);
    }
    return flipped ? flipManifold(manifold) : manifold;
}

#endif
