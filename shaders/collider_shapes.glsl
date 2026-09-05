#ifndef COLLIDER_SHAPES_GLSL
#define COLLIDER_SHAPES_GLSL

// 콜라이더 모양의 기하. src/physics/collider_shapes.h 와 같은 규칙이어야 CPU·GPU 백엔드가 같은 접촉을
// 본다. 모두 콜라이더의 지역 공간이고 원기둥·캡슐의 축과 평면의 법선은 +Y 다.
//
// 번호는 scene::ColliderShape 와 같다. 강체(RIGID_SHAPE_*)와 유체(FLUID_COLLIDER_*)가 이 하나를 쓴다.
#define COLLIDER_SHAPE_SPHERE 0u
#define COLLIDER_SHAPE_BOX 1u
#define COLLIDER_SHAPE_PLANE 2u
#define COLLIDER_SHAPE_CYLINDER 3u
#define COLLIDER_SHAPE_CAPSULE 4u
#define COLLIDER_SHAPE_MESH 5u

// 점에서 가장 가까운 표면점. distance 는 부호 있는 거리(안이면 음수), normal 은 표면 바깥 방향.
struct SurfacePoint {
    vec3 point;
    vec3 normal;
    float distance;
};

SurfacePoint closestOnSphereLocal(float radius, vec3 p) {
    SurfacePoint result;
    float d = length(p);
    result.normal = d > 1.0e-6 ? p / d : vec3(0.0, 1.0, 0.0);
    result.point = result.normal * radius;
    result.distance = d - radius;
    return result;
}

SurfacePoint closestOnBoxLocal(vec3 extent, vec3 p) {
    SurfacePoint result;
    vec3 clamped = clamp(p, -extent, extent);
    vec3 delta = p - clamped;
    float distanceSq = dot(delta, delta);
    if (distanceSq > 1.0e-10) {
        float distance = sqrt(distanceSq);
        result.normal = delta / distance;
        result.point = clamped;
        result.distance = distance;
        return result;
    }
    vec3 gap = extent - abs(p);
    int axis = gap.x < gap.y ? (gap.x < gap.z ? 0 : 2) : (gap.y < gap.z ? 1 : 2);
    vec3 normal = vec3(0.0);
    normal[axis] = p[axis] >= 0.0 ? 1.0 : -1.0;
    result.point = p;
    result.point[axis] = normal[axis] * extent[axis];
    result.normal = normal;
    result.distance = -gap[axis];
    return result;
}

SurfacePoint closestOnPlaneLocal(vec3 p) {
    SurfacePoint result;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.point = vec3(p.x, 0.0, p.z);
    result.distance = p.y;
    return result;
}

SurfacePoint closestOnCapsuleLocal(float radius, float halfHeight, vec3 p) {
    vec3 onSegment = vec3(0.0, clamp(p.y, -halfHeight, halfHeight), 0.0);
    SurfacePoint result = closestOnSphereLocal(radius, p - onSegment);
    result.point += onSegment;
    return result;
}

SurfacePoint closestOnCylinderLocal(float radius, float halfHeight, vec3 p) {
    SurfacePoint result;
    vec3 radial = vec3(p.x, 0.0, p.z);
    float d = length(radial);
    vec3 radialDirection = d > 1.0e-6 ? radial / d : vec3(1.0, 0.0, 0.0);
    bool inside = abs(p.y) <= halfHeight && d <= radius;
    if (inside) {
        float gapY = halfHeight - abs(p.y);
        float gapR = radius - d;
        if (gapY < gapR) {
            float side = p.y >= 0.0 ? 1.0 : -1.0;
            result.normal = vec3(0.0, side, 0.0);
            result.point = vec3(p.x, side * halfHeight, p.z);
            result.distance = -gapY;
        } else {
            result.normal = radialDirection;
            result.point = radialDirection * radius + vec3(0.0, p.y, 0.0);
            result.distance = -gapR;
        }
        return result;
    }
    vec3 closest = radialDirection * min(d, radius);
    closest.y = clamp(p.y, -halfHeight, halfHeight);
    vec3 delta = p - closest;
    float distance = length(delta);
    result.point = closest;
    result.normal = distance > 1.0e-6 ? delta / distance : vec3(0.0, 1.0, 0.0);
    result.distance = distance;
    return result;
}

// 메쉬는 볼록하지 않아 여기 없다(closestOnTriangleSurface).
SurfacePoint closestOnColliderLocal(uint shape, float radius, vec3 halfExtents, vec3 p) {
    if (shape == COLLIDER_SHAPE_SPHERE) {
        return closestOnSphereLocal(radius, p);
    }
    if (shape == COLLIDER_SHAPE_BOX) {
        return closestOnBoxLocal(halfExtents, p);
    }
    if (shape == COLLIDER_SHAPE_PLANE) {
        return closestOnPlaneLocal(p);
    }
    if (shape == COLLIDER_SHAPE_CYLINDER) {
        return closestOnCylinderLocal(radius, halfExtents.y, p);
    }
    if (shape == COLLIDER_SHAPE_CAPSULE) {
        return closestOnCapsuleLocal(radius, halfExtents.y, p);
    }
    return closestOnSphereLocal(0.0, p);
}

#define CYLINDER_RIM_SAMPLES 8
const float CYLINDER_RIM_ANGLES[CYLINDER_RIM_SAMPLES] =
    float[](0.0, 1.5707964, 3.1415927, 4.712389, 0.7853982, 2.3561945, 3.9269907, 5.4977870);

// 상대에 «찔러 보는» 표면 점의 수. C++ 의 probeCount 와 같다.
int probeCount(uint shape) {
    if (shape == COLLIDER_SHAPE_SPHERE) {
        return 1;
    }
    if (shape == COLLIDER_SHAPE_BOX) {
        return 8;
    }
    if (shape == COLLIDER_SHAPE_CAPSULE) {
        return 3;
    }
    if (shape == COLLIDER_SHAPE_CYLINDER) {
        return 4 + 2 * CYLINDER_RIM_SAMPLES;
    }
    return 0;
}

// index 번째 표본점(지역 공간). hint 는 상대 표면에서 이 콜라이더 중심에 가장 가까운 점(지역 공간).
vec3 probePointLocal(uint shape, float radius, vec3 halfExtents, vec3 hint, int index, out float probeRadius) {
    probeRadius = 0.0;
    if (shape == COLLIDER_SHAPE_SPHERE) {
        probeRadius = radius;
        return vec3(0.0);
    }
    if (shape == COLLIDER_SHAPE_BOX) {
        return vec3((index & 1) != 0 ? halfExtents.x : -halfExtents.x,
                    (index & 2) != 0 ? halfExtents.y : -halfExtents.y,
                    (index & 4) != 0 ? halfExtents.z : -halfExtents.z);
    }
    if (shape == COLLIDER_SHAPE_CAPSULE) {
        probeRadius = radius;
        float h = halfExtents.y;
        float y = index == 0 ? h : (index == 1 ? -h : clamp(hint.y, -h, h));
        return vec3(0.0, y, 0.0);
    }
    if (shape == COLLIDER_SHAPE_CYLINDER) {
        float h = halfExtents.y;
        if (index < 2) {
            vec3 radial = vec3(hint.x, 0.0, hint.z);
            float d = length(radial);
            vec3 direction = d > 1.0e-6 ? radial / d : vec3(1.0, 0.0, 0.0);
            return direction * radius + vec3(0.0, index == 0 ? h : -h, 0.0);
        }
        if (index < 4) {
            return vec3(0.0, index == 2 ? h : -h, 0.0);
        }
        int rim = index - 4;
        float angle = CYLINDER_RIM_ANGLES[rim % CYLINDER_RIM_SAMPLES];
        float y = rim < CYLINDER_RIM_SAMPLES ? h : -h;
        return vec3(cos(angle) * radius, y, sin(angle) * radius);
    }
    return vec3(0.0);
}

// 삼각형 위의 가장 가까운 점(Ericson 5.1.5). C++ 의 closestOnTriangle 과 같다.
vec3 closestOnTriangle(vec3 a, vec3 b, vec3 c, vec3 p) {
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }
    vec3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        return a + ab * (d1 / (d1 - d3));
    }
    vec3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        return a + ac * (d2 / (d2 - d6));
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    float denominator = 1.0 / (va + vb + vc);
    return a + ab * (vb * denominator) + ac * (vc * denominator);
}

// 반지름 radius 의 점이 삼각형(앞면 CCW)에 닿는지. C++ 의 closestOnTriangleSurface 와 같다.
bool closestOnTriangleSurface(vec3 a, vec3 b, vec3 c, vec3 p, float radius, float thickness, out SurfacePoint result) {
    result.point = p;
    result.normal = vec3(0.0, 1.0, 0.0);
    result.distance = 0.0;
    vec3 faceNormal = cross(b - a, c - a);
    float area = length(faceNormal);
    if (area < 1.0e-12) {
        return false;
    }
    faceNormal /= area;
    float signedHeight = dot(p - a, faceNormal);
    vec3 closest = closestOnTriangle(a, b, c, p);
    vec3 delta = p - closest;
    float distance = length(delta);
    if (signedHeight >= 0.0) {
        if (distance >= radius) {
            return false;
        }
        result.point = closest;
        result.normal = distance > 1.0e-6 ? delta / distance : faceNormal;
        result.distance = distance;
        return true;
    }
    if (-signedHeight > thickness || abs(distance + signedHeight) > 1.0e-4 * max(1.0, distance)) {
        return false;
    }
    result.point = closest;
    result.normal = faceNormal;
    result.distance = signedHeight;
    return true;
}

#endif
