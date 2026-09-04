#pragma once

#include <cstdint>
#include <string_view>

#include "asset/model.h"

namespace asset {

// 코드로 만드는 기본 도형. glTF 와 같은 경로(buildLodHierarchy → GeometryStore)를 타므로 여기서는
// 정점·인덱스·재질만 채운다.
//
// 크기는 «축 정렬 크기»로 맞춘다. 모두 ±1 정육면체에 들어가고 적어도 한 축이 그 벽에 닿는다. 경계
// 구를 맞추면 정육면체가 구보다 훨씬 작아 보인다.
//
// 기준이 반지름 1 인 이유는 내장 구가 예전부터 그랬고 유체 입자가 그것에 기대고 있기 때문이다
// (fluid_instances.comp 이 인스턴스 배율을 입자 반지름으로 준다). 그래서 상자는 반쪽 크기 1,
// 원기둥·콘은 반지름 1 에 높이 2, 캡슐은 반지름 0.5 에 전체 높이 2, 토러스는 바깥 반지름 1 이다.
enum class Primitive : uint32_t {
    PLANE = 0,
    BOX,
    SPHERE,
    ICO_SPHERE,
    CYLINDER,
    CONE,
    CAPSULE,
    TORUS,
    COUNT,
};

// 편집기 메뉴에 쓰는 이름.
const char* primitiveLabel(Primitive primitive);
// 장면 파일에서 모델 경로 자리에 적히는 이름. `<builtin:box>` 꼴이다.
const char* primitiveAssetName(Primitive primitive);
// 그 이름으로 되찾는다. 내장 도형이 아니면 COUNT.
Primitive primitiveFromAssetName(std::string_view name);

// 메쉬 하나와 재질 하나를 담은 모델을 만든다. 경계는 채워져 있고 meshlet·LOD 는 아직 없다.
Model makePrimitive(Primitive primitive);

} // namespace asset
