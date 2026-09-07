#ifndef OBSERVATION_COMMON_GLSL
#define OBSERVATION_COMMON_GLSL

#include "scene_types.glsl"

// 강화 학습의 «정책이 보는 그림». 주 렌더러를 타지 않는 전용 경로다.
//
// 주 렌더러를 안 쓰는 이유는 세 가지다. (1) 시간축 업스케일·자동 노출·Bloom·디노이저가 프레임마다
// 그림을 흔든다 — 관측이 흔들리면 같은 상태가 다른 입력이 되어 학습이 무너진다. (2) GpuCamera 가
// 프레임당 하나라 다중 뷰가 구조적으로 막혀 있다. (3) 창 없이 돌려야 하는데 Renderer 는 스왑체인에
// 얽혀 있다.
//
// 대신 «표면이 어떻게 보이는가» 를 정하는 코드는 저장소 규약대로 **같은 .glsl 함수를 쓴다**
// (material.glsl 의 재질 읽기·노멀 맵, lighting.glsl 의 BRDF, pbr.glsl 의 톤 매핑).
//
// ponytail: 그림자·광원 클러스터·IBL·후처리는 타지 않는다. 방향광 하나와 상수 환경광뿐이다. 진자와
// 리처처럼 물체가 몇 개뿐인 태스크에는 충분하지만, 그림자가 단서인 태스크에는 모자란다.

// 뷰 하나의 셰이딩 상수. 시점이 뷰마다 다르므로 배열로 두고 층 번호로 고른다.
struct ObservationView {
    // xyz 세계 공간 시점. w 는 쓰지 않는다.
    vec4 cameraPosition;
    // xyz 표면에서 광원으로 향하는 방향(정규화). w 세기.
    vec4 lightDirection;
    // xyz 광원 색. w 상수 환경광 세기.
    vec4 lightColor;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer ObservationViewBuffer {
    ObservationView items[];
};

// 128 바이트 한도 안에 든다(116 바이트). 시점·광원은 주소 하나로 묶어 보내 자리를 아낀다.
layout(push_constant, scalar) uniform ObservationPushConstants {
    mat4 viewProjection;
    VertexBuffer vertices;
    InstanceBuffer instances;
    // 스킨 컴퓨트가 뽑아 둔 변형 정점. 장면 패스와 **같은 버퍼**다(두 벌로 만들지 않는다).
    VertexBuffer skinnedVertices;
    MeshBuffer meshes;
    MaterialBuffer materials;
    ObservationViewBuffer views;
    // 지금 그리는 층. views 의 첨자이기도 하다.
    uint view;
}
observationPush;

#endif
