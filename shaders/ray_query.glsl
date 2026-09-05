#ifndef RAY_QUERY_GLSL
#define RAY_QUERY_GLSL

#extension GL_EXT_ray_query : require

// 광선 질의 가시성. 래스터의 하이브리드 그림자와 반사 컴퓨트의 그림자 광선이 같은 함수를 쓴다.
// 0 이면 완전히 가려진 것이다. 불투명 플래그를 켜므로 컷오프·반투명 재질도 막는 것으로 친다.
float rayQueryVisibility(accelerationStructureEXT topLevel, vec3 position, vec3 normal, vec3 toLight, float maxDistance) {
    // 자기 자신을 맞히지 않도록 표면에서 살짝 띄운다. 접선 방향 오차가 텍셀 크기에 비례하지
    // 않으므로 그림자 맵보다 훨씬 작은 값이면 된다.
    vec3 origin = position + normal * 1.0e-3;

    rayQueryEXT query;
    rayQueryInitializeEXT(query,
                          topLevel,
                          gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                          // 물 표면 인스턴스(마스크 0x01)는 지나간다. 래스터의 물은 반투명으로 그려지므로 그림자를
                          // 던지지 않는 편이 맞다.
                          0xFEu,
                          origin,
                          1.0e-4,
                          toLight,
                          maxDistance);
    rayQueryProceedEXT(query);
    return rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

#endif
