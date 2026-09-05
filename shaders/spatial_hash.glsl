#ifndef SPATIAL_HASH_GLSL
#define SPATIAL_HASH_GLSL

// 균일 격자 해시. 유체 SPH 와 강체 광역 검사가 함께 쓴다. src/physics/spatial_hash.h 의 같은 이름 함수와
// 결과가 같아야 두 백엔드가 같은 이웃을 본다. cellCount 는 2 의 거듭제곱이어야 한다.

ivec3 spatialCell(vec3 position, float cellSize) {
    return ivec3(floor(position / cellSize));
}

uint spatialHash(ivec3 cell, uint cellCount) {
    uint hashed = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u;
    return hashed & (cellCount - 1u);
}

#endif
