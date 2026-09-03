#pragma once

namespace core {
class JobSystem;
} // namespace core

namespace scene {
struct Scene;
} // namespace scene

namespace physics {

// 강체 부품이 붙은 오브젝트를 dt 만큼 진행시킨다. 세계 공간에서 적분하고 충돌을 풀어 오브젝트의 지역
// 변환에 되돌려 쓴다. 고정 간격으로 나눠 부르는 것은 부르는 쪽 몫이다. jobs 가 있으면 적분·광역
// 검사·되돌려 쓰기를 워커에 나누고, 접촉 목록은 원자 카운터로 모아 잠금이 없다.
//
// 구·상자·평면 콜라이더, 순차 임펄스 접촉 해결(반발·마찰·위치 보정). 관절이나 슬립은 없다.
void stepRigidBodies(scene::Scene& scene, float dt, core::JobSystem* jobs);

} // namespace physics
