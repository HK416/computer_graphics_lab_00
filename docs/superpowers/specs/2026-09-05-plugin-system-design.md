# 플러그인 시스템 설계 제안

**결정(2026-09-05): B 안으로 진행, 헤드리스 모드는 후속 작업.** 구현 결과는 README «렌더 그래프»·«플러그인» 절과
CLAUDE.md 에 있다. 아래는 결정 전의 선택지 기록이다.

Bevy 의 `Plugin` 처럼 기능 단위로 등록·해제할 수 있는 구조를 원한다는 요청에 대한 설계 초안이다.
**구현 전에 아래 선택지 가운데 어느 것으로 갈지 확인이 필요하다.** 이 문서는 코드에 손대지 않았다.

## 지금 구조가 플러그인화를 어렵게 하는 자리

- `Renderer::recordCommands` 가 패스 순서를 **한 함수 안에 하드코딩**한다(컬 → 불투명 → HZB → … → UI).
  기능이 켜지고 꺼지는 것은 `if (useReflections)` 같은 플래그다.
- `Application::run` 이 프레임 순서(입력 → 카메라 → 장면 → 물리 → 렌더러 → 편집기 → refresh → draw)를
  직접 부른다. 물리·유체는 여기와 `Renderer` 두 곳에 손이 닿는다.
- 자원 전달이 bindless 집합 하나 + 푸시 상수의 buffer device address 라 «플러그인이 자기 버퍼를 등록하고
  셰이더가 찾아 쓰는» 식으로 바꾸려면 `PushConstants` 배치를 건드려야 한다.
- 편집기(`editor.cpp` 2천 줄)가 모든 기능의 인스펙터·렌더 설정 UI 를 한 파일에 갖고 있다.

Bevy 의 플러그인은 ECS 스케줄(시스템 등록)과 리소스 주입 위에 서 있다. 여기에는 그 둘이 없으므로
«Bevy 와 비슷하게» 가 무엇을 뜻하는지가 먼저 정해져야 한다.

## 선택지

### A. 앱 수준 훅 플러그인 (권장, 작음)

```cpp
struct Plugin {
    virtual ~Plugin() = default;
    virtual const char* name() const = 0;
    virtual void build(App& app) {}          // 기동 시 자원·파이프라인 생성, 편집기 패널 등록
    virtual void update(App& app, float dt) {} // 프레임 앞부분(물리 스텝 등)
    virtual void ui(App& app) {}             // 편집기 패널·인스펙터 절
};
```

- `Application` 이 `std::vector<std::unique_ptr<Plugin>>` 를 들고 정해진 자리에서 순서대로 부른다.
- 첫 대상: `PhysicsPlugin`(강체 스텝·GPU 되읽기 호출), `FluidPlugin`, `DebugLinesPlugin`, `ProfilerPlugin`.
  렌더 패스 자체는 옮기지 않는다 — `Renderer` 는 그대로 두고 플러그인이 그것을 부른다.
- 편집기의 «렌더 설정» 절과 인스펙터 부품 절을 플러그인 `ui()` 로 하나씩 옮긴다.
- 장점: 며칠 안에 끝나고 프레임 흐름·셰이더 배관이 안 바뀐다. 단점: 렌더 패스는 여전히 `Renderer`
  안에 고정이라 «패스를 플러그인으로 끼워 넣기» 는 못 한다.

### B. 렌더 그래프까지 플러그인 (큼)

- A 에 더해 `Renderer::recordCommands` 를 패스 노드 목록으로 풀고, 플러그인이 `addPass(name, after,
  before, record)` 로 노드를 등록한다. 첨부물·배리어를 그래프가 해석한다.
- 두 경로(래스터/경로 추적)가 다른 그래프를 갖고, 업스케일 세 경로의 «톤 매핑 사이에 갈리는» 규칙을
  노드 의존으로 표현해야 한다.
- 장점: 진짜 Bevy 식. 단점: 몇 주 규모, 스크린샷 회귀 위험이 크고 README «업스케일» 절의 규칙을 전부
  그래프 제약으로 옮겨야 한다.

### C. 컴파일 타임 기능 토글만 (가장 작음)

- CMake 옵션(`CG_LAB_FLUID=OFF` 같은)으로 소스와 셰이더를 빼는 것. 플러그인 «시스템» 은 아니다.
- 로보틱스 학습처럼 렌더러 없이 물리만 돌리는 빌드가 목표라면 이것으로 충분할 수 있다.

## 권장

**A 로 시작한다.** 로보틱스 학습(MuJoCo/IsaacSim 류) 아이디어를 감안하면 결국 «렌더러 없이 물리만»,
«헤드리스 다중 환경 스텝» 이 필요해지는데, 그 첫 단계도 `PhysicsPlugin` 이 `Renderer` 에 기대지 않게
떼어 내는 일이라 A 와 겹친다. B 는 렌더 그래프가 정말 필요해질 때(패스를 외부에서 끼우는 요구가 생길
때) 올린다.

## 확인이 필요한 것

1. A / B / C 가운데 어느 것인가.
2. 플러그인 단위: 기능별(물리·유체·디버그 선·프로파일러)인가, 계층별(gfx/physics/editor)인가.
3. 편집기 UI 를 플러그인이 갖게 하는가(A 의 `ui()`), 아니면 `editor.cpp` 에 남기는가.
