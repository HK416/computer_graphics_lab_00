# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트

Vulkan 1.3 기반 실시간 렌더러. mesh shader 경로와 하드웨어 경로 추적 경로를 갖고, 비교용 고전 VS/FS
경로를 함께 둔다.

`README.md` 에 기능별 설계 근거가 상세히 적혀 있다(그림자 캐스케이드, IBL, 업스케일 세 경로, 더티
플래그, 하드웨어 게이트 등). **해당 영역을 건드리기 전에 그 절을 먼저 읽는다.** 여기에는 README 가
다루지 않는 것 — 명령, 구조, 편집 시 지켜야 할 규약 — 만 적는다.

## 언어 규약

주석, 로그 문자열, 커밋 메시지, 문서는 모두 **한국어**다. 식별자만 영어다. 새 코드도 같게 쓴다.

편집기 UI 의 기술 이름은 **영문 표기가 더 자연스러우면 영문**으로 적는다(Path Tracing, Ray Query,
Upscaling, Mesh Renderer, Frustum Culling 처럼). 설명 문장과 일반 명사(강체, 유체, 반지름, 바닥)는
한국어다.

`// ponytail:` 은 알려진 한계와 후속 작업을 적는 이 저장소의 표식이다. 타협을 남길 때 같은 표식을 쓴다.

## 명령

```sh
cmake --preset debug                        # 최초 구성은 external/vcpkg 를 클론/부트스트랩해 수 분 걸린다
cmake --build --preset debug
cmake --build --preset debug --target run   # 빌드 후 저장소 루트에서 실행
cmake --workflow --preset debug             # 구성 + 빌드 + 테스트
```

`release` 프리셋은 RelWithDebInfo 다. 빌드 산출물은 `build/<preset>/cg_lab`.

Windows 에서 Ninja + MSVC 는 `cl.exe` 가 `INCLUDE` 환경 변수로 표준 라이브러리를 찾는다. 개발자
프롬프트가 아닌 셸(예: 맨 Git Bash)에서 빌드하면 `<array>` 를 못 찾는 `C1083` 로 죽는다. VS Code 의
CMake Tools 는 알아서 맞춰 주지만, 직접 부를 때는 VS 개발자 프롬프트를 쓰거나 그 셸에서
`vcvars64.bat` 을 먼저 실행한다.

헤더를 고쳤는데 `.cpp` 가 다시 컴파일되지 않고 링커가 옛 시그니처를 못 찾는다면, ninja 가 그
오브젝트의 헤더 의존성을 기록하지 못한 것이다. `ninja -t deps <오브젝트>` 가 `#deps 0` 이면
그렇다. `cmake --build --preset release --clean-first` 로 한 번 전체를 다시 만들면 그 뒤로는
정상 기록된다.

```sh
ctest --test-dir build/debug                        # 전체
ctest --test-dir build/debug -R scene_io            # 하나만
ctest --test-dir build/debug --output-on-failure
```

테스트 이름: `lod_network` `animation` `camera` `scene` `scene_io` `profiler` `shadow` `upscaler`
`concurrency` `vertex_pack` `physics` `primitives` `debug_lines` `hardware_profile` `fluid`
`marching_cubes` `headless_physics`(cg_lab 을 `--headless` 로 돌려 저장 결과를 `tests/scenes/expected/` 와 cmp).

선택 기능:

```sh
cmake --preset debug -DCG_LAB_FSR=OFF                  # FidelityFX SDK 스파스 체크아웃까지 건너뛴다
cmake --preset debug -DCG_LAB_DLSS_SDK=<NVIDIA/DLSS 경로>   # 주지 않으면 DLSS 만 빠진다
```

### 렌더 결과 확인

창을 띄우지 않고 렌더 결과를 검증하는 것이 이 저장소의 기본 방식이다. 실행 인자 전체는 README 참조.

기본 장면은 빈 `GameScene` 이라 `--model` 로 무엇이든 올려야 화면에 뭔가 보인다.

```sh
./build/release/cg_lab --model public/assets/DamagedHelmet.glb --screenshot out.png --screenshot-frame 120 --upscaler 3 --render-scale 0.5
./build/release/cg_lab --model public/assets/DamagedHelmet.glb --screenshot out.png --debug 8   # 8 = 모션 벡터
./build/release/cg_lab --open public/scenes/<저장한장면>.json --play --screenshot out.png --screenshot-frame 120
./build/release/cg_lab --model public/assets/DamagedHelmet.glb --profile   # 종료할 때 구간 계측을 로그로 남긴다
./build/release/cg_lab --headless --open tests/scenes/rigid_cpu.json --play --frames 120 --save out.json   # 창 없이 물리만
```

강체 솔버를 바꾸면 `headless_physics` 기준 파일이 갈린다. 의도한 변화면 위 명령으로 다시 만들어
`tests/scenes/expected/rigid_cpu_120.json` 을 갱신하고 커밋한다.

기본 캡처에는 편집기 UI 가 함께 들어가고 콘솔에 시각이 찍히므로 두 실행의 PNG 는 바이트로 같지 않다. **바이트로
견줄 때는 `--fixed-dt 0.016666 --capture present` 를 준다**(렌더 결과만, 고정 프레임 간격). 동작이 바뀌지 않아야 하는
변경은 이 인자로 전후를 `cmp` 한다. 비교 행렬: 헬멧(mesh shader / `--no-mesh-shader` / `--pathtrace` / `--upscaler 2
--render-scale 0.5` / `--upscaler 1 --render-scale 0.5` / `--reflections`)과 `tests/scenes/rigid_*.json --play`. GPU 유체는
원자 합 순서 때문에 바이트가 갈리니 눈으로 본다.

기본값은 `--auto-tune safe` 라 **기기마다 렌더 설정이 달라진다**(높은 등급에서는 광선 반사가 켜지고,
낮은 등급에서는 렌더 배율이 내려간다). 다른 기계의 캡처와 견주거나 변경 전후를 정확히 비교할 때는
`--auto-tune off --no-colliders` 를 함께 준다.

시간축 업스케일(TAAU/FSR/DLSS)은 히스토리를 쌓아야 하므로 `--screenshot-frame` 을 뒤쪽(100 이상)으로
준다. 기본 8 로는 수렴 전 화면을 본다.

### 도구

`compile_commands.json` 이 소스 루트에 링크된다(현재 `build/release` 를 가리킴). `.clang-format` 과
`.clang-tidy` 는 그대로 쓰면 된다. clang-tidy 는 명명 규칙 하나만 검사한다: 타입 `CamelCase`, 함수/변수/
멤버 `camelBack`, enum 상수와 전역/지역 상수 `UPPER_CASE`.

include 순서는 clang-format 이 재그룹핑으로 강제한다: 짝꿍 헤더 → C/C++ 표준 → 서드파티 → 프로젝트.

## 작업 절차

기능 하나가 한 단계다. 단계마다 아래를 순서대로 밟고 커밋한다.

1. **구현 → 빌드 → 테스트 → 스크린샷 검증.** Vulkan 코드는 테스트가 없으므로 `--screenshot` 으로 전후를
   비교한다. 동작이 바뀌지 않아야 하는 리팩터링은 `cmp` 로 바이트 동일까지 확인한다.
2. **포맷과 린트.** 바뀐 `.cpp`/`.h` 에 clang-format 과 clang-tidy 를 돌린다. PATH 의 clang-format 은
   pip 로 깔린 18 이라 VS 의 22 와 결과가 갈릴 수 있다. **VS 것을 절대 경로로 부른다.**

   ```sh
   "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang-format.exe" -i <파일...>
   "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin/clang-tidy.exe" -p build/release <파일.cpp...>
   ```

3. **줄 끝과 인코딩.** 소스는 전부 LF + UTF-8(BOM 없음)이다. `.gitattributes` 와 `.editorconfig` 가
   강제하지만 Windows 도구가 어기기 쉬우니 커밋 전에 확인한다.

   ```sh
   git ls-files --eol | grep -i crlf          # 비어야 한다
   grep -rlP '^\xEF\xBB\xBF' src shaders       # 비어야 한다
   ```

4. **코드 리뷰.** 커밋 전에 diff 를 한 번 리뷰(에이전트 또는 사람)하고 지적을 반영한다. 간단해도 건너뛰지
   않는다. 지적은 고치거나 `// ponytail:` 로 남긴다.
5. **커밋.** 제목은 한국어 한 줄, 필요하면 본문. `Co-Authored-By` 같은 꼬릿말은 적지 않는다. 큼직한 구현이
   끝났으면 다음 작업으로 넘어가기 전에 반드시 커밋한다.

빌드는 VS 개발자 환경이 필요하다. PowerShell 에서:

```powershell
cmd /c "call `"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul && cmake --build --preset release"
```

### 설계 원칙

이 저장소는 **mesh shader 경로와 하드웨어 광선 추적의 융합**을 살피기 위한 것이다. 기능을 넣을 때:

- **병렬화할 수 있는 것은 병렬화한다.** 프레임 CPU 작업(컬링 준비, 애니메이션, 장면 비교, 물리 적분·광역
  검사)은 `core::JobSystem` 에 나눈다. 순서가 필요한 부분만 직렬로 남기고 그 이유를 적는다.
- **공유 상태는 잠금 없는 구조를 먼저 쓴다.** 원자 카운터로 채우는 미리 잡은 배열, `std::atomic_ref`
  격자, `core::LockFreeQueue` 가 그 예다. 뮤텍스는 그 방법이 안 될 때만 쓰고 이유를 적는다.
- **하드웨어 가속 경로가 있으면 그것을 우선한다.** 광선 질의·가속 구조·mesh shader·컴퓨트가 있으면
  그 경로를 기본으로 두고, 고전 경로는 비교·폴백용이다.
- **설정 기본값은 장치 능력에 맞춘다.** `gfx::HardwareProfile`(자동 튜닝)이 기기 등급에 따라 렌더 배율·
  반사·업스케일러를 고른다. 새 기능의 기본값도 여기에 등급별로 넣는다.

- 재질 평가, 표면 복원, 환경광, 안개처럼 «표면이 어떻게 보이는가»를 정하는 코드는 **두 경로가 같은
  `.glsl` 함수를 쓴다.** 한쪽에만 넣거나 복사해 두 벌로 만들지 않는다.
- 정점 변형(스키닝)처럼 두 경로가 함께 읽는 자원은 한 번만 만들어 양쪽이 같은 버퍼를 본다.
- 래스터 전용 기능이라도 경로 추적 모드에서 어떻게 되는지(꺼지는지, 같은 결과를 내는지)를 정하고 편집기에
  드러낸다.

## 구조

`src/main.cpp` 가 인자를 파싱해 `app::Application` 을 띄운다. 계층은 `app` → `editor`/`gfx`/`scene`/
`asset`/`physics` → `core` 방향으로만 의존한다. `physics` 는 `scene` 과 `core` 만 보고, `gfx` 는 유체
CPU 백엔드를 부르느라 `physics` 를 하나 본다.

| 경로 | 내용 |
| --- | --- |
| `src/app` | 수명 주기, SDL 창, 이벤트 루프, 모델/장면 적재. `plugin.h` 의 `Plugin`/`Services` 와 `plugins/` 의 기능 플러그인(물리 등) |
| `src/asset` | glTF 적재, meshlet/LOD DAG 구축, 애니메이션 샘플링. CPU 측 표현 |
| `src/scene` | 장면 그래프, 카메라, 커스텀 JSON 직렬화 |
| `src/gfx` | Vulkan 컨텍스트, 리소스, 렌더 경로 전부. `Renderer` 는 클래스 하나지만 정의가 `renderer_*.cpp` 에 기능별로 나뉜다(`renderer_internal.h` 가 공유 푸시 상수·포맷). `render_graph.h` 가 프레임 패스 목록. GPU SPH(`fluid.cpp`)도 여기 |
| `src/physics` | 강체 솔버와 CPU SPH. `scene` 과 `core` 에만 의존한다. 강체는 재생 중 `Application::run` 이 고정 간격으로 부르고, 유체 CPU 백엔드는 `gfx::FluidSimulator` 가 부른다(그래서 `gfx` → `physics` 의존이 하나 있다) |
| `src/editor` | ImGui 도킹 편집기 |
| `src/core` | `fatal`, 잠금 없는 작업 큐 |
| `shaders` | GLSL. `.glsl` 은 include 전용 공통 헤더 |

### 프레임 흐름

`Application::run` 한 바퀴: 이벤트 → `camera.update` → `scene.update`(애니메이션 진행) → **플러그인 `update`**
(재생 중이면 `PhysicsPlugin` 이 `physics::stepRigidBodies` 를 고정 간격으로) → `renderer.prepareFrame`(밀린 크기
변경) → `editor.build` → **`scene.refresh`** → [구조가 바뀐 프레임이면 `collectUnusedModels`] → `renderer.drawFrame`.

기능은 `app::Plugin`(`src/app/plugin.h`) 으로 붙인다. `Application::registerPlugins` 의 등록 순서가 프레임 안의 호출
순서다. 플러그인은 `Services` 참조 묶음으로만 다른 계층을 본다.

`refresh()` 는 편집기가 장면을 만진 **뒤**, 렌더러가 읽기 **전**에 불려야 한다. 훅이 아니라 지난 사본과
필드를 직접 비교해 더티를 찾고 세계 변환/가시성 캐시를 다시 만든다(이유는 README).

`Renderer::drawFrame` → `buildDrawCommands`(유체 `prepare` 포함) → `buildLights` → `recordCommands`. `recordCommands` 는
패스를 `RenderGraph` 노드로 등록하고(플러그인 `addPass` 훅이 그 뒤에 자기 노드를 끼움) `execute` 한다. 노드 순서:

환경 맵 굽기(설정이 바뀔 때만) → 스킨 컴퓨트(변형 정점·meshlet 경계) → [강체 GPU 솔버: PhysicsPlugin 이 끼움] →
유체 컴퓨트(입자 진행, 인스턴스와 TLAS 인스턴스 쓰기; CPU 백엔드는 지난 프레임이 띄운 스텝을 거둬 쓰고 노드 끝에서 다음 스텝을 백그라운드로 띄운다) → 그림자 패스 → [경로 추적] **또는** [컬(1차) → 불투명(1차, 끝에 유체 인스턴스 드로우)
→ HZB → 컬(2차) → 불투명(2차) → 하늘 → 광선 반사 → OIT → 합성 → SSAO] → Bloom·자동 노출 → 시간축
업스케일 → 톤 매핑 → 공간 업스케일 → UI.

유체 입자 인스턴스는 오브젝트 인스턴스 **뒤**(`objects.size()` 부터)에 GPU 가 쓴다. CPU 는 앞쪽만
memcpy 하므로 겹치지 않는다. 상위 가속 구조 인스턴스 버퍼는 반대로 입자가 **앞**이고 오브젝트가 뒤에
붙는다(`updateTopLevel` 의 `prependedInstances`).

업스케일 두 방식이 톤 매핑을 사이에 두고 갈리는 이유와 시간축 경로가 요구하는 지터/모션 벡터/하늘
배관은 README 의 «업스케일» 절에 있다.

### GPU-Driven 자원 전달

- **디스크립터 집합 0 하나뿐**이고 그게 bindless 다(`gfx::BindlessTextures` ↔ `shaders/bindless.glsl`).
  이미지·샘플러·스토리지 이미지 배열이 포맷별로 나뉘어 있고, 슬롯은 하위 24비트 이미지 + 상위 8비트
  샘플러로 인코딩된다. 번호 공간이 배열마다 별개라 섞어 쓰면 안 된다.
- 나머지 버퍼는 전부 **buffer device address** 로 푸시 상수에 실린다
  (`shaders/scene_data.glsl` 의 `PushConstants`, `shaders/scene_types.glsl` 의 `buffer_reference` 타입).
- 집합 1 은 광선 질의 그림자 변종과 반사 컴퓨트가 TLAS 를 묶을 때만 쓴다.

### CPU/GPU 배치가 묶인 자리

한쪽을 고치면 반드시 다른 쪽도 고친다. 컴파일러가 잡아 주지 않는다.

| C++ | GLSL |
| --- | --- |
| `asset::Vertex` (`src/asset/model.h`) | `Vertex` (`shaders/scene_types.glsl`) |
| `GpuMesh` `GpuMeshLod` `GpuMeshlet` `GpuMaterial` `GpuInstance` (`src/gfx/geometry.h`) | 동명 구조체 (`scene_types.glsl`) |
| `GpuLight` (`src/gfx/renderer.h`) | `Light` (`scene_types.glsl`) |
| `GpuFluidCollider` `GpuFluidParams` `FluidPushConstants` (`src/gfx/fluid.h`) | `FluidCollider` `FluidParams` `FluidPushConstants` (`shaders/fluid_common.glsl`) |
| `GpuRigidBody` `RigidPushConstants` (`src/gfx/rigid_body_gpu.h`) | `RigidBody` `RigidPushConstants` (`shaders/rigid_common.glsl`) |
| `physics::Triangle` (`src/physics/rigid_body.h`) | `RigidTriangle` (`rigid_common.glsl`) |
| `collideBoxBox` 등 접촉 생성 (`src/physics/rigid_body.cpp`) | `rigidCollide` (`shaders/rigid_common.glsl`) |
| 모양 기하 `closestOn*Local` `probePointLocal` `closestOnTriangleSurface` (`src/physics/collider_shapes.h`) | 동명 함수 (`shaders/collider_shapes.glsl`) |
| `physics::MAX_MANIFOLD_POINTS` (`src/physics/rigid_body.h`) | `RIGID_MAX_MANIFOLD` (`rigid_common.glsl`) |
| `FluidSurfacePushConstants` (`src/gfx/fluid.h`) | 동명 블록 (`shaders/fluid_surface_common.glsl`) |
| `GpuFluidSurfaceInfo` `FLUID_FLAG_*` `FLUID_SURFACE_CUSTOM_INDEX` `FLUID_SURFACE_RAY_MASK` (`src/gfx/fluid.h`) | `FluidSurfaceInfo` `FLUID_FLAG_*` `FLUID_SURFACE_*` (`shaders/fluid_types.glsl`) |
| `PathTracePushConstants` (`src/gfx/raytracing.cpp`) | 동명 블록 (`shaders/pathtrace_common.glsl`) — 128 바이트 한도라 작은 값은 16비트 둘씩 묶는다 |
| `ReflectPushConstants` (`src/gfx/renderer_internal.h`) | 동명 블록 (`shaders/reflect.comp`) — `samplesResetDebug` 에 셋을 묶는다 |
| `FluidDrawPushConstants` (`src/gfx/renderer_internal.h`) | 동명 블록 (`shaders/fluid_draw_common.glsl`) |
| `physics::SurfaceVertex` (`src/physics/marching_cubes.h`) | `FluidSurfaceVertex` (`shaders/fluid_types.glsl`) |
| `physics::MC_TABLE` `MC_EDGES` `MC_CORNERS` (`marching_cubes.cpp`) | `MC_TABLE` `MC_EDGE_CORNERS` `MC_CORNER_OFFSET` (`shaders/marching_cubes.glsl`) |

마칭 큐브 표는 두 벌이 될 수밖에 없다. **손으로 고치지 않는다** — 규칙에서 만들어 낸 것이고
`marching_cubes` 테스트가 셰이더 표를 파일에서 읽어 C++ 표와 같은지 확인한다. 장을 만드는 식
(`fluid_field.comp` ↔ `physics::buildFluidField`)과 격자 가장자리를 0 으로 두는 규칙도 같아야 한다.

강체 솔버 상수(`GRAVITY` `POSITION_CORRECTION` `PENETRATION_SLOP` `RESTITUTION_THRESHOLD`
`POSITION_ITERATIONS`)는 `src/physics/rigid_body.h` 한 곳에만 두고 GPU 쪽은 푸시 상수로 실어 보낸다.
두 벌로 두면 백엔드를 바꿀 때 거동이 갈린다.
| `scene::ColliderShape` (`src/scene/scene.h`) | `COLLIDER_SHAPE_*` (`collider_shapes.glsl`; `RIGID_SHAPE_*` `FLUID_COLLIDER_*` 는 그 별칭) |

| `Options::debugMode`, `RenderSettings::debugMode` (`src/gfx/render_settings.h`) | `DEBUG_MODE_*` (`scene_types.glsl`) |
| `DebugLineVertex` (`src/gfx/debug_lines.h`), `DebugLinePushConstants` (`renderer_internal.h`) | 동명 구조체 (`shaders/debug_line_common.glsl`) |

전부 `scalar` 레이아웃이다.

배치가 아니라 «값» 이 묶인 자리도 하나 있다. `physics::FluidParams`(`src/physics/fluid_sph.h`)는 CPU
백엔드가 쓰고, `FluidSimulator::fillParams` 가 그것을 `GpuFluidParams` 로 필드마다 옮겨 담는다. 유체
설정을 더할 때는 세 곳(`FluidParams`, `GpuFluidParams`, `fluid_common.glsl`)을 함께 고친다.

**푸시 상수 블록은 `layout(push_constant)` 만 쓰면 std430 이라** `vec2`/`ivec2`/`vec4` 가 8·16바이트
경계로 밀려 C++ 의 빽빽한 배치와 조용히 어긋난다. 앞에 홀수 개의 4바이트 멤버가 오는 벡터를 넣을
때는 `layout(push_constant, scalar)` 를 붙인다(`shaders/exposure_histogram.comp` 가 그 예다).
확인은 `spirv-dis <출력>.spv | grep Offset` 으로 한다.

### 하드웨어 기능 게이트

`gfx::Capabilities`(`src/gfx/context.h`)가 기동 시 조회한 것으로 경로를 켜고 끈다. **미지원 기능에
소프트웨어 폴백을 만들지 않는다** — 그 경로 자체를 비활성화하고 편집기에 사유를 보여준다. 새 확장을
요구하는 기능을 넣을 때도 같은 방식을 따른다.

MoltenVK(macOS)에는 mesh shader 와 광선 추적이 없어 고전 경로만 돈다.

## 편집 시 주의

- **새 `.cpp`** 는 `CMakeLists.txt` 의 `add_executable(cg_lab ...)` 목록에 직접 추가한다.
- **새 셰이더**는 `shader_sources` 에 추가한다. 산출물은 `<basename>.spv` 이고 런타임에
  `createShaderModule("<basename>.spv")` 로 읽는다. `-D` 정의만 다른 변종이 필요하면 `shader_variants`
  에 `소스|출력이름|정의` 로 넣는다(광선 질의 그림자가 이 방식이다).
- **새 테스트**는 라이브러리 타깃이 없어 필요한 `.cpp` 를 전부 나열해야 한다:
  `add_executable` + `target_include_directories(... PRIVATE src)` + 링크 + `add_test`, 그리고
  파일 끝의 `-UNDEBUG` `foreach` 목록에도 타깃을 넣는다. 테스트는 Catch2 가 아니라 `assert` 를 쓰는
  평범한 `main()` 이라, `NDEBUG` 가 살아 있으면 검사가 통째로 사라진다.
- 테스트가 도는 것은 순수 계산 부분(`*_math.cpp`, 장면 그래프, 애니메이션, 직렬화, 잠금 없는 큐)뿐이다.
  Vulkan 을 타는 코드에는 테스트가 없으므로 스크린샷 비교로 확인한다.
- **새 렌더 패스**는 `recordCommands` 의 `graph.add` 노드로(또는 플러그인이면 `Renderer::addPass` 훅의 `addAfter` 로)
  등록한다. 이미지 사용은 `reads`/`writes`/`leaves` 로 선언하고 **노드 안에 `imageBarrier` 를 새로 쓰지 않는다.** 층·밉
  단위 전이만 예외이고, 그때는 `leaves` 로 남긴 상태를 알린다. 조건은 `enabled` 로 두고 노드 자체는 늘 등록한다.
- **새 기능은 `app::Plugin`** 으로 붙이고 `Application::registerPlugins` 에 등록한다. 편집기 절은 `ui()` 에서
  `editor->settingsSection("이름")` 으로 연다. `Services` 의 gfx·editor 멤버는 포인터고 `--headless` 에서 null 이다 —
  `build`/`update` 는 쓰기 전에 살피고, `ui` 는 편집기가 있을 때만 불린다. `editor/` 는 `app::` 를 보지 않는다 — 플러그인 → 편집기 교환은 Editor 의 공개
  상태 필드·콜백으로만.
- **Vulkan 실패는 복구하지 않는다.** `VK_CHECK(...)`(`src/gfx/vk_check.h`) 또는 `core::fatal(...)` 로
  메시지 박스를 띄우고 종료한다.
- **프로파일러 구간**은 `gfx::ProfilerScope scope(profiler, "이름", commandBuffer)` RAII 로 잡는다.
  이름은 수명이 프로그램 전체인 리터럴이어야 하고, 프레임당 구간은 `MAX_PROFILER_ZONES`(32)까지다.
- **Windows**: 경로 문자열 비교는 `path::native()`(wstring)가 아니라 `generic_string()` 으로 한다.
  UTF-8 코드 페이지는 `platform/windows/cg_lab.manifest` 와 `main` 의 `SetConsoleOutputCP` 가 맡는다.
- `external/` 은 CMake 가 고정 태그로 클론하는 자리다. 저장소에 담지 않는다.
