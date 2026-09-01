# Computer Graphics Lab 00

Mesh shader 와 path tracing 을 사용하는 실시간 렌더러. 비교를 위한 고전 VS/FS 파이프라인 경로를 함께 제공한다.

## 요구 사항

- C++20 컴파일러 (MSVC 19.3x / Clang 16+ / GCC 13+)
- CMake 3.25 이상, Ninja
- [Vulkan SDK](https://vulkan.lunarg.com/) 1.3 이상 (셰이더 컴파일러와 검증 레이어)

vcpkg 는 최초 CMake 구성 시 `external/vcpkg` 에 고정 태그로 자동 클론/부트스트랩된다. 별도 설치가 필요 없다.

## 빌드

```sh
cmake --preset debug     # 최초 구성 시 vcpkg 의존성 빌드로 수 분 소요
cmake --build --preset debug
./build/debug/cg_lab
```

`release` 프리셋은 RelWithDebInfo 로 구성된다.

## 실행 인자

| 인자 | 설명 |
| --- | --- |
| `--scene <n>` | 시작할 장면 번호 |
| `--screenshot <경로>` | 몇 프레임 뒤 화면을 PNG 로 저장하고 종료한다. 렌더 결과 검증용 |

장면 뷰 위에서 마우스 오른쪽 버튼을 누른 채 WASD/QE 로 카메라를 움직인다. 장면 전환은 계층 패널의
드롭다운이나 숫자 키로 한다.

기즈모 조작은 `W` 이동, `E` 회전, `R` 크기이며 `Ctrl` 을 누른 채 끌면 스냅이 걸린다. 계층 패널에서
오브젝트를 추가, 복제, 삭제할 수 있다.

## 하드웨어 기능 게이트

렌더 모드별로 필요한 Vulkan 기능을 기동 시 조회하여, 지원되는 모드만 활성화한다. 미지원 기능에 대한
소프트웨어 폴백 경로는 제공하지 않는다. 예를 들어 MoltenVK 에는 `VK_EXT_mesh_shader` 와
`VK_KHR_ray_tracing_pipeline` 이 없으므로 macOS 에서는 고전 파이프라인 경로만 사용할 수 있다.

## 디렉터리

| 경로 | 설명 |
| --- | --- |
| `src/app` | 애플리케이션 수명 주기, 윈도우, 입력 |
| `src/asset` | glTF 적재와 CPU 측 모델 표현 |
| `src/editor` | ImGui 기반 편집기 GUI |
| `src/gfx` | Vulkan 컨텍스트, 리소스, 렌더 경로 |
| `src/scene` | 장면 그래프와 카메라 |
| `shaders` | GLSL 셰이더 |
| `src/core` | 로깅, 오류 처리 등 공통 유틸리티 |
| `public` | 폰트와 샘플 에셋 |
