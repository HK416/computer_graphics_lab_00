# 헤드리스 물리 회귀. cg_lab 을 창 없이 돌려 강체 장면을 120 프레임 진행한 뒤 저장한 결과가 기준 파일과
# 바이트로 같은지 본다. 인자: -DCG_LAB=<실행 파일> -DSOURCE=<저장소 뿌리> -DOUT=<출력 JSON>.
#
# ponytail: 기준 파일은 같은 컴파일러·같은 부동소수 경로를 전제한다. 다른 플랫폼에서 갈리면 허용 오차 비교로 바꾼다.
execute_process(
    COMMAND "${CG_LAB}" --headless --open tests/scenes/rigid_cpu.json --play --frames 120 --save "${OUT}"
    WORKING_DIRECTORY "${SOURCE}"
    RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "헤드리스 실행이 실패했다 (종료 코드 ${run_result})")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${OUT}" "${SOURCE}/tests/scenes/expected/rigid_cpu_120.json"
    RESULT_VARIABLE compare_result)
if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR "헤드리스 결과가 기준 파일과 다르다: ${OUT} vs tests/scenes/expected/rigid_cpu_120.json")
endif()
