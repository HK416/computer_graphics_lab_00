# 신경망 GPU 자기 검사. cg_lab 을 --neural-selfcheck 로 돌려 CPU 기준과 GPU 커널을 견준다. 실행 파일이
# 연산 종류마다 최대 오차를 찍고, 허용치를 넘으면 0 이 아닌 코드로 끝난다.
#
# Vulkan 을 타는 코드에는 단위 테스트가 없다는 이 저장소의 규칙 안에서, 그래도 «두 엔진이 같은 답을
# 내는가» 만은 자동으로 볼 수 있게 하는 자리다. 인자: -DCG_LAB=<실행 파일> -DSOURCE=<저장소 뿌리>.
#
# ponytail: GPU 가 없는 기계에서는 장치를 만들지 못해 실패한다. 그런 환경에서 돌릴 일이 생기면 «장치가
# 없으면 건너뛴다» 를 넣어야 한다.
execute_process(
    COMMAND "${CG_LAB}" --neural-selfcheck
    WORKING_DIRECTORY "${SOURCE}"
    RESULT_VARIABLE run_result)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "신경망 자기 검사가 실패했다 (종료 코드 ${run_result})")
endif()
