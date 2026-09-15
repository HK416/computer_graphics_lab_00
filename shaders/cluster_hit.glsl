#ifndef CLUSTER_HIT_GLSL
#define CLUSTER_HIT_GLSL

// 클러스터 가속 구조(VK_NV_cluster_acceleration_structure) 히트의 클러스터 번호. CLUSTER_HITS 변종만 확장을
// 선언하므로(확장이 없는 장치에서는 그 SPIR-V 능력이 파이프라인 생성을 거부한다) 기본 변종은 «클러스터 아님»
// 을 뜻하는 -1 을 돌려준다. 클러스터가 아닌 하위 구조를 맞히면 확장도 -1(gl_ClusterIDNoneNV)을 준다.
#ifdef CLUSTER_HITS
#extension GL_NV_cluster_acceleration_structure : require
#define HIT_CLUSTER_ID gl_ClusterIDNV
#define QUERY_CLUSTER_ID(query, committed) rayQueryGetIntersectionClusterIdNV(query, committed)
#else
#define HIT_CLUSTER_ID (-1)
#define QUERY_CLUSTER_ID(query, committed) (-1)
#endif

#endif
