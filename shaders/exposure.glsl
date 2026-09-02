#ifndef EXPOSURE_GLSL
#define EXPOSURE_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// 자동 노출이 쓰는 버퍼. 히스토그램은 프레임마다 0 으로 채워지고, 노출은 프레임을 넘어 적응한다.
#define HISTOGRAM_BINS 256u

layout(buffer_reference, scalar) buffer HistogramBuffer {
    uint items[HISTOGRAM_BINS];
};

layout(buffer_reference, scalar) buffer ExposureBuffer {
    // 장면 휘도에서 정한 노출 배율. 사용자의 노출 값에 곱한다.
    float exposure;
};

float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

#endif
