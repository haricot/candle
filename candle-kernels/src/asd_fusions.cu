// ASD-only fusion: deliberately separate from legacy BF16 / FP8 binary.cu.
#include <cuda_runtime.h>

extern "C" __global__ void conv2d_bias_silu_f32(
    const unsigned long long total,
    const unsigned long long channels,
    const unsigned long long spatial,
    const float *src,
    const float *bias,
    float *dst
) {
    const unsigned long long i =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const unsigned long long c = (i / spatial) % channels;
    const float x = src[i] + __ldg(bias + c);
    dst[i] = x / (1.0f + expf(-x));
}
