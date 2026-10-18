#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void d7_shortconv_f16_k3_v1(
    const __half *__restrict__ x,
    const __half *__restrict__ state,
    const __half *__restrict__ weight,
    __half *__restrict__ y,
    __half *__restrict__ next_state,
    uint32_t batch,
    uint32_t channels) {
    const uint64_t row = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= uint64_t(batch) * channels) return;
    const uint64_t base = 3 * row;
    const uint64_t wbase = 3 * (row % channels);
    const __half a = state[base + 1];
    const __half b = state[base + 2];
    const __half c = x[row];
    next_state[base] = a;
    next_state[base + 1] = b;
    next_state[base + 2] = c;
    // Match Candle's materialized F16 multiply followed by fast_sum_small_f16.
    const __half p0 = __float2half_rn(__fmul_rn(__half2float(a), __half2float(weight[wbase])));
    const __half p1 = __float2half_rn(__fmul_rn(__half2float(b), __half2float(weight[wbase + 1])));
    const __half p2 = __float2half_rn(__fmul_rn(__half2float(c), __half2float(weight[wbase + 2])));
    float sum = __fadd_rn(0.0f, __half2float(p0));
    sum = __fadd_rn(sum, __half2float(p1));
    sum = __fadd_rn(sum, __half2float(p2));
    y[row] = __float2half_rn(sum);
}
