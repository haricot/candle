extern "C" __global__ void asd_dw5x5_conv2d_f32(
    const unsigned long long channels,
    const unsigned long long height,
    const unsigned long long width,
    const float *src,
    const float *weight,
    float *dst
) {
    constexpr int BX = 16;
    constexpr int BY = 8;
    constexpr int R = 2;
    constexpr int TW = BX + 2 * R;
    constexpr int TH = BY + 2 * R;

    __shared__ float tile[TH * TW];
    __shared__ float filter[25];

    const int tx = (int)threadIdx.x;
    const int ty = (int)threadIdx.y;
    const int tid = ty * BX + tx;
    const unsigned long long channel = (unsigned long long)blockIdx.z;
    if (channel >= channels) {
        return;
    }

    const int base_y = (int)blockIdx.y * BY;
    const int base_x = (int)blockIdx.x * BX;

    for (int i = tid; i < TH * TW; i += BX * BY) {
        const int ly = i / TW;
        const int lx = i - ly * TW;
        const int iy = base_y + ly - R;
        const int ix = base_x + lx - R;
        float value = 0.0f;
        if ((unsigned)iy < (unsigned)height && (unsigned)ix < (unsigned)width) {
            const unsigned long long src_i =
                (channel * height + (unsigned long long)iy) * width +
                (unsigned long long)ix;
            value = __ldg(src + src_i);
        }
        tile[i] = value;
    }

    if (tid < 25) {
        filter[tid] = __ldg(weight + channel * 25ull + (unsigned long long)tid);
    }
    __syncthreads();

    const int oy = base_y + ty;
    const int ox = base_x + tx;
    if ((unsigned)oy >= (unsigned)height || (unsigned)ox >= (unsigned)width) {
        return;
    }

    float acc = 0.0f;
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
            acc += tile[(ty + ky) * TW + tx + kx] * filter[ky * 5 + kx];
        }
    }

    const unsigned long long dst_i =
        (channel * height + (unsigned long long)oy) * width +
        (unsigned long long)ox;
    dst[dst_i] = acc;
}
