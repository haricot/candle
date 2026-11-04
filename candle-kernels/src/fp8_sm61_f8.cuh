// Haricot F8A/F8B/F8C experimental software E4M3FN linear, Pascal sm_61.
// Same definitions are used by Candle QUANTIZED PTX and standalone benchmark.
// No native FP8 operations; __dp4a only operates on separately requantized i8.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

__device__ __forceinline__ float haricot_e4m3fn_decode(uint8_t code) {
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) return nanf("");
    const float magnitude = exponent == 0 ? ldexpf((float)mantissa, -9)
                                          : ldexpf((float)(8 + mantissa), exponent - 10);
    return code & 0x80 ? -magnitude : magnitude;
}
__device__ __forceinline__ float haricot_reduce_warp_f32(float x) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        x += __shfl_down_sync(0xffffffff, x, delta);
    return x;
}
__device__ __forceinline__ int haricot_reduce_warp_i32(int x) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        x += __shfl_down_sync(0xffffffff, x, delta);
    return x;
}
__device__ __forceinline__ float haricot_scale(const float* scales,
       int outrow, int col, int bn, int bk, int k) {
    const int blocks_per_row = k / bk + (k % bk != 0);
    return scales[(outrow/bn)*blocks_per_row + col/bk];
}

// Decode: one warp per output, no intermediate fp16 weights.
extern "C" __global__ void fp8_sm61_w8a16_gemv(
    const uint8_t* __restrict__ w, const float* __restrict__ scales,
    const __half* __restrict__ x, float* __restrict__ out,
    int m, int n, int k, int bn, int bk) {
    const int row = blockIdx.x*4 + threadIdx.x/32;
    const int token = blockIdx.y;
    const int lane = threadIdx.x%32;
    if (row >= n || token >= m) return;
    float acc=0.0f;
    for(int col=lane;col<k;col+=32) {
        float fw=haricot_e4m3fn_decode(w[(size_t)row*k+col]);
        float sx=haricot_scale(scales,row,col,bn,bk,k);
        acc=fmaf(fw*sx,__half2float(x[(size_t)token*k+col]),acc);
    }
    acc=haricot_reduce_warp_f32(acc);
    if(lane==0) out[(size_t)token*n+row]=acc;
}

// Prefill: weight bytes and scales reused across 4 input rows, 4 warps/block.
extern "C" __global__ void fp8_sm61_w8a16_prefill4(
    const uint8_t* __restrict__ w, const float* __restrict__ scales,
    const __half* __restrict__ x, float* __restrict__ out,
    int m, int n, int k, int bn, int bk) {
    const int row=blockIdx.x*4+threadIdx.x/32;
    const int lane=threadIdx.x%32;
    const int token_base=blockIdx.y*4;
    if(row>=n) return;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int col=lane;col<k;col+=32){
        float weight=haricot_e4m3fn_decode(w[(size_t)row*k+col]) *
                     haricot_scale(scales,row,col,bn,bk,k);
#pragma unroll
        for(int t=0;t<4;t++)
            if(token_base+t<m) acc[t]=fmaf(weight,__half2float(x[(size_t)(token_base+t)*k+col]),acc[t]);
    }
#pragma unroll
    for(int t=0;t<4;t++){
        float sum=haricot_reduce_warp_f32(acc[t]);
        if(lane==0 && token_base+t<m) out[(size_t)(token_base+t)*n+row]=sum;
    }
}

// F8C activation quantization, block groups 32/64/128. Scale shared by the block.
extern "C" __global__ void fp8_sm61_quant_x(
    const __half* __restrict__ x, int8_t* __restrict__ q,
    float* __restrict__ sx, int m,int k,int g) {
    __shared__ float v[128];
    const int lane=threadIdx.x;
    const int groups=k/g;
    const int block=blockIdx.x;
    const int token=block/groups;
    const int part=block%groups;
    const int elem=token*k+part*g+lane;
    v[lane]=(lane<g && token<m)?fabsf(__half2float(x[elem])):0.0f;
    __syncthreads();
    for(int delta=64;delta>0;delta>>=1){
        if(lane<delta) v[lane]=fmaxf(v[lane],v[lane+delta]);
        __syncthreads();
    }
    float scale=v[0]==0.f?0.f:v[0]/127.f;
    if(lane==0) sx[block]=scale;
    if(lane<g && token<m){
        float val=__half2float(x[elem]);
        int qi=scale==0.f?0:__float2int_rn(val/scale);
        qi=qi>127?127:(qi< -127?-127:qi);
        q[elem]=(int8_t)qi;
    }
}
__device__ __forceinline__ int haricot_dp4a(int a,int b,int prior) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a,b,prior);
#else
#pragma unroll
    for (int t=0;t<4;t++) prior += (int)(int8_t)(a>>(8*t)) * (int)(int8_t)(b>>(8*t));
    return prior;
#endif
}
extern "C" __global__ void fp8_sm61_dp4a_gemv(
    const int8_t* __restrict__ w, const float* __restrict__ sw,
    const int8_t* __restrict__ x, const float* __restrict__ sx,
    float* __restrict__ out,int m,int n,int k,int g) {
    const int row=blockIdx.x*4+threadIdx.x/32;
    const int token=blockIdx.y;
    const int lane=threadIdx.x%32;
    if(row>=n || token>=m) return;
    const int groups=k/g;
    float total=0.f;
    for(int group=0;group<groups;group++) {
        int part=0;
        for(int pos=lane*4;pos<g;pos+=128){
            const int wi=((row*k)+(group*g)+pos);
            const int xi=((token*k)+(group*g)+pos);
            // all offsets 4-byte aligned when k and g are multiples of 32
            int pw=*reinterpret_cast<const int*>(w+wi);
            int px=*reinterpret_cast<const int*>(x+xi);
            part=haricot_dp4a(pw,px,part);
        }
        part=haricot_reduce_warp_i32(part);
        if(lane==0) total=fmaf((float)part,sw[row*groups+group]*sx[token*groups+group],total);
    }
    if(lane==0) out[token*n+row]=total;
}
extern "C" __global__ void fp8_sm61_dp4a_prefill4(
    const int8_t* __restrict__ w, const float* __restrict__ sw,
    const int8_t* __restrict__ x, const float* __restrict__ sx,
    float* __restrict__ out,int m,int n,int k,int g) {
    const int row=blockIdx.x*4+threadIdx.x/32;
    const int lane=threadIdx.x%32;
    const int base=blockIdx.y*4;
    if(row>=n) return;
    const int groups=k/g;
    float total[4]={0.f,0.f,0.f,0.f};
    for(int group=0;group<groups;group++) {
        int parts[4]={0,0,0,0};
        for(int pos=lane*4;pos<g;pos+=128){
            int pw=*reinterpret_cast<const int*>(w+row*k+group*g+pos);
#pragma unroll
            for(int t=0;t<4;t++) if(base+t<m){
                int px=*reinterpret_cast<const int*>(x+(base+t)*k+group*g+pos);
                parts[t]=haricot_dp4a(pw,px,parts[t]);
            }
        }
#pragma unroll
        for(int t=0;t<4;t++){
            int part=haricot_reduce_warp_i32(parts[t]);
            if(lane==0 && base+t<m)
                total[t]=fmaf((float)part,sw[row*groups+group]*sx[(base+t)*groups+group],total[t]);
        }
    }
    if(lane==0) for(int t=0;t<4;t++) if(base+t<m)out[(base+t)*n+row]=total[t];
}

// F8E: 8-token prefill with one weight load reused for eight activations.
// Kept separate from prefill4 to benchmark register pressure vs reuse.
extern "C" __global__ void fp8_sm61_dp4a_prefill8(
    const int8_t* __restrict__ w, const float* __restrict__ sw,
    const int8_t* __restrict__ x, const float* __restrict__ sx,
    float* __restrict__ out,int m,int n,int k,int g) {
    const int row=blockIdx.x*4+threadIdx.x/32;
    const int lane=threadIdx.x%32;
    const int base=blockIdx.y*8;
    if(row>=n)return;
    const int groups=k/g;
    float total[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(int group=0;group<groups;group++) {
        int parts[8]={0,0,0,0,0,0,0,0};
        for(int pos=lane*4;pos<g;pos+=128){
            const int pw=*reinterpret_cast<const int*>(w+row*k+group*g+pos);
#pragma unroll
            for(int t=0;t<8;t++) if(base+t<m){
                const int px=*reinterpret_cast<const int*>(x+(base+t)*k+group*g+pos);
                parts[t]=haricot_dp4a(pw,px,parts[t]);
            }
        }
#pragma unroll
        for(int t=0;t<8;t++){
            const int part=haricot_reduce_warp_i32(parts[t]);
            if(lane==0 && base+t<m)
                total[t]=fmaf((float)part,sw[row*groups+group]*sx[(base+t)*groups+group],total[t]);
        }
    }
    if(lane==0) for(int t=0;t<8;t++) if(base+t<m) out[(base+t)*n+row]=total[t];
}
