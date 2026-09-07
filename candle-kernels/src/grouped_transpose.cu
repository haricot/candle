#include "cuda_utils.cuh"
#include <stdint.h>

template <typename T, typename A>
__device__ void grouped_conv_transpose1d(
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t groups,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t *k_dims = info + 6;
  const size_t *k_s = info + 9;
  const size_t l_k = k_dims[2];
  const size_t c_out_group = k_dims[1];
  const size_t c_out = c_out_group * groups;
  const size_t c_in = src_dims[1];
  const size_t c_in_group = c_in / groups;
  const size_t l_in = src_dims[2];
  if (dst_i >= src_dims[0] * c_out * l_out) return;

  const size_t b_idx = dst_i / (l_out * c_out);
  const size_t dst_c_idx = (dst_i / l_out) % c_out;
  const size_t dst_c_local = dst_c_idx % c_out_group;
  const size_t group = dst_c_idx / c_out_group;
  const size_t src_c_begin = group * c_in_group;
  const size_t src_c_end = src_c_begin + c_in_group;
  const size_t out_x = dst_i % l_out;
  const size_t src_idx0 = b_idx * src_s[0];

  A d = 0;
  for (int k_x = 0; k_x < (int)l_k; ++k_x) {
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) continue;
      int inp_x = inp_x_stride / stride;
      if (inp_x >= l_in) continue;
      for (size_t src_c_idx = src_c_begin; src_c_idx < src_c_end; ++src_c_idx) {
          const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_x * src_s[2];
          const size_t k_idx = src_c_idx * k_s[0] + dst_c_local * k_s[1] + k_x * k_s[2];
          d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
      }
  }
  dst[dst_i] = static_cast<T>(d);
}

template <typename T, typename A>
__device__ void grouped_conv_transpose2d(
    const size_t w_out,
    const size_t h_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t groups,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;
  const size_t *k_dims = info + 8;
  const size_t *k_s = info + 12;
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out_group = k_dims[1];
  const size_t c_out = c_out_group * groups;
  const size_t c_in = src_dims[1];
  const size_t c_in_group = c_in / groups;
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];
  if (dst_i >= src_dims[0] * c_out * w_out * h_out) return;

  const size_t b_idx = dst_i / (w_out * h_out * c_out);
  const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
  const size_t dst_c_local = dst_c_idx % c_out_group;
  const size_t group = dst_c_idx / c_out_group;
  const size_t src_c_begin = group * c_in_group;
  const size_t src_c_end = src_c_begin + c_in_group;
  const size_t out_y = (dst_i / w_out) % h_out;
  const size_t out_x = dst_i % w_out;
  const size_t src_idx0 = b_idx * src_s[0];

  A d = 0;
  for (int k_x = 0; k_x < (int)w_k; ++k_x) {
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) continue;
      int inp_x = inp_x_stride / stride;
      if (inp_x >= w_in) continue;
      for (int k_y = 0; k_y < (int)h_k; ++k_y) {
          int inp_y_stride = (int)(out_y + padding) - k_y * dilation;
          if (inp_y_stride < 0 || inp_y_stride % stride) continue;
          int inp_y = inp_y_stride / stride;
          if (inp_y >= h_in) continue;
          for (size_t src_c_idx = src_c_begin; src_c_idx < src_c_end; ++src_c_idx) {
              const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_y * src_s[2] + inp_x * src_s[3];
              const size_t k_idx = src_c_idx * k_s[0] + dst_c_local * k_s[1] + k_y * k_s[2] + k_x * k_s[3];
              d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
          }
      }
  }
  dst[dst_i] = static_cast<T>(d);
}

#define GROUPED_CONVT1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t l_out, const size_t stride, const size_t padding, \
    const size_t out_padding, const size_t dilation, const size_t groups, \
    const size_t *info, const TYPENAME *src, const TYPENAME *kernel, TYPENAME *dst) { \
  grouped_conv_transpose1d<TYPENAME, TYPEACC>(l_out, stride, padding, out_padding, dilation, groups, info, src, kernel, dst); \
}

#define GROUPED_CONVT2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t w_out, const size_t h_out, const size_t stride, const size_t padding, \
    const size_t out_padding, const size_t dilation, const size_t groups, \
    const size_t *info, const TYPENAME *src, const TYPENAME *kernel, TYPENAME *dst) { \
  grouped_conv_transpose2d<TYPENAME, TYPEACC>(w_out, h_out, stride, padding, out_padding, dilation, groups, info, src, kernel, dst); \
}

#if __CUDA_ARCH__ >= 800 || defined(CANDLE_CUDA_BF16_FALLBACK)
GROUPED_CONVT1D_OP(__nv_bfloat16, float, grouped_conv_transpose1d_bf16)
GROUPED_CONVT2D_OP(__nv_bfloat16, float, grouped_conv_transpose2d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
GROUPED_CONVT1D_OP(__half, float, grouped_conv_transpose1d_f16)
GROUPED_CONVT2D_OP(__half, float, grouped_conv_transpose2d_f16)
#endif

GROUPED_CONVT1D_OP(float, float, grouped_conv_transpose1d_f32)
GROUPED_CONVT1D_OP(double, double, grouped_conv_transpose1d_f64)
GROUPED_CONVT1D_OP(uint8_t, uint8_t, grouped_conv_transpose1d_u8)
GROUPED_CONVT1D_OP(uint32_t, uint32_t, grouped_conv_transpose1d_u32)

GROUPED_CONVT2D_OP(float, float, grouped_conv_transpose2d_f32)
GROUPED_CONVT2D_OP(double, double, grouped_conv_transpose2d_f64)
GROUPED_CONVT2D_OP(uint8_t, uint8_t, grouped_conv_transpose2d_u8)
GROUPED_CONVT2D_OP(uint32_t, uint32_t, grouped_conv_transpose2d_u32)
