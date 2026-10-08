#include <metal_stdlib>
using namespace metal;

template <typename T, typename A>
METAL_FUNC void grouped_conv_transpose1d(
    constant size_t &l_out,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &out_padding,
    constant size_t &dilation,
    constant size_t &groups,
    constant size_t *src_dims,
    constant size_t *src_strides,
    constant size_t *k_dims,
    constant size_t *k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid [[thread_position_in_grid]]) {
  const size_t l_k = k_dims[2];
  const size_t c_out_group = k_dims[1];
  const size_t c_out = c_out_group * groups;
  const size_t c_in = src_dims[1];
  const size_t c_in_group = c_in / groups;
  const size_t l_in = src_dims[2];
  if (tid >= src_dims[0] * c_out * l_out) return;

  const size_t b_idx = tid / (l_out * c_out);
  const size_t dst_c_idx = (tid / l_out) % c_out;
  const size_t dst_c_local = dst_c_idx % c_out_group;
  const size_t group = dst_c_idx / c_out_group;
  const size_t src_c_begin = group * c_in_group;
  const size_t src_c_end = src_c_begin + c_in_group;
  const size_t out_x = tid % l_out;
  const size_t src_idx0 = b_idx * src_strides[0];

  A d = 0;
  for (int k_x = 0; k_x < (int)l_k; ++k_x) {
    int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
    if (inp_x_stride < 0 || inp_x_stride % stride) continue;
    int inp_x = inp_x_stride / stride;
    if (inp_x >= l_in) continue;
    for (size_t src_c_idx = src_c_begin; src_c_idx < src_c_end; ++src_c_idx) {
      const size_t src_idx = src_idx0 + src_c_idx * src_strides[1] + inp_x * src_strides[2];
      const size_t k_idx = src_c_idx * k_strides[0] + dst_c_local * k_strides[1] + k_x * k_strides[2];
      d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
    }
  }
  dst[tid] = static_cast<T>(d);
}

template <typename T, typename A>
METAL_FUNC void grouped_conv_transpose2d(
    constant size_t &w_out,
    constant size_t &h_out,
    constant size_t &stride,
    constant size_t &padding,
    constant size_t &out_padding,
    constant size_t &dilation,
    constant size_t &groups,
    constant size_t *src_dims,
    constant size_t *src_strides,
    constant size_t *k_dims,
    constant size_t *k_strides,
    device const T *src,
    device const T *k,
    device T *dst,
    uint tid [[thread_position_in_grid]]) {
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out_group = k_dims[1];
  const size_t c_out = c_out_group * groups;
  const size_t c_in = src_dims[1];
  const size_t c_in_group = c_in / groups;
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];
  if (tid >= src_dims[0] * c_out * w_out * h_out) return;

  const size_t b_idx = tid / (w_out * h_out * c_out);
  const size_t dst_c_idx = (tid / (w_out * h_out)) % c_out;
  const size_t dst_c_local = dst_c_idx % c_out_group;
  const size_t group = dst_c_idx / c_out_group;
  const size_t src_c_begin = group * c_in_group;
  const size_t src_c_end = src_c_begin + c_in_group;
  const size_t out_y = (tid / w_out) % h_out;
  const size_t out_x = tid % w_out;
  const size_t src_idx0 = b_idx * src_strides[0];

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
        const size_t src_idx = src_idx0 + src_c_idx * src_strides[1] + inp_y * src_strides[2] + inp_x * src_strides[3];
        const size_t k_idx = src_c_idx * k_strides[0] + dst_c_local * k_strides[1] + k_y * k_strides[2] + k_x * k_strides[3];
        d += static_cast<A>(src[src_idx]) * static_cast<A>(k[k_idx]);
      }
    }
  }
  dst[tid] = static_cast<T>(d);
}

#define GROUPED_CONVT1D_OP(T, A, NAME) \
kernel void NAME(constant size_t &l_out, constant size_t &stride, constant size_t &padding, \
    constant size_t &out_padding, constant size_t &dilation, constant size_t &groups, \
    constant size_t *src_dims, constant size_t *src_strides, constant size_t *k_dims, \
    constant size_t *k_strides, device const T *src, device const T *k, device T *dst, \
    uint tid [[thread_position_in_grid]]) { \
  grouped_conv_transpose1d<T, A>(l_out, stride, padding, out_padding, dilation, groups, src_dims, src_strides, k_dims, k_strides, src, k, dst, tid); \
}

#define GROUPED_CONVT2D_OP(T, A, NAME) \
kernel void NAME(constant size_t &w_out, constant size_t &h_out, constant size_t &stride, \
    constant size_t &padding, constant size_t &out_padding, constant size_t &dilation, \
    constant size_t &groups, constant size_t *src_dims, constant size_t *src_strides, \
    constant size_t *k_dims, constant size_t *k_strides, device const T *src, device const T *k, \
    device T *dst, uint tid [[thread_position_in_grid]]) { \
  grouped_conv_transpose2d<T, A>(w_out, h_out, stride, padding, out_padding, dilation, groups, src_dims, src_strides, k_dims, k_strides, src, k, dst, tid); \
}

GROUPED_CONVT1D_OP(float, float, grouped_conv_transpose1d_f32)
GROUPED_CONVT1D_OP(half, float, grouped_conv_transpose1d_f16)
GROUPED_CONVT1D_OP(uint8_t, uint8_t, grouped_conv_transpose1d_u8)
GROUPED_CONVT1D_OP(uint32_t, uint32_t, grouped_conv_transpose1d_u32)
#if defined(__HAVE_BFLOAT__)
GROUPED_CONVT1D_OP(bfloat, float, grouped_conv_transpose1d_bf16)
#endif

GROUPED_CONVT2D_OP(float, float, grouped_conv_transpose2d_f32)
GROUPED_CONVT2D_OP(half, float, grouped_conv_transpose2d_f16)
#if defined(__HAVE_BFLOAT__)
GROUPED_CONVT2D_OP(bfloat, float, grouped_conv_transpose2d_bf16)
#endif
