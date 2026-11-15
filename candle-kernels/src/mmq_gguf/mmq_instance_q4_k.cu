#include "mmq_common.cuh"
#include "mmq_gguf.cuh"
#include "mmq_launch.cuh"

extern "C" int candle_launch_mmq_gguf_q4_k(
    void *tmp_fixup, const void *x, const void *y, void *dst,
    int64_t ncols_x, int64_t nrows_x, int64_t ncols_y,
    int64_t stride_row_x, int64_t /*stride_col_dst*/,
    int cc, int nsm, int64_t smpbo, int warp_size, void *stream) {
    return mmq_launch_dense<GGML_TYPE_Q4_K>(
        tmp_fixup, x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x,
        cc, nsm, smpbo, warp_size, static_cast<cudaStream_t>(stream));
}
