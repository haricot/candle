#pragma once

#include <type_traits>

template <typename... Args>
static cudaError_t mmq_launch_kernel(
    void (*kernel)(Args...), dim3 grid, dim3 block, size_t shared_bytes,
    cudaStream_t stream, std::common_type_t<Args>... args) {
    const auto entry = reinterpret_cast<const void *>(kernel);
    // NVIDIA kernels only need opt-in for dynamic shared memory above 48 KiB.
    if (shared_bytes > 48 * 1024) {
        const auto status = cudaFuncSetAttribute(
            entry, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes));
        if (status != cudaSuccess) {
            return status;
        }
    }

    // Args are deduced from the kernel signature, including the device-side integer widths.
    void *params[] = { &args... };
    return cudaLaunchKernel(entry, grid, block, params, shared_bytes, stream);
}

template <ggml_type type, int tile_x>
static cudaError_t mmq_launch_tile(
    const mmq_args & args, float *fixup, cudaStream_t stream,
    int tile_y, int cc, int nsm, int warp_size) {
    const int nwarps = 256 / warp_size;
    const auto smem = mmq_get_nbytes_shared<type>(tile_x, tile_y, cc, warp_size, nwarps);
    const int nty = (args.nrows_x + tile_y - 1) / tile_y;
    const int ntx = (args.ncols_max + tile_x - 1) / tile_x;
    const int ntzw = args.nchannels_y * args.nsamples_y;
    const dim3 grid = args.use_stream_k ? dim3(nsm, 1, 1) : dim3(nty, ntx, ntzw);
    const dim3 block(warp_size, nwarps, 1);
    const bool needs_fixup = args.use_stream_k && ntx * nty * ntzw % nsm != 0;
    auto *tile_fixup = args.use_stream_k ? fixup : nullptr;
    const bool full_rows = args.nrows_x % tile_y == 0;
    const auto matmul = full_rows ? mul_mat_q<type, tile_x, false> : mul_mat_q<type, tile_x, true>;
    const int channel_ratio = args.nchannels_y / args.nchannels_x;
    const int sample_ratio = args.nsamples_y / args.nsamples_x;
    const auto status = mmq_launch_kernel(
        matmul, grid, block, smem, stream,
        args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, tile_fixup,
        args.ncols_x, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
        channel_ratio, args.nchannels_y, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
        sample_ratio, args.nsamples_y, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
        args.ncols_max);
    if (status != cudaSuccess || !needs_fixup) {
        return status;
    }
    const auto fixup_kernel = full_rows ?
        mul_mat_q_stream_k_fixup<type, tile_x, false> : mul_mat_q_stream_k_fixup<type, tile_x, true>;
    return mmq_launch_kernel(
        fixup_kernel, grid, block, 0, stream,
        args.ids_dst, args.expert_bounds, args.dst, fixup, args.ncols_x, args.nrows_x, args.ncols_dst,
        args.nrows_dst, args.nchannels_y, args.stride_channel_dst, args.nsamples_y,
        args.stride_sample_dst, args.ncols_max);
}

template <ggml_type type>
static cudaError_t mmq_launch_dense(
    void *tmp_fixup, const void *x, const void *y, void *dst,
    int64_t ncols_x, int64_t nrows_x, int64_t ncols_y, int64_t stride_row_x,
    int cc, int nsm, size_t smpbo, int warp_size, cudaStream_t stream) {
    const bool use_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) &&
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;
    const mmq_args args = {
        static_cast<const char *>(x), type, static_cast<const int *>(y), nullptr, nullptr, static_cast<float *>(dst),
        ncols_x, nrows_x, ncols_y, stride_row_x, ncols_y, nrows_x,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, ncols_y
    };
    const int tile_y = use_stream_k ? 128 : 64;
    const bool turing = turing_mma_available(cc);
    const int max_tile_x = turing ? 128 : 64;
    const int nwarps = 256 / warp_size;
    int best_tile_x = 0;
    int best_tiles = INT_MAX;
    for (int tile_x = 8; tile_x <= max_tile_x && best_tiles > 1; tile_x += 8) {
        if (turing && tile_x >= 48 && tile_x % 16 != 0) {
            continue;
        }
        if (mmq_get_nbytes_shared<type>(tile_x, tile_y, cc, warp_size, nwarps) > smpbo) {
            continue;
        }
        const int tiles = (args.ncols_max + tile_x - 1) / tile_x;
        if (tiles < best_tiles) {
            best_tile_x = tile_x;
            best_tiles = tiles;
        }
    }

    auto *fixup = static_cast<float *>(tmp_fixup);
#define CANDLE_MMQ_TILE(width) case width: return mmq_launch_tile<type, width>(args, fixup, stream, tile_y, cc, nsm, warp_size)
    switch (best_tile_x) {
        CANDLE_MMQ_TILE(8);
        CANDLE_MMQ_TILE(16);
        CANDLE_MMQ_TILE(24);
        CANDLE_MMQ_TILE(32);
        CANDLE_MMQ_TILE(40);
        CANDLE_MMQ_TILE(48);
        CANDLE_MMQ_TILE(56);
        CANDLE_MMQ_TILE(64);
        CANDLE_MMQ_TILE(72);
        CANDLE_MMQ_TILE(80);
        CANDLE_MMQ_TILE(88);
        CANDLE_MMQ_TILE(96);
        CANDLE_MMQ_TILE(104);
        CANDLE_MMQ_TILE(112);
        CANDLE_MMQ_TILE(120);
        CANDLE_MMQ_TILE(128);
        default: return cudaErrorInvalidConfiguration;
    }
#undef CANDLE_MMQ_TILE
}
