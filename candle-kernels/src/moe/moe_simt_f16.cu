/**
 * Grouped MoE GEMM for pre-Volta GPUs.
 *
 * FP16 is used only for storage and vectorized memory traffic. Products and
 * accumulation are promoted to FP32 so GP104 avoids its 1/64-rate FP16 path.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include "moe_utils.cuh"

namespace vllm_rs {

#define CEILDIV(x,y) (((x) + (y) - 1) / (y))

// Tile sizes optimized for SM 5.x/6.x without Tensor Cores
constexpr int M_BLK = 32;
constexpr int N_BLK = 32;
constexpr int K_BLK = 16;
constexpr int BLOCK_THREADS = 128;

// Thread block configuration: 8 groups x 16 threads
// Each group handles 4 rows, each thread handles 2 columns
constexpr int GROUPS_PER_BLOCK = 8;
constexpr int THREADS_PER_GROUP = 16;

// Vectorized load type
using VecT = float4;
constexpr int VEC_SIZE = 8; // 8 half values per float4

/**
 * @brief Vectorized dot product using FP32 arithmetic after vectorized FP16 loads
 */
static __device__ __forceinline__ void simt_fma2(float &acc, half2 a, half2 b) {
    const float2 af = __half22float2(a);
    const float2 bf = __half22float2(b);
    acc = fmaf(af.x, bf.x, acc);
    acc = fmaf(af.y, bf.y, acc);
}

/**
 * @brief Optimized Non-WMMA MoE GEMM kernel using vectorized arithmetic.
 *
 * Key optimizations:
 * - Larger tile (32x32 vs 16x32) for better arithmetic intensity
 * - Vectorized loads (float4 = 8 half values at once)
 * - Padding to avoid bank conflicts
 * - Each thread computes a 4x2 output tile
 */
__global__ void moe_gemm_simt_f16_kernel(
    const half* __restrict__ input,           // [num_tokens, size_k] or [num_tokens/topk, size_k]
    const half* __restrict__ weights,         // [num_experts, size_n, size_k]
    const int32_t* __restrict__ sorted_token_ids, // [size_m]
    const int32_t* __restrict__ expert_offsets,   // [num_experts+1]
    const float* __restrict__ topk_weights, // [size_m] (can be nullptr)
    half* __restrict__ output,                 // [size_m, size_n]
    const int num_experts, const int topk,
    const int32_t size_m,
    const int32_t size_n,
    const int32_t size_k
) {
    const int expert_id = blockIdx.x;
    const int n_tile_idx = blockIdx.y;
    if (expert_id >= num_experts) return;

    const int segment_start = expert_offsets[expert_id];
    const int segment_end = expert_offsets[expert_id + 1];
    const int num_rows_in_segment = segment_end - segment_start;
    if (num_rows_in_segment <= 0) return;

    const int n_base = n_tile_idx * N_BLK;
    if (n_base >= size_n) return;

    const half* expert_w = weights + (size_t)expert_id * (size_t)size_n * (size_t)size_k;

    // Shared memory layout with padding to avoid bank conflicts
    // A_sh: [M_BLK][K_BLK + 2] - extra padding for bank conflict avoidance
    // B_sh: [N_BLK][K_BLK + 2]
    constexpr int K_PAD = K_BLK + 2; // Padding to avoid 32-bank conflicts
    extern __shared__ uint8_t smem_bytes[];
    half* A_sh = reinterpret_cast<half*>(smem_bytes);
    half* B_sh = reinterpret_cast<half*>(A_sh + M_BLK * K_PAD);

    const int tid = threadIdx.x;

    // Zero vector for boundary handling
    VecT zero_vec;
    zero_vec.x = zero_vec.y = zero_vec.z = zero_vec.w = 0.0f;

    for (int m_base = 0; m_base < num_rows_in_segment; m_base += M_BLK) {
        // Per-thread accumulator: 4 rows x 2 cols = 8 values
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int k_base = 0; k_base < size_k; k_base += K_BLK) {
            // ==== Load A Tile (Input activations) ====
            // 128 threads load M_BLK * K_BLK = 32 * 16 = 512 elements
            // Using half2 loads (4-byte aligned) for safety: 512 / 2 = 256 loads
            // With 128 threads, each thread loads 2 half2 values
            constexpr int A_ELEMS = M_BLK * K_BLK;
            #pragma unroll
            for (int i = tid; i < A_ELEMS / 2; i += BLOCK_THREADS) {
                int idx = i * 2;  // Element index (in half units)
                int m_local = idx / K_BLK;
                int k_local = idx % K_BLK;
                int m_seg = m_base + m_local;
                int k_global = k_base + k_local;

                // Destination in shared mem with padding
                half* dest = &A_sh[m_local * K_PAD + k_local];
                half2 val;

                if (m_seg < num_rows_in_segment && (k_global + 1) < size_k) {
                    int token_pair_index = segment_start + m_seg;
                    int token_index = sorted_token_ids[token_pair_index];
                    int input_index = topk_weights ? token_index : (token_index / topk);
                    val = *reinterpret_cast<const half2*>(&input[(size_t)input_index * size_k + k_global]);
                } else if (m_seg < num_rows_in_segment && k_global < size_k) {
                    // Edge case: only one element valid
                    reinterpret_cast<half*>(&val)[0] = input[(size_t)(topk_weights ? sorted_token_ids[segment_start + m_seg] : (sorted_token_ids[segment_start + m_seg] / topk)) * size_k + k_global];
                    reinterpret_cast<half*>(&val)[1] = half(0.0f);
                } else {
                    val = __float2half2_rn(0.0f);
                }
                *reinterpret_cast<half2*>(dest) = val;
            }

            // ==== Load B Tile (Expert weights) ====
            constexpr int B_ELEMS = N_BLK * K_BLK;
            #pragma unroll
            for (int i = tid; i < B_ELEMS / 2; i += BLOCK_THREADS) {
                int idx = i * 2;
                int n_local = idx / K_BLK;
                int k_local = idx % K_BLK;
                int n_global = n_base + n_local;
                int k_global = k_base + k_local;

                half* dest = &B_sh[n_local * K_PAD + k_local];
                half2 val;

                if (n_global < size_n && (k_global + 1) < size_k) {
                    val = *reinterpret_cast<const half2*>(&expert_w[(size_t)n_global * size_k + k_global]);
                } else if (n_global < size_n && k_global < size_k) {
                    reinterpret_cast<half*>(&val)[0] = expert_w[(size_t)n_global * size_k + k_global];
                    reinterpret_cast<half*>(&val)[1] = half(0.0f);
                } else {
                    val = __float2half2_rn(0.0f);
                }
                *reinterpret_cast<half2*>(dest) = val;
            }
            __syncthreads();

            // ==== Compute using half2 vectorized arithmetic ====
            // Thread mapping: 128 threads = 8 groups x 16 threads
            // Each group handles 4 consecutive M rows
            // Each thread handles 2 consecutive N columns
            // => Each thread computes a 4x2 tile
            const int group_id = tid / THREADS_PER_GROUP;  // 0-7
            const int lane_id = tid % THREADS_PER_GROUP;   // 0-15

            const int m_start = group_id * 4;  // Row offset within tile
            const int n_start = lane_id * 2;   // Column offset within tile

            // Cast to half2 for vectorized compute
            const half2* A2_sh = reinterpret_cast<const half2*>(A_sh);
            const half2* B2_sh = reinterpret_cast<const half2*>(B_sh);

            // K loop in half2 steps
            #pragma unroll
            for (int k2 = 0; k2 < K_BLK / 2; ++k2) {
                // Load 4 values from A (4 rows, same k)
                half2 a2[4];
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    a2[r] = A2_sh[(m_start + r) * (K_PAD / 2) + k2];
                }

                // Load 2 values from B (2 columns, same k)
                half2 b2[2];
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    b2[c] = B2_sh[(n_start + c) * (K_PAD / 2) + k2];
                }

                // Compute 4x2 = 8 dot products
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    #pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        simt_fma2(acc[r * 2 + c], a2[r], b2[c]);
                    }
                }
            }
            __syncthreads();
        } // end k_base loop

        // ==== Store results ====
        const int group_id = tid / THREADS_PER_GROUP;
        const int lane_id = tid % THREADS_PER_GROUP;
        const int m_start = group_id * 4;
        const int n_start = lane_id * 2;

        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int m_seg = m_base + m_start + r;
            if (m_seg >= num_rows_in_segment) continue;

            int token_pair_index = segment_start + m_seg;
            int token_index = sorted_token_ids[token_pair_index];

            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                int n_global = n_base + n_start + c;
                if (n_global >= size_n) continue;

                float val = acc[r * 2 + c];
                if (topk_weights) {
                    val *= topk_weights[token_index];
                }
                output[(size_t)token_index * size_n + n_global] = __float2half_rn(val);
            }
        }
    } // end m_base loop
}

} // namespace vllm_rs

extern "C" void moe_gemm_simt_f16(
    const void* input,
    const void* weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const float* topk_weights,
    void* output,
    int32_t* expert_counts,
    int32_t* expert_offsets,
    int num_experts,
    int topk,
    int size_m,
    int size_n,
    int size_k
    bool is_prefill,
    cudaStream_t stream
) {
    using namespace vllm_rs;

    if (is_prefill) {
        calculate_expert_offsets(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    } else {
        calculate_expert_offsets_light(expert_ids, size_m, expert_counts, expert_offsets, num_experts, stream);
    }

    int grid_n = CEILDIV(size_n, N_BLK);
    dim3 grid(num_experts, grid_n, 1);
    dim3 block(BLOCK_THREADS, 1, 1);

    // Shared memory with padding: A[M_BLK][K_PAD] + B[N_BLK][K_PAD]
    constexpr int K_PAD = K_BLK + 2;
    size_t smem_bytes = (M_BLK * K_PAD + N_BLK * K_PAD) * sizeof(half);

    vllm_rs::moe_gemm_simt_f16_kernel<<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const half*>(input),
        reinterpret_cast<const half*>(weights),
        sorted_token_ids, expert_offsets, topk_weights,
        reinterpret_cast<half*>(output),
        num_experts, topk, size_m, size_n, size_k
    );

}

#ifdef NO_WMMA_KERNEL
extern "C" void moe_gemm_wmma(
    const void* input, const void* weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const float* topk_weights, void* output,
    int32_t* expert_counts, int32_t* expert_offsets,
    int num_experts, int topk, int size_m, int size_n, int size_k,
    int data_type, bool is_prefill, cudaStream_t stream
) {
    if (data_type == 0) {
        moe_gemm_simt_f16(
            input, weights, sorted_token_ids, expert_ids, topk_weights, output,
            expert_counts, expert_offsets, num_experts, topk, size_m, size_n,
            size_k, is_prefill, stream
        );
    }
}
#endif
