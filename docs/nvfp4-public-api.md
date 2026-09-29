# Explicit NVFP4 weights and legacy-CUDA dispatch

This branch adds the public Rust type NvFp4Weights in candle_core::quantized::nvfp4, without introducing NVFP4 into GgmlDType. Existing GGUF MXFP4 remains available under ordinary CUDA, whereas the experimental NVFP4 software kernels require the cuda-legacy-fp4 feature.

The host constructor is NvFp4Weights::new(shape, packed_e2m1, scales_e4m3, global_scale). Dense shape is [n, k] and MoE shape is [experts, n, k]. Every dimension must be positive, k must be divisible by 32, each 16 weights require eight low-nibble-first E2M1 bytes and one non-negative finite E4M3FN block scale, and the F32 global scale must be positive and finite. The constructor validates exact payload lengths and legacy CUDA integer limits. The global scale is MULTIPLICATIVE: loaders for checkpoints that store its reciprocal must invert it.

The CUDA API, compiled only under cuda-legacy-fp4, consists of NvFp4Weights::upload(&CudaDevice), NvFp4Cuda::forward(&Tensor), and NvFp4Cuda::indexed_moe(&Tensor, &Tensor). The dense input is [.., k] with rank >= 2, output [.., n] F32; batch sizes 1 to 8 use GEMV and larger batches use prefill. Inputs may be F32, F16 or BF16 (provided the configured GPU supports conversion to F32). Noncontiguous inputs are materialized. No autograd path is provided.

The indexed-MoE input is [batch, k] and U32 CUDA expert indices [batch, topk], producing F32 [batch, topk, n]. Every expert ID is checked before launching the kernel to prevent out-of-bounds device reads. That check transfers the indices to the CPU, so indexed_moe is not yet compatible with CUDA graph capture. A future GPU-side checked-index path can eliminate this synchronization. Separately quantized gate/up projections can be represented by separate NvFp4Weights, preserving their independent global scales.

This API imports native packed checkpoint payloads; the test-only experimental NVFP4 quantizer is not guaranteed to match NVIDIA ModelOpt's calibration byte for byte. MLX U32-packed NVFP4 and hardware Blackwell FP4 are out of scope.

Validation must run on the exact candidate SHA: CPU payload validation, hosted CUDA 12.9.2 SM61 compile, physical GPU API tests, existing MXFP4 parity and CUDA NVFP4 numerical tests. The previously promoted integrated campaign validated the pre-existing kernels, not this API.

CPU test:

    cargo test -p candle-core --no-default-features --test nvfp4_api

Physical SM61 GPU test (in the existing CUDA 12.9.2 container):

    export CUDA_COMPUTE_CAP=61
    export NVCC_CCBIN=/usr/bin/g++-14
    cargo test -p candle-core --features cuda-legacy-fp4 --test nvfp4_api -- --nocapture

Do not promote this development branch into the validated FP4 branch without the GPU test result.