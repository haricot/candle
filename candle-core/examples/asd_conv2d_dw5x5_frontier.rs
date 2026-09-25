#[cfg(all(feature = "cuda", feature = "cudnn"))]
mod enabled {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::{cudarc, WrapErr};
    use candle_core::{
        CpuStorage, CudaStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor,
    };
    use cudarc::cudnn::safe::{ConvForward, Cudnn};
    use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
    use std::cmp::Ordering;
    use std::sync::Arc;
    use std::time::Instant;

    const DEFAULT_WARMUP: usize = 8;
    const DEFAULT_ITERS: usize = 40;
    const DEFAULT_INNER: usize = 32;
    const DEFAULT_MAX_DRIFT_PCT: f64 = 5.0;
    const DEFAULT_FRONTIER_MARGIN_PCT: f64 = 10.0;

    const RAW_FN: &str = "asd_measure_dw5x5_conv2d_f32";
    const RAW_MODULE: &str = "asd_measure_conv2d_dw5x5_v0";

    // Header-free on purpose: NVRTC does not inherit the host compiler's
    // system include search path. The three dimensions use an explicit 64-bit ABI.
    const RAW_CUDA: &str = r#"
extern "C" __global__ void asd_measure_dw5x5_conv2d_f32(
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
"#;

    #[derive(Clone, Copy, Debug)]
    struct ExactCase {
        c: usize,
        h: usize,
        w: usize,
    }

    const CASES: [ExactCase; 4] = [
        ExactCase {
            c: 48,
            h: 64,
            w: 48,
        },
        ExactCase {
            c: 96,
            h: 32,
            w: 24,
        },
        ExactCase {
            c: 192,
            h: 16,
            w: 12,
        },
        ExactCase { c: 384, h: 8, w: 6 },
    ];

    impl ExactCase {
        fn signature(self) -> String {
            format!(
                "conv2d:f32:b1:c{}:h{}:w{}:weight{}x1x5x5:g{}:k5:s1:p2:d1:contiguous_zero_offset",
                self.c, self.h, self.w, self.c, self.c
            )
        }

        fn output_el(self) -> usize {
            self.c * self.h * self.w
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Backend {
        RawDw5x5,
        Cudnn,
    }

    impl Backend {
        fn as_str(self) -> &'static str {
            match self {
                Self::RawDw5x5 => "raw_dw5x5",
                Self::Cudnn => "cudnn",
            }
        }
    }

    #[derive(Clone, Copy)]
    struct TimingStats {
        median_us: f64,
        p10_us: f64,
        p90_us: f64,
    }

    struct RawDw5x5 {
        ptx: Arc<String>,
    }

    struct CudnnDw5x5 {
        cudnn: Arc<Cudnn>,
    }

    fn validate_exact_domain(
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<ExactCase> {
        if input.device().id() != kernel.device().id() {
            candle_core::bail!("ASD DW5x5 measurement requires input and weight on one CUDA device")
        }
        if input.dtype() != DType::F32 || kernel.dtype() != DType::F32 {
            candle_core::bail!("ASD DW5x5 measurement is exact-domain f32 only")
        }
        if !input_l.is_contiguous()
            || !kernel_l.is_contiguous()
            || input_l.start_offset() != 0
            || kernel_l.start_offset() != 0
        {
            candle_core::bail!(
                "ASD DW5x5 measurement requires contiguous_zero_offset input and weight"
            )
        }
        let dims = input_l.dims();
        let kdims = kernel_l.dims();
        if dims.len() != 4 || kdims.len() != 4 {
            candle_core::bail!(
                "ASD DW5x5 measurement expects NCHW input and OIHW weight, got {:?} / {:?}",
                dims,
                kdims
            )
        }
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        if b != 1 || kdims[0] != c || kdims[1] != 1 || kdims[2] != 5 || kdims[3] != 5 {
            candle_core::bail!(
                "ASD DW5x5 exact-domain mismatch input={:?} weight={:?}",
                dims,
                kdims
            )
        }
        CASES
            .iter()
            .copied()
            .find(|case| case.c == c && case.h == h && case.w == w)
            .ok_or_else(|| {
                candle_core::Error::Msg(
                    format!(
                        "ASD DW5x5 signature is outside the four measurement cases: c={c} h={h} w={w}"
                    )
                    .into(),
                )
            })
    }

    impl CustomOp2 for RawDw5x5 {
        fn name(&self) -> &'static str {
            "asd-measure-conv2d-dw5x5-raw"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_l: &Layout,
            _kernel: &CpuStorage,
            _kernel_l: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("ASD DW5x5 raw measurement is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_l: &Layout,
            kernel: &CudaStorage,
            kernel_l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            let case = validate_exact_domain(input, input_l, kernel, kernel_l)?;
            let dev = input.device().clone();
            let src = input.as_cuda_slice::<f32>()?;
            let weight = kernel.as_cuda_slice::<f32>()?;
            let out = unsafe { dev.alloc::<f32>(case.output_el())? };
            let func = dev.get_or_load_custom_func(RAW_FN, RAW_MODULE, self.ptx.as_str())?;
            let cfg = LaunchConfig {
                grid_dim: (
                    case.w.div_ceil(16) as u32,
                    case.h.div_ceil(8) as u32,
                    case.c as u32,
                ),
                block_dim: (16, 8, 1),
                shared_mem_bytes: 0,
            };
            let channels = case.c as u64;
            let height = case.h as u64;
            let width = case.w as u64;
            let mut builder = func.builder();
            builder.arg(&channels);
            builder.arg(&height);
            builder.arg(&width);
            builder.arg(src);
            builder.arg(weight);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.w()?;
            Ok((
                CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from((1, case.c, case.h, case.w)),
            ))
        }
    }

    impl CustomOp2 for CudnnDw5x5 {
        fn name(&self) -> &'static str {
            "asd-measure-conv2d-dw5x5-cudnn"
        }

        fn cpu_fwd(
            &self,
            _input: &CpuStorage,
            _input_l: &Layout,
            _kernel: &CpuStorage,
            _kernel_l: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("ASD DW5x5 cuDNN measurement is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            input: &CudaStorage,
            input_l: &Layout,
            kernel: &CudaStorage,
            kernel_l: &Layout,
        ) -> Result<(CudaStorage, Shape)> {
            let case = validate_exact_domain(input, input_l, kernel, kernel_l)?;
            let dev = input.device().clone();
            let src = input.as_cuda_slice::<f32>()?;
            let weight = kernel.as_cuda_slice::<f32>()?;
            let src = src.slice(0..);
            let weight = weight.slice(0..);
            let mut out = unsafe { dev.alloc::<f32>(case.output_el())? };

            let mut conv = self.cudnn.create_conv2d::<f32>(
                [2, 2],
                [1, 1],
                [1, 1],
                cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )?;
            conv.set_group_count(case.c as i32)?;
            let x = self.cudnn.create_4d_tensor::<f32>(
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [1, case.c as i32, case.h as i32, case.w as i32],
            )?;
            let w = self.cudnn.create_4d_filter::<f32>(
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [case.c as i32, 1, 5, 5],
            )?;
            let y = self.cudnn.create_4d_tensor::<f32>(
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                [1, case.c as i32, case.h as i32, case.w as i32],
            )?;
            let forward = ConvForward {
                conv: &conv,
                x: &x,
                w: &w,
                y: &y,
            };
            let algo = forward.pick_algorithm()?;
            let workspace_size = forward.get_workspace_size(algo)?;
            let mut workspace = dev.cuda_stream().alloc_zeros::<u8>(workspace_size)?;
            unsafe {
                forward.launch::<CudaSlice<u8>, _, _, _>(
                    algo,
                    Some(&mut workspace),
                    (1.0f32, 0.0f32),
                    &src,
                    &weight,
                    &mut out,
                )?;
            }
            Ok((
                CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from((1, case.c, case.h, case.w)),
            ))
        }
    }

    fn parse_count(flag: &str, default: usize) -> usize {
        let args = std::env::args().collect::<Vec<_>>();
        args.windows(2)
            .find_map(|pair| {
                (pair[0] == flag)
                    .then(|| pair[1].parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(default)
    }

    fn parse_f64(flag: &str, default: f64) -> f64 {
        let args = std::env::args().collect::<Vec<_>>();
        args.windows(2)
            .find_map(|pair| {
                (pair[0] == flag)
                    .then(|| pair[1].parse::<f64>().ok())
                    .flatten()
            })
            .unwrap_or(default)
    }

    fn parse_backend_filter() -> Result<Option<Backend>> {
        let args = std::env::args().collect::<Vec<_>>();
        let value = args
            .windows(2)
            .find_map(|pair| (pair[0] == "--backend").then(|| pair[1].as_str()));
        match value {
            None | Some("frontier") => Ok(None),
            Some("raw_dw5x5") => Ok(Some(Backend::RawDw5x5)),
            Some("cudnn") => Ok(Some(Backend::Cudnn)),
            Some(other) => candle_core::bail!(
                "unknown --backend {other:?}, expected raw_dw5x5, cudnn, or frontier"
            ),
        }
    }

    fn deterministic(len: usize, mul: usize, bias: isize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * mul) % 127) as isize + bias) as f32 / 128.0)
            .collect()
    }

    fn tensors(case: ExactCase, device: &Device) -> Result<(Tensor, Tensor)> {
        let input = Tensor::from_vec(
            deterministic(case.output_el(), 37, -63),
            (1, case.c, case.h, case.w),
            device,
        )?;
        let weight = Tensor::from_vec(
            deterministic(case.c * 25, 53, -63),
            (case.c, 1, 5, 5),
            device,
        )?;
        Ok((input, weight))
    }

    fn call(
        backend: Backend,
        input: &Tensor,
        weight: &Tensor,
        raw: &RawDw5x5,
        cudnn: &CudnnDw5x5,
    ) -> Result<Tensor> {
        match backend {
            Backend::RawDw5x5 => input.apply_op2_no_bwd(weight, raw),
            Backend::Cudnn => input.apply_op2_no_bwd(weight, cudnn),
        }
    }

    fn max_abs_rel(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
        if lhs.dims() != rhs.dims() {
            candle_core::bail!(
                "shape mismatch in ASD DW5x5 measurement: {:?} vs {:?}",
                lhs.dims(),
                rhs.dims()
            )
        }
        let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
        let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (&a, &b) in lhs.iter().zip(&rhs) {
            let abs = (a - b).abs();
            let rel = abs / b.abs().max(1e-6);
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        Ok((max_abs, max_rel))
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
        sorted[idx]
    }

    fn batched_launches(
        backend: Backend,
        input: &Tensor,
        weight: &Tensor,
        raw: &RawDw5x5,
        cudnn: &CudnnDw5x5,
        inner: usize,
    ) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::with_capacity(inner);
        for _ in 0..inner {
            outputs.push(call(backend, input, weight, raw, cudnn)?);
        }
        Ok(outputs)
    }

    fn measure(
        backend: Backend,
        input: &Tensor,
        weight: &Tensor,
        raw: &RawDw5x5,
        cudnn: &CudnnDw5x5,
        device: &Device,
        warmup: usize,
        iters: usize,
        inner: usize,
    ) -> Result<TimingStats> {
        for _ in 0..warmup {
            let outputs = batched_launches(backend, input, weight, raw, cudnn, inner)?;
            device.synchronize()?;
            std::hint::black_box(outputs);
        }
        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            device.synchronize()?;
            let start = Instant::now();
            let outputs = batched_launches(backend, input, weight, raw, cudnn, inner)?;
            device.synchronize()?;
            samples.push(start.elapsed().as_secs_f64() * 1_000_000.0 / inner as f64);
            std::hint::black_box(outputs);
        }
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        Ok(TimingStats {
            median_us: percentile(&samples, 0.50),
            p10_us: percentile(&samples, 0.10),
            p90_us: percentile(&samples, 0.90),
        })
    }

    fn median3(a: f64, b: f64, c: f64) -> f64 {
        let mut values = [a, b, c];
        values.sort_unstable_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
        values[1]
    }

    fn relative_drift_pct(pre: f64, post: f64) -> f64 {
        let center = (pre + post) * 0.5;
        if center == 0.0 {
            0.0
        } else {
            (post - pre).abs() / center * 100.0
        }
    }

    fn print_phase(name: &str, backend: Backend, stats: TimingStats) {
        println!(
            "PHASE phase={} backend={} median_us={:.6} p10_us={:.6} p90_us={:.6}",
            name,
            backend.as_str(),
            stats.median_us,
            stats.p10_us,
            stats.p90_us
        );
    }

    pub fn main() -> Result<()> {
        let warmup = parse_count("--warmup", DEFAULT_WARMUP);
        let iters = parse_count("--iters", DEFAULT_ITERS);
        let inner = parse_count("--inner", DEFAULT_INNER);
        let max_drift_pct = parse_f64("--max-drift-pct", DEFAULT_MAX_DRIFT_PCT);
        let frontier_margin_pct = parse_f64("--frontier-margin-pct", DEFAULT_FRONTIER_MARGIN_PCT);
        let backend_filter = parse_backend_filter()?;
        if warmup == 0 || iters == 0 || inner == 0 {
            candle_core::bail!("--warmup, --iters and --inner must be greater than zero")
        }
        if max_drift_pct < 0.0 || frontier_margin_pct < 0.0 {
            candle_core::bail!("measurement thresholds must be non-negative")
        }

        let device = Device::new_cuda(0)?;
        let cuda_dev = match &device {
            Device::Cuda(dev) => dev.clone(),
            _ => candle_core::bail!("ASD DW5x5 frontier requires CUDA"),
        };
        let ptx = cudarc::nvrtc::compile_ptx(RAW_CUDA).map_err(|err| {
            candle_core::Error::Msg(
                format!("NVRTC failed for ASD DW5x5 measurement kernel: {err}").into(),
            )
        })?;
        let raw = RawDw5x5 {
            ptx: Arc::new(ptx.to_src()),
        };
        let cudnn = CudnnDw5x5 {
            cudnn: Cudnn::new(cuda_dev.cuda_stream())?,
        };

        println!("=== ASD CONV2D DW5X5 EXACT FRONTIER V0 ===");
        println!("scope=measurement_only");
        println!("production_dispatch_modified=false");
        println!("asd_policy_modified=false");
        println!("production_activation=false");
        println!("candidate_state=measured_only");
        println!("nvrtc_source=header_free");
        println!("device={:?}", device.location());
        println!(
            "cuda_build_compute_cap={}",
            candle_core::cuda_backend::kernels::CUDA_BUILD_COMPUTE_CAP
        );
        println!(
            "backend_filter={}",
            backend_filter.map(Backend::as_str).unwrap_or("frontier")
        );
        println!("exact_dtype=f32");
        println!("exact_layout=contiguous_zero_offset");
        println!("exact_op=conv2d_depthwise");
        println!("exact_kernel=5x5");
        println!("exact_stride=1");
        println!("exact_padding=2");
        println!("exact_dilation=1");
        println!("warmup_samples={warmup}");
        println!("timed_samples={iters}");
        println!("launches_per_sample={inner}");
        println!("max_drift_pct={max_drift_pct:.3}");
        println!("frontier_margin_pct={frontier_margin_pct:.3}");
        println!("measurement=batched_api_wall_time_one_sync_per_sample");
        println!("sequence=raw_a,cudnn_b,raw_a2,cudnn_a,raw_b,cudnn_a2");

        if let Some(backend) = backend_filter {
            for case in CASES {
                let (input, weight) = tensors(case, &device)?;
                println!("\nCASE signature={}", case.signature());
                let stats = measure(
                    backend, &input, &weight, &raw, &cudnn, &device, warmup, iters, inner,
                )?;
                print_phase("single", backend, stats);
            }
            println!("\nSTATUS=MEASURED_SINGLE_BACKEND");
            return Ok(());
        }

        let mut all_parity = true;
        let mut all_stable = true;
        for case in CASES {
            let (input, weight) = tensors(case, &device)?;
            let signature = case.signature();
            println!("\nCASE signature={signature}");

            let raw_probe = call(Backend::RawDw5x5, &input, &weight, &raw, &cudnn)?;
            let cudnn_probe = call(Backend::Cudnn, &input, &weight, &raw, &cudnn)?;
            device.synchronize()?;
            let (max_abs, max_rel) = max_abs_rel(&raw_probe, &cudnn_probe)?;
            let parity = max_abs <= 1e-4 || max_rel <= 1e-4;
            all_parity &= parity;
            println!(
                "PARITY raw_dw5x5_vs_cudnn max_abs={:.8} max_rel={:.8} pass={}",
                max_abs, max_rel, parity
            );

            let raw_a = measure(
                Backend::RawDw5x5,
                &input,
                &weight,
                &raw,
                &cudnn,
                &device,
                warmup,
                iters,
                inner,
            )?;
            let cudnn_b = measure(
                Backend::Cudnn,
                &input,
                &weight,
                &raw,
                &cudnn,
                &device,
                warmup,
                iters,
                inner,
            )?;
            let raw_a2 = measure(
                Backend::RawDw5x5,
                &input,
                &weight,
                &raw,
                &cudnn,
                &device,
                warmup,
                iters,
                inner,
            )?;
            let cudnn_a = measure(
                Backend::Cudnn,
                &input,
                &weight,
                &raw,
                &cudnn,
                &device,
                warmup,
                iters,
                inner,
            )?;
            let raw_b = measure(
                Backend::RawDw5x5,
                &input,
                &weight,
                &raw,
                &cudnn,
                &device,
                warmup,
                iters,
                inner,
            )?;
            let cudnn_a2 = measure(
                Backend::Cudnn,
                &input,
                &weight,
                &raw,
                &cudnn,
                &device,
                warmup,
                iters,
                inner,
            )?;

            print_phase("raw_a", Backend::RawDw5x5, raw_a);
            print_phase("cudnn_b", Backend::Cudnn, cudnn_b);
            print_phase("raw_a2", Backend::RawDw5x5, raw_a2);
            print_phase("cudnn_a", Backend::Cudnn, cudnn_a);
            print_phase("raw_b", Backend::RawDw5x5, raw_b);
            print_phase("cudnn_a2", Backend::Cudnn, cudnn_a2);

            let raw_us = median3(raw_a.median_us, raw_a2.median_us, raw_b.median_us);
            let cudnn_us = median3(cudnn_b.median_us, cudnn_a.median_us, cudnn_a2.median_us);
            let raw_p90 = median3(raw_a.p90_us, raw_a2.p90_us, raw_b.p90_us);
            let cudnn_p90 = median3(cudnn_b.p90_us, cudnn_a.p90_us, cudnn_a2.p90_us);
            let raw_drift_pct = relative_drift_pct(raw_a.median_us, raw_b.median_us);
            let cudnn_drift_pct = relative_drift_pct(cudnn_b.median_us, cudnn_a2.median_us);
            let stable = raw_drift_pct <= max_drift_pct && cudnn_drift_pct <= max_drift_pct;
            all_stable &= stable;

            let raw_vs_cudnn_speedup_x = cudnn_us / raw_us;
            let raw_latency_reduction_vs_cudnn_pct = (1.0 - raw_us / cudnn_us) * 100.0;
            let direction = [
                raw_a.median_us < cudnn_b.median_us,
                raw_a2.median_us < cudnn_a.median_us,
                raw_b.median_us < cudnn_a2.median_us,
            ];
            let direction_consistent = direction.iter().all(|&value| value == direction[0]);
            let margin_x = 1.0 + frontier_margin_pct / 100.0;
            let winner = if !parity || !stable || !direction_consistent {
                "hold"
            } else if raw_vs_cudnn_speedup_x >= margin_x {
                "raw_dw5x5"
            } else if raw_vs_cudnn_speedup_x <= 1.0 / margin_x {
                "cudnn"
            } else {
                "hold"
            };

            println!(
                "FRONTIER_RESULT signature={} raw_us={:.6} cudnn_us={:.6} raw_vs_cudnn_speedup_x={:.6} raw_latency_reduction_vs_cudnn_pct={:.3} raw_drift_pct={:.3} cudnn_drift_pct={:.3} raw_p90_us={:.6} cudnn_p90_us={:.6} direction_consistent={} parity={} stable={} winner={} state=measured_only",
                signature,
                raw_us,
                cudnn_us,
                raw_vs_cudnn_speedup_x,
                raw_latency_reduction_vs_cudnn_pct,
                raw_drift_pct,
                cudnn_drift_pct,
                raw_p90,
                cudnn_p90,
                direction_consistent,
                parity,
                stable,
                winner
            );
        }

        println!(
            "\nMEASUREMENT_GATE parity={} drift={} production_activation=false",
            all_parity, all_stable
        );
        println!(
            "STATUS={}",
            if all_parity && all_stable {
                "MEASURED"
            } else {
                "HOLD"
            }
        );
        Ok(())
    }
}

#[cfg(all(feature = "cuda", feature = "cudnn"))]
fn main() -> candle_core::Result<()> {
    enabled::main()
}

#[cfg(not(all(feature = "cuda", feature = "cudnn")))]
fn main() {
    eprintln!("asd_conv2d_dw5x5_frontier requires --features cuda,cudnn");
}
