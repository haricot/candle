#[cfg(feature = "cuda")]
use crate::backend::BackendStorage;
use crate::{CpuStorage, CudaStorage, CustomOp2, Layout, MetalStorage, Result, Shape, Tensor};

use super::grouped::{GroupedConvTranspose1D, GroupedConvTranspose2D};
use super::{ParamsConvTranspose1D, ParamsConvTranspose2D};

fn env_truthy(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

#[cfg(feature = "cuda")]
fn dispatch_fallback_context(
    decision: &super::grouped_transpose_dispatch::GroupedTransposeDispatchDecision,
) -> Option<super::asd_exact_cuda::QualifiedFallbackContext> {
    if !decision.exact_requires_raw() {
        return None;
    }

    Some(super::asd_exact_cuda::QualifiedFallbackContext {
        decision_id: decision.exact_decision_id()?.to_owned(),
        decision_identity_sha256: decision.exact_decision_identity_sha256()?.to_owned(),
        profile_id: decision.exact_profile_id()?.to_owned(),
    })
}

#[cfg(all(feature = "cuda", feature = "cudnn"))]
fn try_qualified_ct1d_fallback(
    decision: &super::grouped_transpose_dispatch::GroupedTransposeDispatchDecision,
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    params: &ParamsConvTranspose1D,
) -> Result<Option<CudaStorage>> {
    if !kernel_l.is_contiguous() {
        return Ok(None);
    }

    let context = if let Some(context) = dispatch_fallback_context(decision) {
        Some(context)
    } else if super::grouped_transpose_dispatch::automatic_exact_eligible() {
        super::asd_exact_cuda::ct1d_qualified_fallback_context(
            input, input_l, kernel, kernel_l, params,
        )?
    } else {
        None
    };

    let Some(context) = context else {
        return Ok(None);
    };

    let decision_id = context.decision_id.as_str();
    let decision_identity_sha256 = context.decision_identity_sha256.as_str();
    let profile_id = context.profile_id.as_str();

    let runtime_cudnn_version = unsafe { cudarc::cudnn::sys::cudnnGetVersion() };
    let actual_gpu_uuid = super::asd_exact_cuda::actual_uuid(input).ok_or_else(|| {
        crate::Error::Msg("unable to authenticate CUDA UUID for ASD fallback lookup".into())
    })?;

    let fallbacks = crate::asd_fallback_store::qualified_fallbacks_for_decision(
        decision_id,
        decision_identity_sha256,
        profile_id,
        &actual_gpu_uuid,
    )?;

    for fallback in fallbacks {
        if fallback.provider != "cudnn"
            || fallback.implementation_id != "candle.cudnn.grouped-transpose.v1"
            || fallback
                .required_cudnn_version_raw
                .is_some_and(|required| required != runtime_cudnn_version)
        {
            continue;
        }

        let out = super::grouped_transpose_cudnn::launch_grouped_conv_transpose1d(
            input, input_l, kernel, kernel_l, params,
        )?;
        if env_truthy("CANDLE_GROUPED_TRANSPOSE_TRACE") || env_truthy("CANDLE_ASD_EXACT_TRACE") {
            eprintln!(
                "[candle asd-v3] runtime_role=fallback reason=primary_artifact_unavailable decision={} decision_identity_sha256={} fallback_rank={} provider={} implementation={} protocol={} cudnn_version_raw={}",
                decision_id,
                decision_identity_sha256,
                fallback.rank,
                fallback.provider,
                fallback.implementation_id,
                fallback.protocol,
                runtime_cudnn_version,
            );
        }
        return Ok(Some(out));
    }

    Ok(None)
}

#[cfg(all(feature = "cuda", feature = "cudnn"))]
fn try_qualified_ct2d_fallback(
    decision: &super::grouped_transpose_dispatch::GroupedTransposeDispatchDecision,
    input: &CudaStorage,
    input_l: &Layout,
    kernel: &CudaStorage,
    kernel_l: &Layout,
    params: &ParamsConvTranspose2D,
) -> Result<Option<CudaStorage>> {
    if !kernel_l.is_contiguous() {
        return Ok(None);
    }

    let context = if let Some(context) = dispatch_fallback_context(decision) {
        Some(context)
    } else if super::grouped_transpose_dispatch::automatic_exact_eligible() {
        super::asd_exact_cuda::ct2d_qualified_fallback_context(
            input, input_l, kernel, kernel_l, params,
        )?
    } else {
        None
    };

    let Some(context) = context else {
        return Ok(None);
    };

    let decision_id = context.decision_id.as_str();
    let decision_identity_sha256 = context.decision_identity_sha256.as_str();
    let profile_id = context.profile_id.as_str();

    let runtime_cudnn_version = unsafe { cudarc::cudnn::sys::cudnnGetVersion() };
    let actual_gpu_uuid = super::asd_exact_cuda::actual_uuid(input).ok_or_else(|| {
        crate::Error::Msg("unable to authenticate CUDA UUID for ASD fallback lookup".into())
    })?;

    let fallbacks = crate::asd_fallback_store::qualified_fallbacks_for_decision(
        decision_id,
        decision_identity_sha256,
        profile_id,
        &actual_gpu_uuid,
    )?;

    for fallback in fallbacks {
        if fallback.provider != "cudnn"
            || fallback.implementation_id != "candle.cudnn.grouped-transpose.v1"
            || fallback
                .required_cudnn_version_raw
                .is_some_and(|required| required != runtime_cudnn_version)
        {
            continue;
        }

        let out = super::grouped_transpose_cudnn::launch_grouped_conv_transpose2d(
            input, input_l, kernel, kernel_l, params,
        )?;
        if env_truthy("CANDLE_GROUPED_TRANSPOSE_TRACE") || env_truthy("CANDLE_ASD_EXACT_TRACE") {
            eprintln!(
                "[candle asd-v3] runtime_role=fallback reason=primary_artifact_unavailable decision={} decision_identity_sha256={} fallback_rank={} provider={} implementation={} protocol={} cudnn_version_raw={}",
                decision_id,
                decision_identity_sha256,
                fallback.rank,
                fallback.provider,
                fallback.implementation_id,
                fallback.protocol,
                runtime_cudnn_version,
            );
        }
        return Ok(Some(out));
    }

    Ok(None)
}

#[derive(Clone, Debug)]
pub(super) struct NativeGroupedConvTranspose1D(pub(super) ParamsConvTranspose1D);

impl CustomOp2 for NativeGroupedConvTranspose1D {
    fn name(&self) -> &'static str {
        "native-grouped-conv-transpose1d"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        kernel: &CpuStorage,
        kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        match super::grouped_transpose_cpu::launch1d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => Ok((out, Shape::from(self.0.out_dims()))),
            Err(err)
                if std::env::var_os("CANDLE_CPU_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
            {
                Err(err)
            }
            Err(_) => {
                GroupedConvTranspose1D(self.0.clone()).cpu_fwd(input, input_l, kernel, kernel_l)
            }
        }
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        // Strict cuDNN is an execution constraint, not merely a fallback policy.
        // Check it before exact ASD or a dispatcher-selected raw route.
        #[cfg(feature = "cuda")]
        let hot_policy = super::grouped_transpose_dispatch::hot_policy();
        #[cfg(feature = "cuda")]
        let require_cudnn = hot_policy.explicit_cudnn_required()?;
        #[cfg(not(feature = "cuda"))]
        let require_cudnn = std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT")
            .is_some()
            || std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH")
                .ok()
                .as_deref()
                == Some("cudnn");
        if require_cudnn {
            #[cfg(feature = "cudnn")]
            {
                if !kernel_l.is_contiguous() {
                    crate::bail!("strict grouped ConvTranspose1D cuDNN requires contiguous kernel")
                }
                let out = super::grouped_transpose_cudnn::launch_grouped_conv_transpose1d(
                    input, input_l, kernel, kernel_l, &self.0,
                )?;
                if env_truthy("CANDLE_GROUPED_TRANSPOSE_TRACE")
                    || env_truthy("CANDLE_ASD_EXACT_TRACE")
                {
                    eprintln!(
                        "CANDLE_GROUPED_TRANSPOSE_BACKEND=cudnn strict={} dim=1d",
                        std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some()
                            as u8
                    );
                }
                return Ok((out, Shape::from(self.0.out_dims())));
            }
            #[cfg(not(feature = "cudnn"))]
            crate::bail!("strict grouped ConvTranspose1D requires cuDNN feature")
        }

        // An exact candidate is eligible only in auto mode, never above an
        // explicitly selected backend or a global ASD disable.
        #[cfg(feature = "cuda")]
        if hot_policy.automatic_exact_eligible() {
            if let Some(out) =
                super::asd_exact_cuda::try_launch_ct1d(input, input_l, kernel, kernel_l, &self.0)?
            {
                return Ok((out, Shape::from(self.0.out_dims())));
            }
        }

        #[cfg(feature = "cuda")]
        let decision = super::grouped_transpose_dispatch::decision_1d(
            input,
            &self.0,
            input_l,
            kernel_l,
            input.dtype(),
        )?;

        #[cfg(feature = "cuda")]
        if env_truthy("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED") && !decision.is_exact_asd() {
            crate::bail!(
                "CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED requires an authenticated ASD Exact Profile decision; generic auto dispatch is forbidden"
            )
        }

        #[cfg(feature = "cuda")]
        if decision.exact_requires_cudnn() {
            #[cfg(feature = "cudnn")]
            {
                if !kernel_l.is_contiguous() {
                    crate::bail!("ASD Exact Profile selected cuDNN for grouped ConvTranspose1D but kernel is not contiguous")
                }
                let out = super::grouped_transpose_cudnn::launch_grouped_conv_transpose1d(
                    input, input_l, kernel, kernel_l, &self.0,
                )?;
                decision.trace_submission("cudnn");
                return Ok((out, Shape::from(self.0.out_dims())));
            }
            #[cfg(not(feature = "cudnn"))]
            crate::bail!("ASD Exact Profile selected cuDNN for grouped ConvTranspose1D but candle-core was built without the cudnn feature")
        }

        // Phase A: if the promoted raw implementation matched but its external
        // and builtin artifacts are unavailable, consult only evidence-qualified
        // provider fallbacks. Historical challenge state is never used directly.
        #[cfg(all(feature = "cuda", feature = "cudnn"))]
        if let Some(out) =
            try_qualified_ct1d_fallback(&decision, input, input_l, kernel, kernel_l, &self.0)?
        {
            return Ok((out, Shape::from(self.0.out_dims())));
        }

        #[cfg(feature = "cuda")]
        if decision.exact_requires_raw() && env_truthy("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED") {
            crate::bail!(
                "ASD exact raw primary is unavailable and no evidence-qualified provider fallback could execute decision {}",
                decision.exact_decision_id().unwrap_or("unknown")
            )
        }

        #[cfg(feature = "cudnn")]
        if !decision.prefers_raw() && kernel_l.is_contiguous() {
            match super::grouped_transpose_cudnn::launch_grouped_conv_transpose1d(
                input, input_l, kernel, kernel_l, &self.0,
            ) {
                Ok(out) => {
                    decision.trace_submission("cudnn");
                    return Ok((out, Shape::from(self.0.out_dims())));
                }
                Err(err) if decision.is_exact_asd() => return Err(err),
                Err(err)
                    if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT")
                        .is_some() =>
                {
                    return Err(err)
                }
                Err(_) => {}
            }
        }

        #[cfg(feature = "cuda")]
        match super::grouped_transpose_cuda::launch1d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => {
                #[cfg(feature = "cuda")]
                decision.trace_submission("raw");
                return Ok((out, Shape::from(self.0.out_dims())));
            }
            Err(err)
                if std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH")
                    .ok()
                    .as_deref()
                    == Some("raw") =>
            {
                return Err(err)
            }
            #[cfg(feature = "cuda")]
            Err(err) if decision.is_exact_asd() => return Err(err),
            Err(err)
                if std::env::var_os("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
            {
                return Err(err)
            }
            Err(_) => {}
        }

        GroupedConvTranspose1D(self.0.clone()).cuda_fwd(input, input_l, kernel, kernel_l)
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_l: &Layout,
        kernel: &MetalStorage,
        kernel_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        #[cfg(feature = "metal")]
        match super::grouped_transpose_metal::launch1d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
            Err(err)
                if std::env::var_os("CANDLE_METAL_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
            {
                return Err(err)
            }
            Err(_) => {}
        }

        GroupedConvTranspose1D(self.0.clone()).metal_fwd(input, input_l, kernel, kernel_l)
    }

    fn bwd(
        &self,
        arg: &Tensor,
        kernel: &Tensor,
        res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        GroupedConvTranspose1D(self.0.clone()).bwd(arg, kernel, res, grad)
    }
}

#[derive(Clone, Debug)]
pub(super) struct NativeGroupedConvTranspose2D(pub(super) ParamsConvTranspose2D);

impl CustomOp2 for NativeGroupedConvTranspose2D {
    fn name(&self) -> &'static str {
        "native-grouped-conv-transpose2d"
    }

    fn cpu_fwd(
        &self,
        input: &CpuStorage,
        input_l: &Layout,
        kernel: &CpuStorage,
        kernel_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        match super::grouped_transpose_cpu::launch2d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => Ok((out, Shape::from(self.0.out_dims()))),
            Err(err)
                if std::env::var_os("CANDLE_CPU_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
            {
                Err(err)
            }
            Err(_) => {
                GroupedConvTranspose2D(self.0.clone()).cpu_fwd(input, input_l, kernel, kernel_l)
            }
        }
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        // Strict cuDNN is an execution constraint, not merely a fallback policy.
        // Check it before exact ASD or a dispatcher-selected raw route.
        #[cfg(feature = "cuda")]
        let hot_policy = super::grouped_transpose_dispatch::hot_policy();
        #[cfg(feature = "cuda")]
        let require_cudnn = hot_policy.explicit_cudnn_required()?;
        #[cfg(not(feature = "cuda"))]
        let require_cudnn = std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT")
            .is_some()
            || std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH")
                .ok()
                .as_deref()
                == Some("cudnn");
        if require_cudnn {
            #[cfg(feature = "cudnn")]
            {
                if !kernel_l.is_contiguous() {
                    crate::bail!("strict grouped ConvTranspose2D cuDNN requires contiguous kernel")
                }
                let out = super::grouped_transpose_cudnn::launch_grouped_conv_transpose2d(
                    input, input_l, kernel, kernel_l, &self.0,
                )?;
                if env_truthy("CANDLE_GROUPED_TRANSPOSE_TRACE")
                    || env_truthy("CANDLE_ASD_EXACT_TRACE")
                {
                    eprintln!(
                        "CANDLE_GROUPED_TRANSPOSE_BACKEND=cudnn strict={} dim=2d",
                        std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some()
                            as u8
                    );
                }
                return Ok((out, Shape::from(self.0.out_dims())));
            }
            #[cfg(not(feature = "cudnn"))]
            crate::bail!("strict grouped ConvTranspose2D requires cuDNN feature")
        }

        // An exact candidate is eligible only in auto mode, never above an
        // explicitly selected backend or a global ASD disable.
        #[cfg(feature = "cuda")]
        if hot_policy.automatic_exact_eligible() {
            if let Some(out) =
                super::asd_exact_cuda::try_launch_ct2d(input, input_l, kernel, kernel_l, &self.0)?
            {
                return Ok((out, Shape::from(self.0.out_dims())));
            }
        }

        #[cfg(feature = "cuda")]
        let decision = super::grouped_transpose_dispatch::decision_2d(
            input,
            &self.0,
            input_l,
            kernel_l,
            input.dtype(),
        )?;

        #[cfg(feature = "cuda")]
        if env_truthy("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED") && !decision.is_exact_asd() {
            crate::bail!(
                "CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED requires an authenticated ASD Exact Profile decision; generic auto dispatch is forbidden"
            )
        }

        #[cfg(feature = "cuda")]
        if decision.exact_requires_cudnn() {
            #[cfg(feature = "cudnn")]
            {
                if !kernel_l.is_contiguous() {
                    crate::bail!("ASD Exact Profile selected cuDNN for grouped ConvTranspose2D but kernel is not contiguous")
                }
                let out = super::grouped_transpose_cudnn::launch_grouped_conv_transpose2d(
                    input, input_l, kernel, kernel_l, &self.0,
                )?;
                decision.trace_submission("cudnn");
                return Ok((out, Shape::from(self.0.out_dims())));
            }
            #[cfg(not(feature = "cudnn"))]
            crate::bail!("ASD Exact Profile selected cuDNN for grouped ConvTranspose2D but candle-core was built without the cudnn feature")
        }

        // Phase A: if the promoted raw implementation matched but its external
        // and builtin artifacts are unavailable, consult only evidence-qualified
        // provider fallbacks.
        #[cfg(all(feature = "cuda", feature = "cudnn"))]
        if let Some(out) =
            try_qualified_ct2d_fallback(&decision, input, input_l, kernel, kernel_l, &self.0)?
        {
            return Ok((out, Shape::from(self.0.out_dims())));
        }

        #[cfg(feature = "cuda")]
        if decision.exact_requires_raw() && env_truthy("CANDLE_ASD_QUALIFIED_FALLBACK_REQUIRED") {
            crate::bail!(
                "ASD exact raw primary is unavailable and no evidence-qualified provider fallback could execute decision {}",
                decision.exact_decision_id().unwrap_or("unknown")
            )
        }

        #[cfg(feature = "cudnn")]
        if !decision.prefers_raw() && kernel_l.is_contiguous() {
            match super::grouped_transpose_cudnn::launch_grouped_conv_transpose2d(
                input, input_l, kernel, kernel_l, &self.0,
            ) {
                Ok(out) => {
                    decision.trace_submission("cudnn");
                    return Ok((out, Shape::from(self.0.out_dims())));
                }
                Err(err) if decision.is_exact_asd() => return Err(err),
                Err(err)
                    if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT")
                        .is_some() =>
                {
                    return Err(err)
                }
                Err(_) => {}
            }
        }

        #[cfg(feature = "cuda")]
        match super::grouped_transpose_cuda::launch2d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => {
                #[cfg(feature = "cuda")]
                decision.trace_submission("raw");
                return Ok((out, Shape::from(self.0.out_dims())));
            }
            Err(err)
                if std::env::var("CANDLE_GROUPED_TRANSPOSE_DISPATCH")
                    .ok()
                    .as_deref()
                    == Some("raw") =>
            {
                return Err(err)
            }
            #[cfg(feature = "cuda")]
            Err(err) if decision.is_exact_asd() => return Err(err),
            Err(err)
                if std::env::var_os("CANDLE_CUDA_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
            {
                return Err(err)
            }
            Err(_) => {}
        }

        GroupedConvTranspose2D(self.0.clone()).cuda_fwd(input, input_l, kernel, kernel_l)
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        input_l: &Layout,
        kernel: &MetalStorage,
        kernel_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        #[cfg(feature = "metal")]
        match super::grouped_transpose_metal::launch2d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
            Err(err)
                if std::env::var_os("CANDLE_METAL_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
            {
                return Err(err)
            }
            Err(_) => {}
        }

        GroupedConvTranspose2D(self.0.clone()).metal_fwd(input, input_l, kernel, kernel_l)
    }

    fn bwd(
        &self,
        arg: &Tensor,
        kernel: &Tensor,
        res: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        GroupedConvTranspose2D(self.0.clone()).bwd(arg, kernel, res, grad)
    }
}
