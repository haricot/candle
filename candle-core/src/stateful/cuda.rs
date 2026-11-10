use crate::asd_runtime_extensions::{RuntimeExactMatch, RuntimeExecutionProvider};
use crate::cuda_backend::{
    AsdCudaImplementation, CudaStorage, CudaStorageSlice, DeviceId, WrapErr,
};
use crate::{op::BackpropOp, DType, Result, Storage, Tensor};
use candle_kernels::asd_exact::{ExactOperation, ExactOperationCall};
use cudarc::driver::PushKernelArg;
use std::{cell::RefCell, sync::Arc};

struct Plan {
    selected: Arc<RuntimeExactMatch>,
    implementation: AsdCudaImplementation,
}

type PlanKey = (DeviceId, u64, usize, usize);
thread_local! {
    static LAST_PLAN: RefCell<Option<(PlanKey, Option<Arc<Plan>>)>> = const { RefCell::new(None) };
}

fn decision(selected: bool, reason: &str) {
    crate::diagnostics::event("asd.decision", || {
        crate::diagnostics::Payload::new()
            .boolean("selected", selected)
            .text("reason", reason)
    });
}

fn resolve(input: &CudaStorage, batch: usize, channels: usize) -> Result<Option<Arc<Plan>>> {
    let key = (
        input.device.id(),
        input.device.asd_runtime_generation(),
        batch,
        channels,
    );
    if let Some(plan) = LAST_PLAN.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|(k, _)| *k == key)
            .map(|(_, p)| p.clone())
    }) {
        decision(
            plan.is_some(),
            if plan.is_some() {
                "selected"
            } else {
                "profile_miss"
            },
        );
        return Ok(plan);
    }
    let call = ExactOperationCall {
        op: ExactOperation::CausalDepthwiseConv1dStep,
        dim: 1,
        batch,
        c_in: channels,
        c_out: channels,
        spatial0: 1,
        spatial1: 0,
        weight_rank: 3,
        weight0: channels,
        weight1: 1,
        weight2: 3,
        weight3: 0,
        groups: channels,
        kernel: 3,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
        dtype: "f16",
        input_contiguous: true,
        input_start_offset: 0,
        weight_contiguous: true,
        weight_start_offset: 0,
    };
    let uuid = crate::conv::asd_exact_cuda::actual_uuid(input);
    let selected = crate::asd_runtime_extensions::lookup_exact(call, uuid.as_deref())?
        .filter(|m| m.execution_provider == RuntimeExecutionProvider::RawCuda);
    let plan = match selected {
        Some(selected) => input
            .device
            .asd_modules()
            .resolve_stateful_f16(
                &selected.implementation_id,
                &selected.decision_identity_sha256,
            )?
            .map(|implementation| {
                Arc::new(Plan {
                    selected,
                    implementation,
                })
            }),
        None => None,
    };
    decision(
        plan.is_some(),
        if plan.is_some() {
            "selected"
        } else {
            "profile_miss"
        },
    );
    LAST_PLAN.with(|slot| *slot.borrow_mut() = Some((key, plan.clone())));
    Ok(plan)
}

pub(super) fn try_step(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let reason = if !x.device().is_cuda() {
        Some("unsupported_device")
    } else if x.dtype() != DType::F16 {
        Some("unsupported_dtype")
    } else if state.dim(2)? != 3 {
        Some("unsupported_kernel")
    } else if x.track_op() || state.track_op() || weight.track_op() {
        Some("autograd_requested")
    } else if [x, state, weight]
        .iter()
        .any(|t| !t.is_contiguous() || t.layout().start_offset() != 0)
    {
        Some("unsupported_layout")
    } else if crate::conv::grouped_transpose_dispatch::exact_disabled()
        || crate::conv::grouped_transpose_dispatch::specialized_cuda_disabled()
    {
        Some("asd_disabled")
    } else {
        None
    };
    if let Some(reason) = reason {
        decision(false, reason);
        return Ok(None);
    }
    let (batch, channels, _) = x.dims3()?;
    let rows = batch
        .checked_mul(channels)
        .ok_or_else(|| crate::Error::Msg("stateful element count overflow".into()))?;
    if rows == 0 {
        decision(false, "empty_input");
        return Ok(None);
    }
    let (xs, _) = x.storage_and_layout();
    let (ss, _) = state.storage_and_layout();
    let (ws, _) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ss), Storage::Cuda(ws)) = (&*xs, &*ss, &*ws) else {
        crate::bail!("stateful CUDA storage mismatch");
    };
    let Some(plan) = resolve(xs, batch, channels)? else {
        return Ok(None);
    };
    let cfg = plan.implementation.launch_config();
    if plan.implementation.output_count() != rows
        || cfg.block_dim.0 > 1024
        || u64::from(cfg.grid_dim.0) * u64::from(cfg.block_dim.0) < rows as u64
    {
        crate::bail!("stateful F16 manifest output count or launch coverage mismatch");
    }
    let batch =
        u32::try_from(batch).map_err(|_| crate::Error::Msg("stateful batch exceeds ABI".into()))?;
    let channels = u32::try_from(channels)
        .map_err(|_| crate::Error::Msg("stateful channels exceed ABI".into()))?;
    let (CudaStorageSlice::F16(x), CudaStorageSlice::F16(state), CudaStorageSlice::F16(weight)) =
        (&xs.slice, &ss.slice, &ws.slice)
    else {
        crate::bail!("stateful F16 storage dtype mismatch");
    };
    let mut y = unsafe { xs.device.alloc::<half::f16>(rows)? };
    let mut next = unsafe {
        xs.device.alloc::<half::f16>(
            rows.checked_mul(3)
                .ok_or_else(|| crate::Error::Msg("stateful allocation overflow".into()))?,
        )?
    };
    let mut builder = plan.implementation.function().builder();
    builder
        .arg(x)
        .arg(state)
        .arg(weight)
        .arg(&mut y)
        .arg(&mut next)
        .arg(&batch)
        .arg(&channels);
    if let Err(error) = unsafe { builder.launch(cfg) }.w() {
        crate::diagnostics::event("asd.submission_error", || {
            crate::diagnostics::Payload::new().text("error", &error)
        });
        return Err(error);
    }
    drop(builder);
    crate::diagnostics::event("asd.submitted", || {
        crate::diagnostics::Payload::new()
            .text("implementation", &plan.selected.implementation_id)
            .text(
                "artifact_sha256",
                plan.implementation
                    .artifact_sha256()
                    .unwrap_or("unavailable"),
            )
            .text("provider", plan.implementation.provider_name())
            .text("abi", super::SHORTCONV_F16_ABI)
            .text("state", &plan.selected.state)
            .text("completion", "unobserved")
    });
    let make = |slice, shape| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage {
                slice: CudaStorageSlice::F16(slice),
                device: xs.device.clone(),
            }),
            shape,
            BackpropOp::none(),
            false,
        )
    };
    Ok(Some((
        make(y, (batch as usize, channels as usize, 1)),
        make(next, (batch as usize, channels as usize, 3)),
    )))
}
