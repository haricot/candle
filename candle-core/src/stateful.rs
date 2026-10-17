//! Stateful tensor operators with immutable input and next-state storage.
use crate::{Result, Tensor, D};

pub fn causal_depthwise_conv1d_step(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, c, t) = x.dims3()?;
    let (sb, sc, k) = state.dims3()?;
    if t != 1 || k == 0 || (b, c) != (sb, sc) || weight.dims() != [c, 1, k] {
        crate::bail!("causal depthwise step requires x[B,C,1], state[B,C,K], weight[C,1,K], K>0");
    }
    if x.dtype() != state.dtype()
        || x.dtype() != weight.dtype()
        || !x.device().same_device(state.device())
        || !x.device().same_device(weight.device())
    {
        crate::bail!("causal depthwise step requires one dtype and device");
    }
    let _scope = crate::diagnostics::scope("stateful.step", || {
        crate::diagnostics::Payload::new()
            .text("operation", "causal_depthwise_conv1d_step")
            .object("input", crate::diagnostics::tensor(x))
            .object("state", crate::diagnostics::tensor(state))
            .object("weight", crate::diagnostics::tensor(weight))
            .text("state_contract", "immutable_input_and_next_state")
            .number("kernel", k as u64)
    });
    crate::diagnostics::event("asd.decision", || {
        crate::diagnostics::Payload::new()
            .boolean("selected", false)
            .text("reason", "stateful_abi_not_implemented")
    });
    let next = if k > 1 {
        Tensor::cat(&[state.narrow(2, 1, k - 1)?, x.clone()], 2)?
    } else {
        x.copy()?
    };
    let w = weight.squeeze(1)?.unsqueeze(0)?;
    let output = next.broadcast_mul(&w)?.sum(D::Minus1)?.unsqueeze(2)?;
    crate::diagnostics::event("state.next", || {
        crate::diagnostics::Payload::new()
            .object("previous", crate::diagnostics::tensor(state))
            .object("next", crate::diagnostics::tensor(&next))
            .object("output", crate::diagnostics::tensor(&output))
            .text("mutation", "none")
    });
    Ok((output, next))
}
