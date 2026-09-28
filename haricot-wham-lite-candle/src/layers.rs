use candle_core::{D, IndexOp, Result, Tensor};
use candle_nn::rnn::{lstm, LSTMConfig, LSTMState, RNN};
use candle_nn::{linear, Linear, Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct StackedLstmState {
    pub layers: Vec<LSTMState>,
}

#[derive(Clone, Debug)]
pub struct StackedLstm {
    layers: Vec<candle_nn::rnn::LSTM>,
}

impl StackedLstm {
    pub fn load(
        input_dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            let layer_input = if layer_idx == 0 {
                input_dim
            } else {
                hidden_dim
            };
            let cfg = LSTMConfig {
                layer_idx,
                ..Default::default()
            };
            layers.push(lstm(layer_input, hidden_dim, cfg, vb.clone())?);
        }
        Ok(Self { layers })
    }

    pub fn zero_state(&self, batch: usize) -> Result<StackedLstmState> {
        let mut states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            states.push(layer.zero_state(batch)?);
        }
        Ok(StackedLstmState { layers: states })
    }

    pub fn step(
        &self,
        input: &Tensor,
        state: &StackedLstmState,
    ) -> Result<(Tensor, StackedLstmState)> {
        let mut x = input.clone();
        let mut next = Vec::with_capacity(self.layers.len());
        for (layer, layer_state) in self.layers.iter().zip(state.layers.iter()) {
            let s = layer.step(&x, layer_state)?;
            x = s.h.clone();
            next.push(s);
        }
        Ok((x, StackedLstmState { layers: next }))
    }
}

#[derive(Clone, Debug)]
pub struct NeuralInitialization {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
    hidden_dim: usize,
    n_layers: usize,
}

impl NeuralInitialization {
    pub fn load(
        in_dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear1: linear(in_dim, hidden_dim, vb.pp("linear1"))?,
            linear2: linear(hidden_dim, hidden_dim * n_layers, vb.pp("linear2"))?,
            linear3: linear(
                hidden_dim * n_layers,
                hidden_dim * 2 * n_layers,
                vb.pp("linear3"),
            )?,
            hidden_dim,
            n_layers,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<StackedLstmState> {
        let b = x.dim(0)?;
        let out = self.linear1.forward(x)?.relu()?;
        let out = self.linear2.forward(&out)?.relu()?;
        let out = self.linear3.forward(&out)?;
        let out = out.reshape((b, 2, self.n_layers, self.hidden_dim))?;

        let mut layers = Vec::with_capacity(self.n_layers);
        for i in 0..self.n_layers {
            let h = out.i((.., 0, i, ..))?.contiguous()?;
            let c = out.i((.., 1, i, ..))?.contiguous()?;
            layers.push(LSTMState::new(h, c));
        }
        Ok(StackedLstmState { layers })
    }
}

#[derive(Clone, Debug)]
pub struct Regressor {
    rnn: StackedLstm,
    heads: Vec<Linear>,
}

impl Regressor {
    pub fn load(
        in_dim: usize,
        hidden_dim: usize,
        out_dims: &[usize],
        init_dim: usize,
        n_layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rnn = StackedLstm::load(in_dim + init_dim, hidden_dim, n_layers, vb.pp("rnn"))?;
        let mut heads = Vec::with_capacity(out_dims.len());
        for (i, &out_dim) in out_dims.iter().enumerate() {
            heads.push(linear(
                hidden_dim,
                out_dim,
                vb.pp(format!("declayer{i}")),
            )?);
        }
        Ok(Self { rnn, heads })
    }

    pub fn zero_state(&self, batch: usize) -> Result<StackedLstmState> {
        self.rnn.zero_state(batch)
    }

    pub fn step(
        &self,
        x: &Tensor,
        inits: &[&Tensor],
        state: &StackedLstmState,
    ) -> Result<(Vec<Tensor>, Tensor, StackedLstmState)> {
        let mut cat_inputs: Vec<&Tensor> = Vec::with_capacity(1 + inits.len());
        cat_inputs.push(x);
        cat_inputs.extend_from_slice(inits);
        let xc = Tensor::cat(&cat_inputs, D::Minus1)?;
        let (hidden, next_state) = self.rnn.step(&xc, state)?;
        let mut outputs = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            outputs.push(head.forward(&hidden)?);
        }
        Ok((outputs, hidden, next_state))
    }
}
