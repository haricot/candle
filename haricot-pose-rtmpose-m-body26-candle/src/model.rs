use crate::cspnext::CspNextM;
use crate::rtmcc::RtmccHead26;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::profiling::LayerProfiler;

#[derive(Debug)]
pub struct RtmPoseOutput {
    pub simcc_x: Tensor,
    pub simcc_y: Tensor,
}

#[derive(Debug)]
pub struct RtmPoseBody26 {
    backbone: CspNextM,
    head: RtmccHead26,
}

impl RtmPoseBody26 {
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        Ok(Self { backbone: CspNextM::load(vb)?, head: RtmccHead26::load(vb)? })
    }

    /// Native Candle forward. Input shape: `[B,3,256,192]`.
    pub fn forward(&self, input: &Tensor) -> Result<RtmPoseOutput> {
        let dims = input.dims();
        if dims.len() != 4 || dims[1] != 3 || dims[2] != 256 || dims[3] != 192 {
            candle_core::bail!("RTMPose-m Body26 expects [B,3,256,192], got {:?}", dims);
        }
        let feature = self.backbone.forward(input)?;
        let (simcc_x, simcc_y) = self.head.forward(&feature)?;
        Ok(RtmPoseOutput { simcc_x, simcc_y })
    }

    /// Intrusive CUDA/CPU layer audit. The numerical graph is identical to
    /// `forward`, but each leaf operation is bounded by device synchronization
    /// through `LayerProfiler`. Use only for diagnostics.
    pub fn forward_profiled(
        &self,
        input: &Tensor,
        profiler: &mut LayerProfiler,
    ) -> Result<RtmPoseOutput> {
        let dims = input.dims();
        if dims.len() != 4 || dims[1] != 3 || dims[2] != 256 || dims[3] != 192 {
            candle_core::bail!("RTMPose-m Body26 expects [B,3,256,192], got {:?}", dims);
        }
        let feature = self.backbone.forward_profiled(input, profiler)?;
        let (simcc_x, simcc_y) = self.head.forward_profiled(&feature, profiler)?;
        Ok(RtmPoseOutput { simcc_x, simcc_y })
    }
}
