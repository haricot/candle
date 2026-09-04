use crate::{CpuStorage, CudaStorage, CustomOp2, Layout, MetalStorage, Result, Shape, Tensor};

use super::grouped::{GroupedConvTranspose1D, GroupedConvTranspose2D};
use super::{ParamsConvTranspose1D, ParamsConvTranspose2D};

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
            Err(_) => GroupedConvTranspose1D(self.0.clone()).cpu_fwd(
                input, input_l, kernel, kernel_l,
            ),
        }
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let force_kernel =
            std::env::var_os("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL").is_some();

        #[cfg(feature = "cudnn")]
        if !force_kernel && kernel_l.is_contiguous() {
            match super::grouped_transpose_cudnn::launch_grouped_conv_transpose1d(
                input, input_l, kernel, kernel_l, &self.0,
            ) {
                Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
                Err(err)
                    if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
                {
                    return Err(err)
                }
                Err(_) => {}
            }
        }

        #[cfg(feature = "cuda")]
        match super::grouped_transpose_cuda::launch1d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
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
            Err(_) => GroupedConvTranspose2D(self.0.clone()).cpu_fwd(
                input, input_l, kernel, kernel_l,
            ),
        }
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_l: &Layout,
        kernel: &CudaStorage,
        kernel_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let force_kernel =
            std::env::var_os("CANDLE_CUDA_GROUPED_TRANSPOSE_FORCE_KERNEL").is_some();

        #[cfg(feature = "cudnn")]
        if !force_kernel && kernel_l.is_contiguous() {
            match super::grouped_transpose_cudnn::launch_grouped_conv_transpose2d(
                input, input_l, kernel, kernel_l, &self.0,
            ) {
                Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
                Err(err)
                    if std::env::var_os("CANDLE_CUDNN_NATIVE_GROUPED_TRANSPOSE_STRICT").is_some() =>
                {
                    return Err(err)
                }
                Err(_) => {}
            }
        }

        #[cfg(feature = "cuda")]
        match super::grouped_transpose_cuda::launch2d(input, input_l, kernel, kernel_l, &self.0) {
            Ok(out) => return Ok((out, Shape::from(self.0.out_dims()))),
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
