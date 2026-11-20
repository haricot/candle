//! R1.24 S5.2: native-only SM61 numeric qualification.
#![cfg(feature = "cuda-dkr-embedded")]
use candle_core::{quantized::{self,GgmlDType}, DType, Device, Module, Result, Tensor};

const CODECS: [GgmlDType;10] = [
    GgmlDType::Q4_0, GgmlDType::Q4_1, GgmlDType::Q5_0,
    GgmlDType::Q5_1, GgmlDType::Q8_0, GgmlDType::Q2K,
    GgmlDType::Q3K, GgmlDType::Q4K, GgmlDType::Q5K, GgmlDType::Q6K,
];

fn compare(label: &str, lhs: &Tensor, rhs: &Tensor, atol: f32, rtol: f32) -> Result<()> {
    assert_eq!(lhs.dims(), rhs.dims(), "{label}: dimensions");
    let a = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "{label}: lengths");
    for (i, (&got, &expected)) in a.iter().zip(&b).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(got.is_finite() && expected.is_finite()
            && (got - expected).abs() <= tolerance,
            "{label}[{i}]: got={got}, reference={expected}, tol={tolerance}");
    }
    Ok(())
}

fn native() -> Result<Device> {
    let gpu = Device::new_cuda(0)?;
    assert!(gpu.as_cuda_device()?.dkr_moe_enabled(), "native catalogue is required");
    Ok(gpu)
}

#[test]
#[ignore = "SM61 headless archive only; checks native module provenance externally"]
fn dkr_s52_ten_codecs_mmvq_mmq() -> Result<()> {
    let gpu = native()?;
    let (n,k) = (65usize,256usize); // row tail plus K-quant block
    let weight = (0..n*k).map(|i| 0.2f32 + ((i * 13 % 19) as f32)*0.025)
        .collect::<Vec<_>>();
    let w = Tensor::from_vec(weight,(n,k),&gpu)?;
    for codec in CODECS {
        let qt = quantized::QTensor::quantize(&w,codec)?;
        // Compare against the quantized *effective* weights, not unquantized
        // source weights, so the test isolates launch/math errors from
        // expected weight quantization losses.
        let dequant = qt.dequantize(&gpu)?;
        let reference_w_t = dequant.t()?.contiguous()?;
        let mm = quantized::QMatMul::from_qtensor(qt)?;
        for dtype in [DType::F16,DType::BF16,DType::F32] {
            for batch in [1usize,8,9,17] {
                let input = (0..batch*k).map(|i|
                    0.15f32 + ((i * 7 % 23) as f32)*0.01).collect::<Vec<_>>();
                let x = Tensor::from_vec(input,(batch,k),&gpu)?.to_dtype(dtype)?;
                let output = mm.forward(&x)?;
                assert_eq!(output.dims(), &[batch,n], "{codec:?} {dtype:?} {batch}");
                assert_eq!(output.dtype(), dtype);
                let expected = x.to_dtype(DType::F32)?.matmul(&reference_w_t)?;
                // Lower-bit GGUF and BF16 accumulation differ slightly
                // between fused Q8_1 paths and dequantize+cuBLAS.
                compare(&format!("{codec:?} {dtype:?} batch={batch}"),
                        &output,&expected,0.55,0.03)?;
                println!("DKR_R124_S52_QUANT_CASE codec={codec:?} dtype={dtype:?} batch={batch} route={}",
                         if batch<=8 {"MMVQ"} else {"MMQ"});
            }
        }
    }
    println!("DKR_R124_S52_QUANT=PASS cases=120");
    Ok(())
}

#[test]
#[ignore = "SM61 headless archive only; compare CPU and native GPU across tensor families"]
fn dkr_s52_core_families_cpu_parity() -> Result<()> {
    let gpu=native()?;
    let cpu=Device::Cpu;
    let a=(0..32usize).map(|i| (i as f32 - 14.)*0.0625).collect::<Vec<_>>();
    let b=(0..32usize).map(|i| ((i*7%23) as f32 - 11.)*0.03125).collect::<Vec<_>>();
    let ag=Tensor::from_vec(a.clone(),(4,8),&gpu)?;
    let ac=Tensor::from_vec(a,(4,8),&cpu)?;
    let bg=Tensor::from_vec(b.clone(),(4,8),&gpu)?;
    let bc=Tensor::from_vec(b,(4,8),&cpu)?;
    compare("affine",&ag.affine(1.25,-0.5)?,&ac.affine(1.25,-0.5)?,1e-6,1e-6)?;
    compare("binary",&(&ag+&bg)?,&(&ac+&bc)?,1e-6,1e-6)?;
    compare("unary",&ag.exp()?,&ac.exp()?,1e-5,1e-5)?;
    compare("cast",&ag.to_dtype(DType::F16)?.to_dtype(DType::F32)?,
                   &ac.to_dtype(DType::F16)?.to_dtype(DType::F32)?,0.,0.)?;
    compare("fill",&Tensor::ones((4,8),DType::F32,&gpu)?,
                   &Tensor::ones((4,8),DType::F32,&cpu)?,0.,0.)?;
    compare("reduce",&ag.sum(1)?,&ac.sum(1)?,1e-5,1e-5)?;
    let ids_g=Tensor::from_vec(vec![3u32,0,3],3,&gpu)?;
    let ids_c=Tensor::from_vec(vec![3u32,0,3],3,&cpu)?;
    compare("indexing",&ag.index_select(&ids_g,0)?,
                       &ac.index_select(&ids_c,0)?,0.,0.)?;
    compare("ternary",&ag.gt(&bg)?.where_cond(&ag,&bg)?,
                      &ac.gt(&bc)?.where_cond(&ac,&bc)?,0.,0.)?;
    let (sorted_g,indices_g)=ag.sort_last_dim(true)?;
    let (sorted_c,indices_c)=ac.sort_last_dim(true)?;
    compare("sort",&sorted_g,&sorted_c,0.,0.)?;
    assert_eq!(indices_g.to_vec2::<u32>()?,indices_c.to_vec2::<u32>()?);
    let x=(0..16).map(|i| (i as f32)*0.125-0.75).collect::<Vec<_>>();
    let kernel=vec![0.25f32,-0.5,0.75];
    let xg=Tensor::from_vec(x.clone(),(1,1,16),&gpu)?;
    let xc=Tensor::from_vec(x,(1,1,16),&cpu)?;
    let kg=Tensor::from_vec(kernel.clone(),(1,1,3),&gpu)?;
    let kc=Tensor::from_vec(kernel,(1,1,3),&cpu)?;
    compare("conv1d",&xg.conv1d(&kg,1,1,1,1)?,&xc.conv1d(&kc,1,1,1,1)?,1e-5,1e-5)?;
    println!("DKR_R124_S52_CORE=PASS families=affine,binary,unary,cast,fill,reduce,indexing,ternary,sort,conv");
    Ok(())
}
