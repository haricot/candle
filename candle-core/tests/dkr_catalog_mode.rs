//! Explicit process-level qualification for the decoupled feature.
//! Invoke the ignored missing-provider test with all catalogue env vars unset
//! and without DKR_EMBED_NATIVE_DIR at build time.
#![cfg(all(feature = "cuda-dkr-catalog", not(feature = "cuda-dkr-s31")))]

use candle_core::Device;

#[cfg(not(feature = "cuda-dkr-embedded"))]
#[test]
#[ignore = "requires a build without embedded catalogue and no filesystem provider"]
fn dkr_catalog_missing_provider_is_error() {
    match Device::new_cuda(0) {
        Ok(_) => panic!("decoupled catalogue mode must not create a static-only device"),
        Err(e) => assert!(
            e.to_string().contains("cuda-dkr-catalog requires a verified native catalogue"),
            "unexpected CUDA device failure: {e}"
        ),
    }
}

#[test]
#[ignore = "requires selected verified native catalogue"]
fn dkr_catalog_selected_provider() -> candle_core::Result<()> {
    let gpu = Device::new_cuda(0)?;
    let dev = gpu.as_cuda_device()?;
    assert!(dev.dkr_moe_enabled(), "native catalogue not selected");
    Ok(())
}

#[test]
fn dkr_catalog_builtin_ptx_omitted() {
    use candle_core::cuda_backend::kernels;
    assert!(!kernels::BUILTIN_PTX_ENABLED,
        "catalogue-only test must not be unified with static legacy producer features");
    let modules = [kernels::AFFINE, kernels::BINARY, kernels::CAST, kernels::CONV,
        kernels::FILL, kernels::INDEXING, kernels::QUANTIZED, kernels::REDUCE,
        kernels::SORT, kernels::TERNARY, kernels::UNARY];
    assert_eq!(modules.len(), kernels::ALL_IDS.len());
    for (index, module) in modules.iter().enumerate() {
        assert_eq!(module.index(), index);
        assert!(module.static_ptx().is_none(), "static PTX remains at index {index}");
    }
    println!("DKR_R124_S42_BUILTIN_PTX=0");
}

#[test]
fn dkr_catalog_static_ffi_omitted() {
    use candle_core::cuda_backend::kernels;
    assert!(!kernels::STATIC_FFI_ENABLED,
        "static FFI producer unexpectedly enabled");
    assert!(!kernels::BUILTIN_PTX_ENABLED);
    println!("DKR_R124_S43_STATIC_FFI=0");
}

#[cfg(feature = "cuda-dkr-embedded")]
#[test]
#[ignore = "requires embedded deployment with runtime overrides unset"]
fn dkr_embedded_catalog_device() -> candle_core::Result<()> {
    assert!(std::env::var_os("CANDLE_DKR_S31_DIR").is_none());
    let started = std::time::Instant::now();
    let gpu = Device::new_cuda(0)?;
    println!("DKR_R124_S44_DEVICE_INIT_US={}", started.elapsed().as_micros());
    assert!(gpu.as_cuda_device()?.dkr_moe_enabled());
    let started = std::time::Instant::now();
    let x = candle_core::Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &gpu)?;
    assert_eq!(x.affine(2., 1.)?.flatten_all()?.to_vec1::<f32>()?, vec![3., 5., 7., 9.]);
    println!("DKR_R124_S44_FIRST_OPERATION_US={}", started.elapsed().as_micros());
    Ok(())
}

#[cfg(feature = "cuda-dkr-embedded")]
#[test]
#[ignore = "run in a process with MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem"]
fn dkr_embedded_rejects_filesystem() {
    assert_eq!(std::env::var("MISTRALRS_DKR_ARTIFACT_SOURCE").unwrap(), "filesystem");
    match Device::new_cuda(0) {
        Ok(_) => panic!("filesystem override admitted"),
        Err(e) => assert!(e.to_string().contains("cuda-dkr-embedded requires embedded artifact source"), "{e}"),
    }
}

#[cfg(feature = "cuda-dkr-embedded")]
#[test]
#[ignore = "run in a process with CANDLE_DKR_S31_DIR set"]
fn dkr_embedded_rejects_directory_override() {
    assert!(std::env::var_os("CANDLE_DKR_S31_DIR").is_some());
    match Device::new_cuda(0) {
        Ok(_) => panic!("runtime directory override admitted"),
        Err(e) => assert!(e.to_string().contains("cuda-dkr-embedded forbids CANDLE_DKR_S31_DIR"), "{e}"),
    }
}
