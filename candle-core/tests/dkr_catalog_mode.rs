//! Explicit process-level qualification for the decoupled feature.
//! Invoke the ignored missing-provider test with all catalogue env vars unset
//! and without DKR_EMBED_NATIVE_DIR at build time.
#![cfg(all(feature = "cuda-dkr-catalog", not(feature = "cuda-dkr-s31")))]

use candle_core::Device;

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
