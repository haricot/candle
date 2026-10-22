"""Isolated CPU MKL frontend/projection profile; original R8B remains intact."""
from .c3_r8_source import once
from .c3_r9_source import outputs as text_outputs


def outputs(parent, root, metadata):
    result=text_outputs(parent,root,metadata)
    probe=result['src/bin/hdb2c_r9_probe.rs']
    probe=once(probe,'let a=Args::parse(); fs::create_dir_all(&a.output)?;',
        'let a=Args::parse(); ensure!(a.device=="cpu","R10 profile is CPU-only"); fs::create_dir_all(&a.output)?;')
    probe=once(probe,'from_config_value,get_logmel,load_audio','from_config_value,load_audio')
    probe=once(probe,'let mut model=from_config_value(cfg,vb)?;',
        'let mut model=from_config_value(cfg,vb.clone())?;\n'
        '    fs::write(a.output.join("cpu_backend.json"),serde_json::to_vec_pretty(&json!({\n'
        '        "profile":"R10_NATIVE_CPU_MKL_CHECKPOINT_FRONTEND_FUSED_PROJECTION",\n'
        '        "mkl":parakeet_candle::parakeet::hdb2c_r10_mkl::profile(),\n'
        '        "threads":candle::utils::get_num_threads(),"torch_runtime_linked":false}))?)?;')
    probe=once(probe,'get_logmel(&audio,&m.preprocessor_config,&dev)?',
        'parakeet_candle::parakeet::hdb2c_r10_frontend::get_logmel(&audio,&m.preprocessor_config,&dev,&vb)?')
    probe=once(probe,'parakeet_candle::parakeet::audio::hdb2c8_frontend_operators(&cases.join("frontend"),&a.output.join("frontend"),&m.preprocessor_config)?;',
        'parakeet_candle::parakeet::hdb2c_r10_frontend::replay(&cases.join("frontend"),&a.output.join("frontend"),&m.preprocessor_config,&vb)?;')
    conformer=(parent/'src/parakeet/conformer.rs').read_text()
    conformer=once(conformer,'let x = self.out.forward(&x)?;',
        'let x = crate::parakeet::hdb2c_r10_mkl::linear(&self.out,&x)?;')
    conformer=once(conformer,'&pre.out.forward(&x)?',
        '&crate::parakeet::hdb2c_r10_mkl::linear(&pre.out,&x)?')
    result.update({'src/bin/hdb2c_r10_probe.rs':probe,
        'src/parakeet/conformer.rs':conformer,
        'src/parakeet/mod.rs':result['src/parakeet/mod.rs']+
            '\npub mod hdb2c_r10_mkl;\npub mod hdb2c_r10_frontend;\n',
        'build.rs':(root/'rust/hdb2c_r10_build.rs').read_text()})
    for name in ('hdb2c_r10_mkl.rs','hdb2c_r10_frontend.rs','hdb2c_r10_mkl.c'):
        result['src/parakeet/'+name]=(root/'rust'/name).read_text()
    return result
