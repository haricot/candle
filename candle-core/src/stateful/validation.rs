use crate::{
    diagnostics::{Capture, Payload},
    stateful::{causal_depthwise_conv1d_step, causal_depthwise_conv1d_step_reference},
    DType, Device, Tensor,
};
use half::f16;
use serde_json::json;
use std::path::Path;

fn bits(tensor: &Tensor) -> crate::Result<Vec<u16>> {
    Ok(tensor
        .flatten_all()?
        .to_vec1::<f16>()?
        .into_iter()
        .map(f16::to_bits)
        .collect())
}

fn values(count: usize, seed: usize, pattern: usize) -> Vec<f16> {
    (0..count)
        .map(|i| {
            let value = match pattern {
                0 => 0.0,
                1 => {
                    if (i + seed) % 3 == 0 {
                        1.0
                    } else {
                        -0.25
                    }
                }
                2 => ((i * 37 + seed * 17) % 4093) as f32 / 1024.0 - 2.0,
                3 => {
                    if (i + seed) % 2 == 0 {
                        0.00006103515625
                    } else {
                        -0.000030517578125
                    }
                }
                4 => ((i + seed) % 7) as f32 * 0.000000059604644775390625,
                _ => ((i * 11 + seed * 23) % 257) as f32 / 128.0 - 1.0,
            };
            f16::from_f32(value)
        })
        .collect()
}

fn check_fallbacks(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    let x = Tensor::from_vec(values(2048, 5, 2), (1, 2048, 1), device)?;
    let state = Tensor::from_vec(values(2048 * 3, 3, 2), (1, 2048, 3), device)?;
    let weight = Tensor::from_vec(values(2048 * 3, 7, 5), (2048, 1, 3), device)?;
    let offset_x = Tensor::cat(&[&x, &x], 0)?.narrow(0, 1, 1)?;
    let offset_state = Tensor::cat(&[&state, &state], 0)?.narrow(0, 1, 1)?;
    let offset_weight = Tensor::cat(&[&weight, &weight], 0)?.narrow(0, 2048, 2048)?;
    let strided_state = state.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
    let variable_weight = crate::Var::from_tensor(&weight)?;
    for (x, state, weight) in [
        (&offset_x, &state, &weight),
        (&x, &offset_state, &weight),
        (&x, &state, &offset_weight),
        (&x, &strided_state, &weight),
        (&x, &state, variable_weight.as_tensor()),
    ] {
        let before = bits(state)?;
        let (expected, expected_next) = causal_depthwise_conv1d_step_reference(x, state, weight)?;
        let (actual, next) = causal_depthwise_conv1d_step(x, state, weight)?;
        if bits(&actual)? != bits(&expected)?
            || bits(&next)? != bits(&expected_next)?
            || bits(state)? != before
        {
            return Err("stateful layout/autograd fallback parity failed".into());
        }
    }
    Ok(())
}

pub fn validate_f16_candidate(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if out.exists() {
        return Err("output directory exists".into());
    }
    std::fs::create_dir_all(out)?;
    if std::env::var("CANDLE_ASD_ALLOW_STATEFUL_CANDIDATE").as_deref() != Ok("1") {
        return Err("stateful candidate opt-in is required".into());
    }
    let device = Device::new_cuda(0)?;
    let mut cases = 0;
    let mut steps = 0;
    let mut expected_submissions = 0;
    let mut fork_checks = 0;
    let nonce = std::env::var("D7_STATEFUL_VALIDATION_NONCE")
        .unwrap_or_else(|_| format!("{:032x}", rand::random::<u128>()));
    for trace in [false, true] {
        let capture = if trace {
            Some(Capture::start(
                "stateful-f16-parity".into(),
                nonce.clone(),
                500000,
                Payload::new().text("source", "synthetic_candidate_parity"),
                false,
            )?)
        } else {
            None
        };
        {
            let _root = crate::diagnostics::scope_in_window(
                "engine.forward",
                Some("validation"),
                Payload::new,
            );
            for batch in [1, 2] {
                for channels in [7, 2048] {
                    for pattern in 0..6 {
                        let mut reference_state = Tensor::from_vec(
                            values(batch * channels * 3, 3, pattern),
                            (batch, channels, 3),
                            &device,
                        )?;
                        let mut candidate_state = reference_state.copy()?;
                        let initial = candidate_state.clone();
                        let initial_bits = bits(&initial)?;
                        let weight = Tensor::from_vec(
                            values(channels * 3, 7, 5),
                            (channels, 1, 3),
                            &device,
                        )?;
                        let weight_bits = bits(&weight)?;
                        let mut prior_next = None;
                        for step in 0..16 {
                            let x = Tensor::from_vec(
                                values(batch * channels, step + 11, pattern),
                                (batch, channels, 1),
                                &device,
                            )?;
                            let x_bits = bits(&x)?;
                            let old_state = candidate_state.clone();
                            let old_bits = bits(&old_state)?;
                            let (expected, next_reference) =
                                causal_depthwise_conv1d_step_reference(
                                    &x,
                                    &reference_state,
                                    &weight,
                                )?;
                            let (actual, next_candidate) =
                                causal_depthwise_conv1d_step(&x, &candidate_state, &weight)?;
                            let actual_bits = bits(&actual)?;
                            let expected_bits = bits(&expected)?;
                            if actual_bits != expected_bits {
                                let differences: Vec<_> = actual_bits.iter().zip(&expected_bits).enumerate()
                                    .filter(|(_, (a,b))| a != b).take(16)
                                    .map(|(index, (&actual, &expected))| json!({"index":index,"actual_f16_bits":actual,"expected_f16_bits":expected})).collect();
                                std::fs::write(
                                    out.join("failure-details.json"),
                                    serde_json::to_vec_pretty(
                                        &json!({"batch":batch,"channels":channels,"pattern":pattern,"step":step,"differences":differences}),
                                    )?,
                                )?;
                                return Err(format!("F16 output bit parity failed B={batch} C={channels} pattern={pattern} step={step}").into());
                            }
                            if bits(&next_candidate)? != bits(&next_reference)? {
                                return Err(format!("state bit parity failed B={batch} C={channels} pattern={pattern} step={step}").into());
                            }
                            if bits(&old_state)? != old_bits
                                || bits(&initial)? != initial_bits
                                || bits(&weight)? != weight_bits
                                || bits(&x)? != x_bits
                            {
                                return Err("candidate mutated a prior input or state".into());
                            }
                            if let Some((prior, snapshot)) = &prior_next {
                                if bits(prior)? != *snapshot {
                                    return Err("candidate mutated a retained next-state".into());
                                }
                            }
                            prior_next = Some((next_candidate.clone(), bits(&next_candidate)?));
                            reference_state = next_reference;
                            candidate_state = next_candidate;
                            steps += 1;
                            if trace {
                                expected_submissions += 1;
                            }
                        }
                        let retained = bits(&candidate_state)?;
                        let fork_input = Tensor::from_vec(
                            values(batch * channels, 77, pattern),
                            (batch, channels, 1),
                            &device,
                        )?;
                        let (expected, expected_next) =
                            causal_depthwise_conv1d_step_reference(&fork_input, &initial, &weight)?;
                        let (actual, next) =
                            causal_depthwise_conv1d_step(&fork_input, &initial, &weight)?;
                        if bits(&actual)? != bits(&expected)?
                            || bits(&next)? != bits(&expected_next)?
                            || bits(&initial)? != initial_bits
                            || bits(&candidate_state)? != retained
                        {
                            return Err(
                                "fork from retained checkpoint changed output or prior state"
                                    .into(),
                            );
                        }
                        fork_checks += 1;
                        if trace {
                            expected_submissions += 1;
                        }
                        cases += 1;
                    }
                }
            }
            check_fallbacks(&device)?;
        }
        if let Some(capture) = capture {
            capture.finish(&out.join("trace"))?;
        }
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("trace/trace-manifest.json"))?)?;
    if manifest["nonce"] != nonce
        || manifest["run_id"] != "stateful-f16-parity"
        || manifest["complete"] != true
        || manifest["dropped_records"] != 0
        || manifest["open_scopes"] != 0
    {
        return Err("candidate validation trace incomplete".into());
    }
    let mut submissions = 0;
    let mut artifacts = std::collections::BTreeSet::new();
    let raw = std::fs::read_to_string(out.join("trace/events.ndjson"))?;
    for line in raw.lines() {
        let event: serde_json::Value = serde_json::from_str(line)?;
        if event["event"] == "asd.submitted"
            && event["payload"]["abi"] == crate::stateful::SHORTCONV_F16_ABI
        {
            submissions += 1;
            artifacts.insert(
                event["payload"]["artifact_sha256"]
                    .as_str()
                    .ok_or("submission artifact missing")?
                    .to_owned(),
            );
        }
    }
    if submissions != expected_submissions || submissions == 0 {
        return Err(format!("candidate was not submitted for every checked step: {submissions}/{expected_submissions}").into());
    }
    if artifacts.len() != 1 {
        return Err("candidate validation used inconsistent artifacts".into());
    }
    let artifact_sha256 = artifacts
        .into_iter()
        .next()
        .ok_or("missing candidate artifact")?;
    if let Ok(expected) = std::env::var("D7_STATEFUL_ARTIFACT_SHA256") {
        if expected != artifact_sha256 {
            return Err("candidate validation artifact mismatch".into());
        }
    }
    let report = json!({"status":"PASS_CANDIDATE_F16_OUTPUT_STATE_AND_IMMUTABILITY", "nonce":nonce, "artifact_sha256":artifact_sha256, "device":format!("{:?}", device.location()), "dtype":format!("{:?}", DType::F16), "abi":crate::stateful::SHORTCONV_F16_ABI, "cases":cases, "steps":steps, "fork_checks":fork_checks, "fallback_cases":10, "candidate_submissions":submissions, "output_parity":"bit_exact_against_same_device_reference", "state_parity":"bit_exact", "state_contract":"immutable_inputs_and_retained_next_states", "batches":[1,2], "channels":[7,2048], "kernel":3, "patterns":6, "steps_per_case":16, "trace_off_on":true, "engine_gain_qualified":false});
    std::fs::write(
        out.join("validation.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}
