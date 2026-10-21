use candle_core::{
    diagnostics::{self, Capture, Payload},
    stateful::causal_depthwise_conv1d_step,
    DType, Device, Tensor,
};
use std::path::PathBuf;

fn directory(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("candle-d7-{label}-{}", std::process::id()))
}

#[test]
fn capture_and_immutable_state_contract() -> candle_core::Result<()> {
    assert!(!diagnostics::enabled());
    diagnostics::event("off", || panic!("trace-off evaluated payload"));
    let capture =
        Capture::start("test".into(), "nonce".into(), 20000, Payload::new(), false).unwrap();
    {
        let _root = diagnostics::scope_in_window("engine.forward", Some("active"), || {
            Payload::new().text("phase", "decode")
        });
        for dtype in [DType::F32, DType::F16] {
            for k in [1, 3, 4] {
                let state = Tensor::arange(0f32, (2 * 4 * k) as f32, &Device::Cpu)?
                    .reshape((2, 4, k))?
                    .to_dtype(dtype)?;
                let snapshot = state
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let x = Tensor::ones((2, 4, 1), dtype, &Device::Cpu)?;
                let weight = Tensor::ones((4, 1, k), dtype, &Device::Cpu)?;
                let (out, next) = causal_depthwise_conv1d_step(&x, &state, &weight)?;
                let next_snapshot = next.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                let values = out.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                for (row, actual) in values.iter().enumerate() {
                    let expected = (1..k).map(|j| snapshot[row * k + j]).sum::<f32>() + 1.;
                    assert_eq!(*actual, expected);
                    assert_eq!(next_snapshot[row * k + k - 1], 1.);
                }
                let _ = causal_depthwise_conv1d_step(&x, &next, &weight)?;
                assert_eq!(
                    snapshot,
                    state
                        .flatten_all()?
                        .to_dtype(DType::F32)?
                        .to_vec1::<f32>()?
                );
                assert_eq!(
                    next_snapshot,
                    next.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?
                );
            }
        }
        let base = Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?;
        let offset = base.narrow(0, 1, 2)?.t()?;
        let _ = offset.sqr()?;
    }
    let out = directory("complete");
    let _ = std::fs::remove_dir_all(&out);
    capture.finish(&out).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("trace-manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["complete"], true);
    let events = std::fs::read_to_string(out.join("events.ndjson")).unwrap();
    assert!(events.contains("unsupported_device"));
    assert!(events.contains("\"offset\":4"));
    assert!(events.contains("\"contiguous\":false"));
    let _ = std::fs::remove_dir_all(out);

    let capture = Capture::start("loss".into(), "nonce2".into(), 1, Payload::new(), false).unwrap();
    {
        let _scope = diagnostics::scope("overflow", Payload::new);
    }
    let out = directory("loss");
    let _ = std::fs::remove_dir_all(&out);
    capture.finish(&out).unwrap();
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("trace-manifest.json")).unwrap()).unwrap();
    assert_eq!(m["complete"], false);
    assert!(m["dropped_records"].as_u64().unwrap() > 0);
    let _ = std::fs::remove_dir_all(out);
    capture_drains_admitted_work();
    capture_cpu_clocks_and_observer();
    Ok(())
}

fn capture_drains_admitted_work() {
    for payload_only in [false, true] {
        let capture =
            Capture::start("drain".into(), "nonce3".into(), 32, Payload::new(), false).unwrap();
        let (ready, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let wait_for_finish = || {
                while diagnostics::enabled() {
                    std::thread::yield_now();
                }
                diagnostics::event("off", || panic!("draining capture admitted new work"));
                std::thread::sleep(std::time::Duration::from_millis(20));
            };
            if payload_only {
                diagnostics::event("memory.snapshot", || {
                    ready.send(()).unwrap();
                    wait_for_finish();
                    Payload::new()
                });
            } else {
                let scope = diagnostics::scope("op", Payload::new);
                ready.send(()).unwrap();
                wait_for_finish();
                drop(scope);
            }
        });
        received.recv().unwrap();
        let out = directory("drain");
        let _ = std::fs::remove_dir_all(&out);
        capture.finish(&out).unwrap();
        worker.join().unwrap();
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("trace-manifest.json")).unwrap())
                .unwrap();
        assert_eq!(m["complete"], true);
        assert_eq!(m["drain_complete"], true);
        assert_eq!(m["open_scopes"], 0);
        assert_eq!(m["dropped_records"], 0);
        assert_eq!(m["record_count"], if payload_only { 1 } else { 2 });
        let events = std::fs::read_to_string(out.join("events.ndjson")).unwrap();
        assert!(events.contains(if payload_only {
            "memory.snapshot"
        } else {
            "op.end"
        }));
        let _ = std::fs::remove_dir_all(out);
    }

    let capture =
        Capture::start("timeout".into(), "nonce4".into(), 32, Payload::new(), false).unwrap();
    let held = diagnostics::scope("op", Payload::new);
    let out = directory("timeout");
    let _ = std::fs::remove_dir_all(&out);
    capture.finish(&out).unwrap();
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("trace-manifest.json")).unwrap()).unwrap();
    assert_eq!(m["complete"], false);
    assert_eq!(m["drain_complete"], false);
    assert_eq!(m["open_scopes"], 1);
    drop(held);
    let _ = std::fs::remove_dir_all(out);
}

fn capture_cpu_clocks_and_observer() {
    use std::sync::{Arc, Mutex};
    struct Observer(Mutex<Vec<(bool, u64)>>);
    impl diagnostics::ScopeObserver for Observer {
        fn enter(&self, id: u64) -> bool {
            self.0.lock().unwrap().push((true, id));
            true
        }
        fn exit(&self, id: u64) {
            self.0.lock().unwrap().push((false, id));
        }
    }
    assert!(diagnostics::set_scope_observer(None).is_err());
    let capture = Capture::start(
        "cpu".into(),
        "cpu-nonce".into(),
        1024,
        Payload::new(),
        false,
    )
    .unwrap();
    let observer = Arc::new(Observer(Mutex::new(Vec::new())));
    diagnostics::set_scope_observer(Some(observer.clone())).unwrap();
    let outer = diagnostics::scope("outer", Payload::new);
    {
        let _inner = diagnostics::scope("inner", Payload::new);
        let mut x = 1u64;
        for _ in 0..100_000 {
            x = std::hint::black_box(x.wrapping_mul(3).wrapping_add(1));
        }
        std::hint::black_box(x);
    }
    diagnostics::set_scope_observer(None).unwrap();
    drop(outer);
    drop(diagnostics::scope("unobserved", Payload::new));
    let pairs = observer.0.lock().unwrap().clone();
    assert_eq!(pairs, vec![(true, 1), (true, 2), (false, 2), (false, 1)]);
    let out = directory("cpu-clocks");
    let _ = std::fs::remove_dir_all(&out);
    capture.finish(&out).unwrap();
    let records: Vec<serde_json::Value> = std::fs::read_to_string(out.join("events.ndjson"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    if cfg!(target_os = "linux") {
        for event in &records {
            assert!(event["cpu"]["os_thread_id"].as_u64().unwrap() > 0);
            assert!(event["cpu"]["monotonic_ns"].as_u64().unwrap() > 0);
            assert!(event["cpu"]["thread_cpu_ns"].as_u64().unwrap() > 0);
        }
        assert!(
            records[3]["cpu"]["thread_cpu_ns"].as_u64().unwrap()
                >= records[0]["cpu"]["thread_cpu_ns"].as_u64().unwrap()
        );
    }
    let _ = std::fs::remove_dir_all(out);
}
