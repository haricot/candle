use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    batch: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    groups: usize,
}

fn deterministic_values(len: usize, mul: usize, bias: isize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let v = ((i * mul) % 101) as isize + bias;
            v as f32 / 64.0
        })
        .collect()
}

fn legacy_grouped_conv2d(
    x: &Tensor,
    kernel: &Tensor,
    padding: usize,
    stride: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let mut ys = Vec::with_capacity(groups);
    for (xg, kg) in xs.iter().zip(ks.iter()) {
        ys.push(xg.conv2d(kg, padding, stride, 1, 1)?);
    }
    Ok(Tensor::cat(&ys, 1)?)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) * 0.5
    } else {
        values[n / 2]
    }
}

fn timed<F>(device: &Device, runs: usize, mut f: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        device.synchronize()?;
        let t0 = Instant::now();
        let _y = f()?;
        device.synchronize()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(median(&mut samples))
}

fn run_case(device: &Device, case: Case) -> Result<bool> {
    let x_len = case.batch * case.c_in * case.h * case.w;
    let k_len = case.c_out * (case.c_in / case.groups) * case.kernel * case.kernel;
    let x = Tensor::from_vec(
        deterministic_values(x_len, 37, -50),
        (case.batch, case.c_in, case.h, case.w),
        device,
    )?;
    let kernel = Tensor::from_vec(
        deterministic_values(k_len, 53, -50),
        (
            case.c_out,
            case.c_in / case.groups,
            case.kernel,
            case.kernel,
        ),
        device,
    )?;

    std::env::set_var("CANDLE_CUDNN_NATIVE_GROUPED_STRICT", "1");

    let native = x.conv2d(
        &kernel,
        case.padding,
        case.stride,
        1,
        case.groups,
    )?;
    let legacy = legacy_grouped_conv2d(
        &x,
        &kernel,
        case.padding,
        case.stride,
        case.groups,
    )?;
    device.synchronize()?;

    let native_v = native.flatten_all()?.to_vec1::<f32>()?;
    let legacy_v = legacy.flatten_all()?.to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut numerical_pass = true;
    for (&a, &b) in native_v.iter().zip(&legacy_v) {
        let abs = (a - b).abs();
        let rel = abs / b.abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > 1e-4 + 1e-4 * b.abs() {
            numerical_pass = false;
        }
    }

    for _ in 0..3 {
        let _ = x.conv2d(
            &kernel,
            case.padding,
            case.stride,
            1,
            case.groups,
        )?;
        let _ = legacy_grouped_conv2d(
            &x,
            &kernel,
            case.padding,
            case.stride,
            case.groups,
        )?;
        device.synchronize()?;
    }

    let runs = 10;
    let native_ms = timed(device, runs, || {
        Ok(x.conv2d(
            &kernel,
            case.padding,
            case.stride,
            1,
            case.groups,
        )?)
    })?;
    let legacy_ms = timed(device, runs, || {
        legacy_grouped_conv2d(
            &x,
            &kernel,
            case.padding,
            case.stride,
            case.groups,
        )
    })?;
    let speedup = legacy_ms / native_ms;

    println!(
        "CASE name={} b={} cin={} cout={} h={} w={} k={} stride={} groups={} native_ms={:.6} legacy_ms={:.6} speedup={:.3}x max_abs={:.9} max_rel={:.9} numerical_pass={}",
        case.name,
        case.batch,
        case.c_in,
        case.c_out,
        case.h,
        case.w,
        case.kernel,
        case.stride,
        case.groups,
        native_ms,
        legacy_ms,
        speedup,
        max_abs,
        max_rel,
        numerical_pass
    );
    Ok(numerical_pass)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("=== CANDLE NATIVE GROUPED CUDNN PR AUDIT ===");
    println!("schema=candle.grouped-cudnn-pr-audit.v1");
    println!("device={device:?}");
    println!("dtype=f32 dilation=1 strict_native=true");

    let cases = [
        Case {
            name: "generic-g2-b1-k3-s1",
            batch: 1,
            c_in: 16,
            c_out: 24,
            h: 32,
            w: 24,
            kernel: 3,
            padding: 1,
            stride: 1,
            groups: 2,
        },
        Case {
            name: "generic-g4-b1-k3-s2",
            batch: 1,
            c_in: 32,
            c_out: 48,
            h: 32,
            w: 24,
            kernel: 3,
            padding: 1,
            stride: 2,
            groups: 4,
        },
        Case {
            name: "generic-g8-b3-k5-s1",
            batch: 3,
            c_in: 32,
            c_out: 64,
            h: 24,
            w: 20,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 8,
        },
        Case {
            name: "depthwise-c48-b1-k5-s1",
            batch: 1,
            c_in: 48,
            c_out: 48,
            h: 64,
            w: 48,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 48,
        },
        Case {
            name: "depthwise-c96-b3-k5-s1",
            batch: 3,
            c_in: 96,
            c_out: 96,
            h: 32,
            w: 24,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 96,
        },
        Case {
            name: "depthwise-c384-b1-k5-s1",
            batch: 1,
            c_in: 384,
            c_out: 384,
            h: 8,
            w: 6,
            kernel: 5,
            padding: 2,
            stride: 1,
            groups: 384,
        },
    ];

    let mut ok = true;
    for case in cases {
        ok &= run_case(&device, case)?;
    }

    if !ok {
        bail!("STATUS=FAIL numerical parity");
    }
    println!("STATUS=PASS");
    Ok(())
}
