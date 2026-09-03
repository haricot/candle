use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use std::time::Instant;

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
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let xs = x.chunk(groups, 1)?;
    let ks = kernel.chunk(groups, 0)?;
    let mut ys = Vec::with_capacity(groups);
    for (xg, kg) in xs.iter().zip(ks.iter()) {
        ys.push(xg.conv2d(kg, padding, stride, dilation, 1)?);
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

fn run_case(device: &Device, c: usize, h: usize, w: usize) -> Result<bool> {
    let x_data = deterministic_values(c * h * w, 37, -50);
    let k_data = deterministic_values(c * 5 * 5, 53, -50);
    let x = Tensor::from_vec(x_data, (1, c, h, w), device)?;
    let kernel = Tensor::from_vec(k_data, (c, 1, 5, 5), device)?;

    std::env::set_var("CANDLE_CUDNN_NATIVE_GROUPED_STRICT", "1");

    let native = x.conv2d(&kernel, 2, 1, 1, c)?;
    let legacy = legacy_grouped_conv2d(&x, &kernel, 2, 1, 1, c)?;
    device.synchronize()?;

    let max_abs = native
        .sub(&legacy)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()? as f64;

    for _ in 0..3 {
        let _ = x.conv2d(&kernel, 2, 1, 1, c)?;
        let _ = legacy_grouped_conv2d(&x, &kernel, 2, 1, 1, c)?;
        device.synchronize()?;
    }

    let runs = 10;
    let native_ms = timed(device, runs, || Ok(x.conv2d(&kernel, 2, 1, 1, c)?))?;
    let legacy_ms = timed(device, runs, || {
        legacy_grouped_conv2d(&x, &kernel, 2, 1, 1, c)
    })?;
    let speedup = legacy_ms / native_ms;
    let numerical_pass = max_abs <= 1e-4;

    println!(
        "CASE c={c} h={h} w={w} native_ms={native_ms:.6} legacy_ms={legacy_ms:.6} speedup={speedup:.3}x max_abs={max_abs:.9} numerical_pass={numerical_pass}"
    );
    Ok(numerical_pass)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("=== CANDLE NATIVE GROUPED CUDNN PROBE ===");
    println!("schema=candle.grouped-cudnn-probe.v1");
    println!("device={device:?}");
    println!("kernel=depthwise-5x5 stride=1 padding=2 dtype=f32");
    println!("strict_native=true");

    let mut ok = true;
    for &(c, h, w) in &[(48, 64, 48), (96, 32, 24), (192, 16, 12), (384, 8, 6)] {
        ok &= run_case(&device, c, h, w)?;
    }

    if !ok {
        bail!("STATUS=FAIL numerical parity");
    }
    println!("STATUS=PASS");
    Ok(())
}
