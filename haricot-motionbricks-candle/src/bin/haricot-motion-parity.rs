use anyhow::{bail, Context, Result};
use haricot_motionbricks_candle::{
    encode_rust_sequence, G1Assets, G1ParitySequence, OfficialMotionBricksReference,
};

#[derive(Clone, Copy, Debug)]
struct Stats {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
}

fn arg_value(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == name {
            return args.next();
        }
    }
    None
}

fn parse_or<T>(name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match arg_value(name) {
        Some(v) => v
            .parse::<T>()
            .map_err(|e| anyhow::anyhow!("invalid {name}={v}: {e}")),
        None => Ok(default),
    }
}

fn main() -> Result<()> {
    if std::env::args().any(|arg| arg == "--help" || arg == "-h") {
        println!("haricot-motion-parity — v0.2.1-r2 gate reporting cleanup");
        println!("usage:");
        println!(
            "  haricot-motion-parity --g1-assets <json> --input <g1-sequence.json> \
             --reference <official.json> [--atol <f64>] [--rmse-tol <f64>]"
        );
        println!();
        println!("threshold policy:");
        println!("  raw/groups:      atol=ATOL, rmse_tol=RMSE_TOL");
        println!("  normalized:      atol=10*ATOL, rmse_tol=10*RMSE_TOL");
        println!("  foot_contacts:   exact (atol=0, rmse_tol=0)");
        return Ok(());
    }

    let assets_path = arg_value("--g1-assets").context("missing --g1-assets <json>")?;
    let input_path = arg_value("--input").context("missing --input <g1-sequence.json>")?;
    let reference_path = arg_value("--reference").context("missing --reference <official.json>")?;
    let atol: f64 = parse_or("--atol", 2e-5_f64)?;
    let rmse_tol: f64 = parse_or("--rmse-tol", 5e-6_f64)?;

    let assets = G1Assets::load_json(&assets_path)?;
    if !assets.has_stats() {
        bail!("official parity requires G1 assets with released 418-D stats")
    }
    let input = G1ParitySequence::load_json(&input_path)?;
    let reference = OfficialMotionBricksReference::load_json(&reference_path)?;
    if input.frames.len() != reference.frames {
        bail!(
            "input/reference frame mismatch: {} vs {}",
            input.frames.len(),
            reference.frames
        )
    }
    if (input.fps - reference.fps).abs() > 1e-6 {
        bail!("input/reference fps mismatch")
    }

    let rust = encode_rust_sequence(assets, &input)?;
    let rust_dual = rust.iter().map(|x| x.dual.clone()).collect::<Vec<_>>();
    let rust_global = rust.iter().map(|x| x.global.clone()).collect::<Vec<_>>();
    let rust_local = rust.iter().map(|x| x.local.clone()).collect::<Vec<_>>();
    let rust_nd = rust
        .iter()
        .map(|x| x.normalized_dual.clone().expect("stats validated"))
        .collect::<Vec<_>>();
    let rust_ng = rust
        .iter()
        .map(|x| x.normalized_global.clone().expect("stats validated"))
        .collect::<Vec<_>>();
    let rust_nl = rust
        .iter()
        .map(|x| x.normalized_local.clone().expect("stats validated"))
        .collect::<Vec<_>>();

    println!("=== HARICOT V0.2.1-R2 OFFICIAL MOTIONBRICKS FEATURE PARITY ===");
    println!("frames={}", reference.frames);
    println!("fps={:.6}", reference.fps);
    println!("source={}", reference.source);
    println!("source_rev={}", reference.source_rev);
    let normalized_atol = atol * 10.0;
    let normalized_rmse_tol = rmse_tol * 10.0;

    println!("raw_atol={atol:.3e}");
    println!("raw_rmse_tol={rmse_tol:.3e}");
    println!("normalized_atol={normalized_atol:.3e}");
    println!("normalized_rmse_tol={normalized_rmse_tol:.3e}");
    println!("foot_contacts_atol=0.000e0");
    println!("foot_contacts_rmse_tol=0.000e0");
    println!("gate_policy=raw:1x,normalized:10x,foot_contacts:exact");

    let mut raw_pass = true;
    raw_pass &= report("dual_418", &rust_dual, &reference.dual, atol, rmse_tol)?;
    raw_pass &= report("global_414", &rust_global, &reference.global, atol, rmse_tol)?;
    raw_pass &= report("local_413", &rust_local, &reference.local, atol, rmse_tol)?;

    let mut groups_pass = true;
    groups_pass &= report_slice(
        "global_root",
        &rust_dual,
        &reference.dual,
        0,
        5,
        atol,
        rmse_tol,
    )?;
    groups_pass &= report_slice(
        "local_root",
        &rust_dual,
        &reference.dual,
        5,
        9,
        atol,
        rmse_tol,
    )?;
    groups_pass &= report_slice(
        "ric_data",
        &rust_dual,
        &reference.dual,
        9,
        108,
        atol,
        rmse_tol,
    )?;
    groups_pass &= report_slice(
        "global_rot_data",
        &rust_dual,
        &reference.dual,
        108,
        312,
        atol,
        rmse_tol,
    )?;
    groups_pass &= report_slice(
        "local_vel",
        &rust_dual,
        &reference.dual,
        312,
        414,
        atol,
        rmse_tol,
    )?;
    groups_pass &= report_slice(
        "foot_contacts",
        &rust_dual,
        &reference.dual,
        414,
        418,
        0.0,
        0.0,
    )?;

    let mut normalized_pass = true;
    normalized_pass &= report(
        "normalized_dual_418",
        &rust_nd,
        &reference.normalized_dual,
        normalized_atol,
        normalized_rmse_tol,
    )?;
    normalized_pass &= report(
        "normalized_global_414",
        &rust_ng,
        &reference.normalized_global,
        normalized_atol,
        normalized_rmse_tol,
    )?;
    normalized_pass &= report(
        "normalized_local_413",
        &rust_nl,
        &reference.normalized_local,
        normalized_atol,
        normalized_rmse_tol,
    )?;

    println!(
        "OFFICIAL_MOTIONBRICKS_RAW_GATE={}",
        if raw_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "OFFICIAL_MOTIONBRICKS_FEATURE_GROUP_GATE={}",
        if groups_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "OFFICIAL_MOTIONBRICKS_NORMALIZED_GATE={}",
        if normalized_pass { "PASS" } else { "FAIL" }
    );
    let pass = raw_pass && groups_pass && normalized_pass;
    println!("V0_2_1_GATE={}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        bail!("official MotionBricks parity gate failed")
    }
    Ok(())
}

fn report(
    name: &str,
    got: &[Vec<f32>],
    reference: &[Vec<f32>],
    atol: f64,
    rmse_tol: f64,
) -> Result<bool> {
    let stats = stats(got, reference, None)?;
    let pass = stats.max_abs <= atol && stats.rmse <= rmse_tol;
    println!(
        "tensor={name} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} atol={atol:.9e} rmse_tol={rmse_tol:.9e} pass={pass}",
        stats.max_abs, stats.mean_abs, stats.rmse
    );
    Ok(pass)
}

fn report_slice(
    name: &str,
    got: &[Vec<f32>],
    reference: &[Vec<f32>],
    start: usize,
    end: usize,
    atol: f64,
    rmse_tol: f64,
) -> Result<bool> {
    let stats = stats(got, reference, Some(start..end))?;
    let pass = stats.max_abs <= atol && stats.rmse <= rmse_tol;
    println!(
        "feature={name} range={start}..{end} max_abs={:.9e} mean_abs={:.9e} rmse={:.9e} atol={atol:.9e} rmse_tol={rmse_tol:.9e} pass={pass}",
        stats.max_abs, stats.mean_abs, stats.rmse
    );
    Ok(pass)
}

fn stats(
    got: &[Vec<f32>],
    reference: &[Vec<f32>],
    range: Option<std::ops::Range<usize>>,
) -> Result<Stats> {
    if got.len() != reference.len() {
        bail!("frame count mismatch in parity comparison")
    }
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;
    for (t, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
        if a.len() != b.len() {
            bail!("frame {t}: dimension mismatch {} vs {}", a.len(), b.len())
        }
        let r = range.clone().unwrap_or(0..a.len());
        if r.end > a.len() {
            bail!("comparison range out of bounds")
        }
        for i in r {
            let d = (a[i] as f64 - b[i] as f64).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            sum_sq += d * d;
            n += 1;
        }
    }
    if n == 0 {
        bail!("empty parity comparison")
    }
    Ok(Stats {
        max_abs,
        mean_abs: sum_abs / n as f64,
        rmse: (sum_sq / n as f64).sqrt(),
    })
}
