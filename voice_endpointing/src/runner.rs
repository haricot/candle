//! Resident CPU endpoint models; the callback may hold a resident CUDA ASR.
use crate::{
    endpoint::{Config, Pipeline, Turn},
    firered::FireRedVad,
    frontend::{read_wav, write_wav},
    smart_turn::SmartTurn,
};
use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser, Debug, Clone)]
#[command(about = "Native Rust/Candle FireRedVAD + Smart Turn. No Python or ONNX runtime.")]
pub struct Args {
    #[arg(long)]
    pub models: PathBuf,
    #[arg(long, conflicts_with = "input")]
    pub mic: bool,
    #[arg(long)]
    pub input: Option<PathBuf>,
    #[arg(long, default_value = "reports")]
    pub report_root: PathBuf,
    #[arg(long, default_value_t = 1)]
    pub max_turns: usize,
    #[arg(long, default_value_t = 1200)]
    pub endpoint_hold_ms: usize,
    #[arg(long, default_value_t = 3200)]
    pub max_hesitation_ms: usize,
    #[arg(long, default_value_t = 25000)]
    pub max_turn_ms: usize,
    #[arg(long, default_value_t = 0.5)]
    pub smart_threshold: f32,
    #[arg(long)]
    pub pw_target: Option<String>,
    #[arg(long)]
    pub asr_model_dir: Option<PathBuf>,
    #[arg(long, default_value = "cuda-cudnn")]
    pub asr_device: String,
    /// Optional existing R9 probe. The combined hdb1_native binary does not use this.
    #[arg(long)]
    pub asr_binary: Option<PathBuf>,
}

struct Capture(Child);
impl Drop for Capture {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn emit(pipeline: &mut Pipeline, log: &mut File) -> Result<()> {
    for row in pipeline.events.drain(..) {
        println!("HDB_NATIVE_EVENT={row}");
        writeln!(log, "{row}")?;
    }
    Ok(())
}

fn save_turn<F>(turn: Turn, root: &Path, index: usize, asr: &mut F) -> Result<Value>
where
    F: FnMut(&Path, &Path) -> Result<Value>,
{
    let folder = root.join(format!("turn-{index:03}"));
    fs::create_dir(&folder)?;
    let wav = folder.join("turn.wav");
    write_wav(&wav, &turn.pcm)?;
    let mut row = json!({"reason":turn.reason,"duration_sec":turn.pcm.len() as f64/16000.0,"samples":turn.pcm.len(),"wav":wav});
    let start = Instant::now();
    let outcome = asr(&wav, &folder);
    match outcome {
        Ok(result) => {
            row["asr"] = result;
            row["asr_wall_ms"] = json!(start.elapsed().as_secs_f64() * 1000.0);
        }
        Err(error) => {
            row["asr"] = json!({"status":"FAILED","error":error.to_string()});
            fs::write(
                folder.join("turn-report.json"),
                serde_json::to_vec_pretty(&row)?,
            )?;
            return Err(error);
        }
    }
    fs::write(
        folder.join("turn-report.json"),
        serde_json::to_vec_pretty(&row)?,
    )?;
    println!("HDB_NATIVE_TURN={row}");
    Ok(row)
}

pub fn run<F>(args: Args, mut asr: F) -> Result<PathBuf>
where
    F: FnMut(&Path, &Path) -> Result<Value>,
{
    ensure!(
        args.mic || args.input.is_some(),
        "Select --mic or --input PATH"
    );
    ensure!(args.max_turns > 0, "max-turns must be positive");
    let cfg = Config {
        threshold: args.smart_threshold,
        endpoint_hold_ms: args.endpoint_hold_ms,
        max_hesitation_ms: args.max_hesitation_ms,
        max_turn_ms: args.max_turn_ms,
    };
    cfg.validate()?;
    let mut pipeline = Pipeline::new(
        FireRedVad::load(&args.models.join("firered"))?,
        SmartTurn::load(&args.models.join("smart-turn"))?,
        cfg.clone(),
    )?;
    fs::create_dir_all(&args.report_root)?;
    let id = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = args
        .report_root
        .join(format!("hdb-native-{id}-{}", std::process::id()));
    fs::create_dir(&root)?;
    let mut log = File::create(root.join("events.jsonl"))?;
    let mut turns = Vec::new();
    let result = (|| -> Result<()> {
        if let Some(path) = &args.input {
            let pcm = read_wav(path)?;
            for chunk in pcm.chunks(1600) {
                let turn = pipeline.feed(chunk)?;
                emit(&mut pipeline, &mut log)?;
                if let Some(turn) = turn {
                    turns.push(save_turn(turn, &root, turns.len(), &mut asr)?);
                    if turns.len() >= args.max_turns {
                        return Ok(());
                    }
                }
            }
            if let Some(turn) = pipeline.eof()? {
                emit(&mut pipeline, &mut log)?;
                turns.push(save_turn(turn, &root, turns.len(), &mut asr)?);
            }
        } else {
            while turns.len() < args.max_turns {
                let mut cmd = Command::new("pw-record");
                cmd.args([
                    "--raw",
                    "--format",
                    "s16",
                    "--rate",
                    "16000",
                    "--channels",
                    "1",
                ]);
                if let Some(target) = &args.pw_target {
                    cmd.args(["--target", target]);
                }
                let stderr = File::create(root.join(format!("capture-{}.stderr", turns.len())))?;
                let mut capture = Capture(
                    cmd.arg("-")
                        .stdout(Stdio::piped())
                        .stderr(stderr)
                        .spawn()
                        .context("Cannot launch PipeWire pw-record")?,
                );
                let mut stream = capture.0.stdout.take().context("Missing PipeWire stream")?;
                println!(
                    "HDB_NATIVE_READY=LISTENING turn={} endpoint_device=Cpu",
                    turns.len()
                );
                std::io::stdout().flush()?;
                loop {
                    let mut raw = [0u8; 3200];
                    stream.read_exact(&mut raw).context(
                        "PipeWire capture ended before an accepted turn; see capture stderr",
                    )?;
                    let pcm: Vec<_> = raw
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    let turn = pipeline.feed(&pcm)?;
                    emit(&mut pipeline, &mut log)?;
                    if let Some(turn) = turn {
                        // Stop capture BEFORE ASR. Discard queued audio explicitly;
                        // do not replay stale speech as the following live turn.
                        drop(stream);
                        drop(capture);
                        println!("HDB_NATIVE_STATE=TRANSCRIBING microphone=PAUSED");
                        turns.push(save_turn(turn, &root, turns.len(), &mut asr)?);
                        break;
                    }
                }
            }
        }
        Ok(())
    })();
    let report = json!({"status":if result.is_ok(){"COMPLETE"}else{"FAILED"},"error":result.as_ref().err().map(ToString::to_string),
        "runtime":"RUST_CANDLE_NATIVE","python_runtime":false,"onnx_runtime":false,
        "endpoint_models_resident":true,"sample_clock_excludes_asr_pause":true,"config":cfg,"turns":turns});
    fs::write(
        root.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("HDB_NATIVE_REPORT={}", root.join("report.json").display());
    result?;
    Ok(root)
}

pub fn external_asr(args: &Args, wav: &Path, folder: &Path) -> Result<Value> {
    let Some(binary) = &args.asr_binary else {
        return Ok(json!({"status":"ENDPOINT_ONLY_NO_ASR_REQUESTED"}));
    };
    ensure!(
        args.asr_device == "cuda-cudnn" || args.asr_device == "cpu",
        "Unsupported ASR device"
    );
    let model = args
        .asr_model_dir
        .as_ref()
        .context("--asr-model-dir is required with --asr-binary")?;
    let output = folder.join("r9");
    let run = Command::new(binary)
        .env("HDB2C6_MASK_MODE", "both")
        .arg("--model-dir")
        .arg(model)
        .arg("--input")
        .arg(wav)
        .arg("--output")
        .arg(&output)
        .arg("--device")
        .arg(&args.asr_device)
        .output()?;
    fs::write(folder.join("asr.stdout"), &run.stdout)?;
    fs::write(folder.join("asr.stderr"), &run.stderr)?;
    ensure!(
        run.status.success(),
        "R9 transcription failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let mut result: Value = serde_json::from_slice(&fs::read(output.join("result.json"))?)?;
    let device = result["device"]
        .as_str()
        .context("R9 omitted device identity")?;
    if args.asr_device == "cuda-cudnn" {
        ensure!(device.starts_with("Cuda("), "R9 did not use CUDA");
    } else {
        ensure!(device == "Cpu", "R9 did not use CPU");
    }
    if !result["text"].is_string() {
        bail!("R9 omitted transcription");
    }
    result["execution"] = json!("SEPARATE_COLD_R9_PROCESS");
    Ok(result)
}
