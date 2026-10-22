use candle_voice_endpointing::runner::{self, Args};
use clap::Parser;
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    runner::run(args.clone(), |wav, folder| {
        runner::external_asr(&args, wav, folder)
    })?;
    Ok(())
}
