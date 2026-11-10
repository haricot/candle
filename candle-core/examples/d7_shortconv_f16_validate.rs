fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any())]
    {
        let out =
            std::path::PathBuf::from(std::env::args().nth(1).ok_or("missing output directory")?);
        candle_core::stateful::validate_f16_candidate(&out)
    }
    Err("stateful candidate needs the reviewed Unified ASD V3 composition".into())
}
