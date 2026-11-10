fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(feature = "cuda", feature = "d7-trace"))]
    {
        let out =
            std::path::PathBuf::from(std::env::args().nth(1).ok_or("missing output directory")?);
        candle_core::stateful::validate_f16_candidate(&out)
    }
    #[cfg(not(all(feature = "cuda", feature = "d7-trace")))]
    Err("candidate validation requires cuda and d7-trace".into())
}
