//! Deployment image selection. Generic across CUDA architectures; the
//! attested catalogue itself supplies the *exact* target compute capability.
//!
//! One policy is read at CUDA-device construction. It must not be changed in
//! the middle of an inference or while a module/function cache is populated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DkrImageMode {
    /// Verified CUBIN first; PTX only after a CUDA module/function load error.
    Auto,
    /// Verified CUBIN; never fall back to PTX on driver/entrypoint errors.
    Cubin,
    /// Explicit rollback to the verified PTX from the same embedded catalogue.
    Ptx,
}

impl DkrImageMode {
    pub(super) fn parse(value: Option<&str>) -> std::result::Result<Self, &'static str> {
        match value {
            None | Some("auto") => Ok(Self::Auto),
            Some("cubin") => Ok(Self::Cubin),
            Some("ptx") => Ok(Self::Ptx),
            _ => Err("CANDLE_DKR_IMAGE_MODE must be auto, cubin or ptx"),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cubin => "cubin",
            Self::Ptx => "ptx",
        }
    }

    pub(super) fn from_environment() -> crate::Result<Self> {
        let setting = std::env::var_os("CANDLE_DKR_IMAGE_MODE");
        let parsed = setting.as_deref().map(|s| s.to_str().ok_or_else(|| {
            crate::Error::Msg("CANDLE_DKR_IMAGE_MODE must be UTF-8".into())
        })).transpose()?;
        Self::parse(parsed).map_err(|message| crate::Error::Msg(message.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::DkrImageMode as Mode;

    #[test]
    fn qualified_default_policy_has_explicit_rollback_only() {
        assert_eq!(Mode::parse(None), Ok(Mode::Auto));
        assert_eq!(Mode::parse(Some("auto")), Ok(Mode::Auto));
        assert_eq!(Mode::parse(Some("cubin")), Ok(Mode::Cubin));
        assert_eq!(Mode::parse(Some("ptx")), Ok(Mode::Ptx));
        for value in ["", "PTX", "legacy", "filesystem", "disable", "auto ", "none"] {
            assert!(Mode::parse(Some(value)).is_err(), "{value:?}");
        }
        assert_eq!(Mode::Auto.as_str(), "auto");
        assert_eq!(Mode::Cubin.as_str(), "cubin");
        assert_eq!(Mode::Ptx.as_str(), "ptx");
        println!("DKR_R125_P1_POLICY=PASS auto=cubin rollback=ptx invalid=reject");
    }
}
