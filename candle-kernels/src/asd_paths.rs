//! User-scoped ASD data paths.
//!
//! Runtime artifacts and profiles live outside the Candle source tree so ASD V3
//! can evolve specialized implementations without requiring a Candle rebuild.
//! CANDLE_ASD_HOME overrides the XDG/default location.

use std::path::PathBuf;

pub const ASD_HOME_ENV: &str = "CANDLE_ASD_HOME";

pub fn user_data_home() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os(ASD_HOME_ENV).filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(root));
    }
    if let Some(root) = std::env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(root).join("asd"));
    }
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(|home| PathBuf::from(home).join(".local/share/asd"))
}

pub fn artifacts_dir(architecture: &str) -> Option<PathBuf> {
    user_data_home().map(|root| root.join("artifacts").join(architecture))
}

pub fn profiles_dir() -> Option<PathBuf> {
    user_data_home().map(|root| root.join("profiles"))
}

pub fn current_profile_path() -> Option<PathBuf> {
    profiles_dir().map(|root| root.join("current.asd"))
}

pub fn profile_extensions_path() -> Option<PathBuf> {
    profiles_dir().map(|root| root.join("extensions.asd"))
}

pub fn fallbacks_dir() -> Option<PathBuf> {
    user_data_home().map(|root| root.join("fallbacks"))
}

pub fn current_fallbacks_path() -> Option<PathBuf> {
    fallbacks_dir().map(|root| root.join("current.json"))
}

pub fn history_dir() -> Option<PathBuf> {
    user_data_home().map(|root| root.join("history"))
}

pub fn current_history_path() -> Option<PathBuf> {
    history_dir().map(|root| root.join("current.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_home_has_expected_layout() {
        let root = PathBuf::from("/tmp/asd-test-home");
        assert_eq!(
            root.join("artifacts/sm61"),
            root.join("artifacts").join("sm61")
        );
        assert_eq!(root.join("profiles/current.asd"), root.join("profiles").join("current.asd"));
        assert_eq!(root.join("profiles/extensions.asd"), root.join("profiles").join("extensions.asd"));
        assert_eq!(root.join("fallbacks/current.json"), root.join("fallbacks").join("current.json"));
        assert_eq!(root.join("history/current.json"), root.join("history").join("current.json"));
    }
}
