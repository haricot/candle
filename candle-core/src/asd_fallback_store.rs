//! ASD qualified fallback control plane.
//!
//! Fallback authority lives in the user ASD store, not in compiled Rust tables.
//! The file is read, parsed and validated only on the cold path. Successful,
//! unavailable and invalid states are all cached so steady-state execution never
//! polls the filesystem.

use crate::{Error, Result};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

const FALLBACK_SCHEMA: &str = "ASD-FALLBACKS-V1";
const FALLBACK_FILE_ENV: &str = "CANDLE_ASD_FALLBACKS";

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct QualifiedFallback {
    pub decision_id: String,
    pub decision_identity_sha256: String,
    pub rank: u32,
    pub provider: String,
    pub implementation_id: String,
    pub protocol: String,
    pub evidence_sha256: Vec<String>,
    pub required_cudnn_version_raw: Option<usize>,
    pub qualification: String,
    pub note: String,
}

#[derive(Debug, Deserialize)]
struct FallbackStore {
    schema: String,
    profile_id: String,
    target_gpu_uuid: String,
    fallbacks: Vec<QualifiedFallback>,
}

#[derive(Debug)]
enum CacheState {
    Unresolved,
    Unavailable,
    Loaded(Arc<FallbackStore>),
    Invalid(String),
}

static CACHE: OnceLock<RwLock<CacheState>> = OnceLock::new();

fn cache() -> &'static RwLock<CacheState> {
    CACHE.get_or_init(|| RwLock::new(CacheState::Unresolved))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn fallback_path() -> Option<PathBuf> {
    std::env::var_os(FALLBACK_FILE_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(candle_kernels::asd_paths::current_fallbacks_path)
}

fn validate_store(store: &FallbackStore, path: &std::path::Path) -> Result<()> {
    if store.schema != FALLBACK_SCHEMA {
        return Err(Error::Msg(format!(
            "unsupported ASD fallback schema in {}: {:?}",
            path.display(),
            store.schema
        )));
    }
    if store.profile_id.is_empty() || store.target_gpu_uuid.is_empty() {
        return Err(Error::Msg(format!(
            "ASD fallback store {} is missing profile/device identity",
            path.display()
        )));
    }

    let mut ranks = BTreeSet::new();
    for fallback in &store.fallbacks {
        if fallback.decision_id.is_empty()
            || !is_sha256(&fallback.decision_identity_sha256)
            || fallback.rank == 0
            || !matches!(fallback.provider.as_str(), "cudnn" | "native")
            || fallback.implementation_id.is_empty()
            || fallback.protocol.is_empty()
            || fallback.qualification.is_empty()
            || fallback.evidence_sha256.is_empty()
            || fallback
                .evidence_sha256
                .iter()
                .any(|evidence| !is_sha256(evidence))
        {
            return Err(Error::Msg(format!(
                "invalid ASD fallback entry for decision {:?} in {}",
                fallback.decision_id,
                path.display()
            )));
        }
        if !ranks.insert((
            fallback.decision_id.as_str(),
            fallback.decision_identity_sha256.as_str(),
            fallback.rank,
        )) {
            return Err(Error::Msg(format!(
                "duplicate ASD fallback rank {} for decision {} in {}",
                fallback.rank,
                fallback.decision_id,
                path.display()
            )));
        }
    }
    Ok(())
}

fn load_store() -> Result<Option<Arc<FallbackStore>>> {
    let Some(path) = fallback_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let source = std::fs::read_to_string(&path).map_err(|err| {
        Error::Msg(format!(
            "failed to read ASD fallback store {}: {err}",
            path.display()
        ))
    })?;
    let store: FallbackStore = serde_json::from_str(&source).map_err(|err| {
        Error::Msg(format!(
            "failed to parse ASD fallback store {}: {err}",
            path.display()
        ))
    })?;
    validate_store(&store, &path)?;
    Ok(Some(Arc::new(store)))
}

fn resolved_store() -> Result<Option<Arc<FallbackStore>>> {
    {
        let state = cache().read().unwrap();
        match &*state {
            CacheState::Loaded(store) => return Ok(Some(store.clone())),
            CacheState::Unavailable => return Ok(None),
            CacheState::Invalid(message) => return Err(Error::Msg(message.clone())),
            CacheState::Unresolved => {}
        }
    }

    let mut state = cache().write().unwrap();
    match &*state {
        CacheState::Loaded(store) => return Ok(Some(store.clone())),
        CacheState::Unavailable => return Ok(None),
        CacheState::Invalid(message) => return Err(Error::Msg(message.clone())),
        CacheState::Unresolved => {}
    }

    match load_store() {
        Ok(Some(store)) => {
            *state = CacheState::Loaded(store.clone());
            Ok(Some(store))
        }
        Ok(None) => {
            *state = CacheState::Unavailable;
            Ok(None)
        }
        Err(err) => {
            let message = err.to_string();
            *state = CacheState::Invalid(message.clone());
            Err(Error::Msg(message))
        }
    }
}

#[allow(dead_code)]
pub(crate) fn refresh() {
    *cache().write().unwrap() = CacheState::Unresolved;
}

pub(crate) fn qualified_fallbacks_for_decision(
    decision_id: &str,
    decision_identity_sha256: &str,
    profile_id: &str,
    actual_gpu_uuid: &str,
) -> Result<Vec<QualifiedFallback>> {
    let Some(store) = resolved_store()? else {
        return Ok(Vec::new());
    };

    if store.profile_id != profile_id {
        return Err(Error::Msg(format!(
            "ASD fallback profile mismatch: local={} runtime={profile_id}",
            store.profile_id
        )));
    }
    if !actual_gpu_uuid.starts_with("GPU-") {
        return Err(Error::Msg(format!(
            "ASD fallback lookup received invalid runtime GPU UUID {actual_gpu_uuid:?}"
        )));
    }
    if store.target_gpu_uuid != actual_gpu_uuid {
        return Err(Error::Msg(format!(
            "ASD fallback GPU UUID mismatch: local={} runtime={actual_gpu_uuid}",
            store.target_gpu_uuid
        )));
    }

    let candidates = store
        .fallbacks
        .iter()
        .filter(|fallback| fallback.decision_id == decision_id)
        .collect::<Vec<_>>();

    if candidates
        .iter()
        .any(|fallback| fallback.decision_identity_sha256 != decision_identity_sha256)
    {
        return Err(Error::Msg(format!(
            "ASD fallback decision identity mismatch for {decision_id}: runtime={decision_identity_sha256}; refresh/sync the fallback store"
        )));
    }

    let mut matched = candidates.into_iter().cloned().collect::<Vec<_>>();
    matched.sort_by_key(|fallback| fallback.rank);
    Ok(matched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha_validation_is_strict() {
        assert!(is_sha256(&"a".repeat(64)));
        assert!(!is_sha256(&"g".repeat(64)));
        assert!(!is_sha256(&"a".repeat(63)));
    }
}
