//! Sample-clock endpoint policy, ported from the previously exercised HDB1-R1.
use crate::{firered::FireRedVad, smart_turn::SmartTurn};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::VecDeque, time::Instant};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub threshold: f32,
    pub endpoint_hold_ms: usize,
    pub max_hesitation_ms: usize,
    pub max_turn_ms: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            endpoint_hold_ms: 1200,
            max_hesitation_ms: 3200,
            max_turn_ms: 25000,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.threshold.is_finite() && (0.0..=1.0).contains(&self.threshold),
            "Invalid Smart Turn threshold"
        );
        ensure!(
            self.endpoint_hold_ms > 0
                && self.endpoint_hold_ms < self.max_hesitation_ms
                && self.max_hesitation_ms < self.max_turn_ms
                && self.max_turn_ms <= 25000,
            "Invalid hold/hesitation/turn bounds (maximum turn: 25000 ms)"
        );
        Ok(())
    }
}

pub struct Policy {
    cfg: Config,
    pub started: Option<usize>,
    last_end: Option<usize>,
    last: usize,
    checked: bool,
    eligible: bool,
}
impl Policy {
    pub fn new(cfg: Config) -> Result<Self> {
        cfg.validate()?;
        Ok(Self {
            cfg,
            started: None,
            last_end: None,
            last: 0,
            checked: false,
            eligible: false,
        })
    }
    pub fn start(&mut self, sample: usize) -> bool {
        self.last = sample.max(self.last);
        let cancelled = self.last_end.is_some();
        if self.started.is_none() {
            self.started = Some(self.last);
        }
        self.last_end = None;
        self.checked = false;
        self.eligible = false;
        cancelled
    }
    pub fn end(&mut self, sample: usize) -> bool {
        self.last = sample.max(self.last);
        if let Some(start) = self.started {
            self.last_end = Some(self.last);
            self.checked = false;
            self.eligible = self.last - start >= 4800;
        }
        self.eligible
    }
    pub fn advance(&mut self, sample: usize) -> Option<&'static str> {
        self.last = sample.max(self.last);
        let start = self.started?;
        if self.last - start >= self.cfg.max_turn_ms * 16 {
            return Some("MAX_TURN_TIMEOUT");
        }
        if let Some(end) = self.last_end {
            let silent = self.last - end;
            if silent >= self.cfg.max_hesitation_ms * 16 {
                return Some("HESITATION_TIMEOUT");
            }
            if self.eligible && !self.checked && silent >= self.cfg.endpoint_hold_ms * 16 {
                return Some("SMART_TURN_CHECK");
            }
        }
        None
    }
    pub fn smart_result(&mut self, probability: f32) -> Result<bool> {
        ensure!(
            probability.is_finite() && (0.0..=1.0).contains(&probability),
            "Invalid Smart Turn result"
        );
        ensure!(
            self.started.is_some()
                && self.eligible
                && !self.checked
                && self
                    .last_end
                    .is_some_and(|end| self.last - end >= self.cfg.endpoint_hold_ms * 16),
            "Premature Smart Turn decision"
        );
        self.checked = true;
        Ok(probability >= self.cfg.threshold)
    }
    pub fn eof_reason(&self) -> Option<&'static str> {
        self.started.map(|_| {
            if self.last_end.is_some() {
                "EOF_FORCED_UNCONFIRMED"
            } else {
                "EOF_WHILE_SPEAKING"
            }
        })
    }
}

pub struct Turn {
    pub pcm: Vec<i16>,
    pub reason: String,
}
pub struct Pipeline {
    pub vad: FireRedVad,
    pub smart: SmartTurn,
    policy: Policy,
    pub cursor: usize,
    vad_base: usize,
    pre: VecDeque<i16>,
    pcm: Vec<i16>,
    pub events: Vec<Value>,
}
impl Pipeline {
    pub fn new(vad: FireRedVad, smart: SmartTurn, cfg: Config) -> Result<Self> {
        Ok(Self {
            vad,
            smart,
            policy: Policy::new(cfg)?,
            cursor: 0,
            vad_base: 0,
            pre: VecDeque::new(),
            pcm: Vec::new(),
            events: Vec::new(),
        })
    }
    fn log(&mut self, event: &str, fields: Value) {
        let mut row =
            json!({"event":event,"sample":self.cursor,"seconds":self.cursor as f64/16000.0});
        if let (Some(row), Some(fields)) = (row.as_object_mut(), fields.as_object()) {
            row.extend(fields.clone());
        }
        self.events.push(row);
    }
    pub fn feed(&mut self, chunk: &[i16]) -> Result<Option<Turn>> {
        ensure!(
            chunk.len() <= 1600,
            "Controller accepts at most 100 ms per call"
        );
        self.cursor += chunk.len();
        let active = self.policy.started.is_some();
        if active {
            self.pcm.extend_from_slice(chunk);
        }
        let (events, _) = self.vad.feed(chunk)?;
        for event in events {
            let point = (self.vad_base + event.sample)
                .min(self.cursor)
                .max(self.policy.last);
            if event.kind == "speech_start" {
                if self.policy.started.is_none() {
                    self.pcm.extend(self.pre.iter());
                    if !active {
                        self.pcm.extend_from_slice(chunk);
                    }
                }
                if self.policy.start(point) {
                    self.log(
                        "endpoint_candidate_cancelled",
                        json!({"reason":"SPEECH_RESUMED_DURING_HOLD","vad_frame_sample":self.vad_base+event.sample}),
                    );
                }
                self.log(
                    "speech_start",
                    json!({"vad_frame_sample":self.vad_base+event.sample}),
                );
            } else {
                let candidate = self.policy.end(point);
                self.log(
                    "speech_end",
                    json!({"candidate":candidate,"vad_frame_sample":self.vad_base+event.sample}),
                );
                if candidate {
                    self.log(
                        "endpoint_candidate_deferred",
                        json!({"hold_ms":self.policy.cfg.endpoint_hold_ms,"reason":"AWAIT_UNINTERRUPTED_SILENCE"}),
                    );
                }
            }
        }
        ensure!(
            self.pcm.len() <= 28000 * 16,
            "Turn exceeded bounded PCM capacity"
        );
        match self.policy.advance(self.cursor) {
            Some("SMART_TURN_CHECK") => {
                let start = Instant::now();
                let probability = self.smart.predict(&self.pcm)?;
                let complete = self.policy.smart_result(probability)?;
                self.log("smart_turn",json!({"probability":probability,"complete":complete,"latency_ms":start.elapsed().as_secs_f64()*1000.0,
                    "post_vad_silence_ms":(self.cursor-self.policy.last_end.unwrap()) as f64/16.0}));
                if complete {
                    return Ok(Some(self.finalize("SMART_TURN_CONFIRMED")?));
                }
            }
            Some(reason) => {
                self.log("forced_endpoint", json!({"reason":reason}));
                return Ok(Some(self.finalize(reason)?));
            }
            None => {}
        }
        self.pre.extend(chunk);
        while self.pre.len() > 4800 {
            self.pre.pop_front();
        }
        Ok(None)
    }
    fn finalize(&mut self, reason: &str) -> Result<Turn> {
        ensure!(!self.pcm.is_empty(), "Cannot finalize an empty turn");
        let pcm = std::mem::take(&mut self.pcm);
        self.pre.clear();
        self.vad.reset()?;
        self.vad_base = self.cursor;
        self.policy = Policy::new(self.policy.cfg.clone())?;
        Ok(Turn {
            pcm,
            reason: reason.into(),
        })
    }
    pub fn eof(&mut self) -> Result<Option<Turn>> {
        match self.policy.eof_reason() {
            Some(reason) => {
                self.log("forced_endpoint", json!({"reason":reason}));
                Ok(Some(self.finalize(reason)?))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resumption_cancels_candidate_and_preserves_the_same_turn() {
        let mut p = Policy::new(Config::default()).unwrap();
        assert!(!p.start(1600));
        assert!(p.end(16000));
        assert_eq!(p.advance(25000), None);
        assert!(p.start(26000));
        assert_eq!(p.started, Some(1600));
        assert_eq!(p.advance(40000), None);
        assert!(p.end(42000));
        assert_eq!(p.advance(61200), Some("SMART_TURN_CHECK"));
        assert!(p.smart_result(0.97).unwrap());
    }
    #[test]
    fn rejection_does_not_authorize_asr() {
        let mut p = Policy::new(Config::default()).unwrap();
        p.start(0);
        p.end(16000);
        assert!(p.smart_result(0.9).is_err());
        assert_eq!(p.advance(35200), Some("SMART_TURN_CHECK"));
        assert!(!p.smart_result(0.01).unwrap());
        assert_eq!(p.advance(40000), None);
        assert_eq!(p.advance(67200), Some("HESITATION_TIMEOUT"));
        assert_eq!(p.eof_reason(), Some("EOF_FORCED_UNCONFIRMED"));
    }
    #[test]
    fn invalid_configuration_is_rejected() {
        let cfg = Config {
            threshold: f32::NAN,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
        let cfg = Config {
            endpoint_hold_ms: 3200,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }
}
