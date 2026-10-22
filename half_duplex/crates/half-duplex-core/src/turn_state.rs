use crate::asr::Transcript;
use crate::events::{Event, Phase};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnError { InvalidTransition { from: Phase, action: &'static str }, NonMonotonicSample, EmptyTurn, TurnIdOverflow }
impl fmt::Display for TurnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl Error for TurnError {}

#[derive(Debug)]
pub struct TurnController {
    phase: Phase,
    turn_id: u64,
    active_start: Option<u64>,
    last_sample: u64,
}
impl Default for TurnController { fn default() -> Self { Self::new() } }
impl TurnController {
    pub fn new() -> Self { Self { phase: Phase::Listening, turn_id: 0, active_start: None, last_sample: 0 } }
    pub fn phase(&self) -> Phase { self.phase }
    pub fn turn_id(&self) -> u64 { self.turn_id }
    pub fn speech_started(&mut self, sample: u64) -> Result<Event, TurnError> {
        self.expect_phase(Phase::Listening, "speech_started")?;
        self.check_sample(sample)?;
        let new_id = self.turn_id.checked_add(1).ok_or(TurnError::TurnIdOverflow)?;
        self.turn_id = new_id;
        self.active_start = Some(sample);
        self.phase = Phase::Speech;
        self.last_sample = sample;
        Ok(Event::SpeechStarted { turn_id: new_id, sample })
    }
    pub fn pause_detected(&mut self, sample: u64) -> Result<Event, TurnError> {
        self.expect_phase(Phase::Speech, "pause_detected")?;
        self.check_sample(sample)?;
        if sample <= self.active_start.unwrap_or(sample) { return Err(TurnError::EmptyTurn); }
        self.phase = Phase::MaybeEnd;
        self.last_sample = sample;
        Ok(Event::PauseDetected { turn_id: self.turn_id, sample })
    }
    pub fn endpoint_decision(&mut self, completed: bool, sample: u64) -> Result<Event, TurnError> {
        self.expect_phase(Phase::MaybeEnd, "endpoint_decision")?;
        self.check_sample(sample)?;
        self.last_sample = sample;
        if completed {
            let start_sample = self.active_start.expect("active turn required in MAYBE_END");
            if sample <= start_sample { return Err(TurnError::EmptyTurn); }
            self.phase = Phase::Transcribing;
            Ok(Event::TurnCompleted { turn_id: self.turn_id, start_sample, end_sample: sample })
        } else {
            self.phase = Phase::Speech;
            Ok(Event::TurnContinued { turn_id: self.turn_id, sample })
        }
    }
    pub fn transcript_ready(&mut self, transcript: Transcript) -> Result<Event, TurnError> {
        self.expect_phase(Phase::Transcribing, "transcript_ready")?;
        self.phase = Phase::Responding;
        Ok(Event::TranscriptReady { turn_id: self.turn_id, text: transcript.text, language: transcript.language, backend_id: transcript.backend_id })
    }
    pub fn response_finished(&mut self) -> Result<Event, TurnError> {
        self.expect_phase(Phase::Responding, "response_finished")?;
        self.phase = Phase::Listening;
        self.active_start = None;
        Ok(Event::ResponseFinished { turn_id: self.turn_id })
    }
    /// Abort handling is explicit: do not silently mark an ASR failure as a valid transcript.
    pub fn abort_turn(&mut self) -> Result<(), TurnError> {
        if self.phase == Phase::Listening { return Err(TurnError::InvalidTransition { from: self.phase, action: "abort_turn" }); }
        self.phase = Phase::Listening;
        self.active_start = None;
        Ok(())
    }
    fn check_sample(&self, sample: u64) -> Result<(), TurnError> {
        if sample < self.last_sample { Err(TurnError::NonMonotonicSample) } else { Ok(()) }
    }
    fn expect_phase(&self, expected: Phase, action: &'static str) -> Result<(), TurnError> {
        if self.phase != expected { Err(TurnError::InvalidTransition { from: self.phase, action }) } else { Ok(()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn false_endpoint_retains_turn_and_true_one_finishes() {
        let mut t = TurnController::new();
        assert_eq!(t.speech_started(100).unwrap(), Event::SpeechStarted { turn_id: 1, sample: 100 });
        t.pause_detected(150).unwrap();
        assert_eq!(t.endpoint_decision(false, 180).unwrap(), Event::TurnContinued { turn_id: 1, sample: 180 });
        assert_eq!(t.phase(), Phase::Speech);
        t.pause_detected(200).unwrap();
        assert_eq!(t.endpoint_decision(true, 220).unwrap(), Event::TurnCompleted { turn_id: 1, start_sample: 100, end_sample: 220 });
        assert_eq!(t.phase(), Phase::Transcribing);
        t.transcript_ready(Transcript::new("SYNTHETIC", "fr", "mock-test")).unwrap();
        assert_eq!(t.phase(), Phase::Responding);
        t.response_finished().unwrap();
        assert_eq!(t.phase(), Phase::Listening);
        t.speech_started(300).unwrap();
        assert_eq!(t.turn_id(), 2);
    }
    #[test]
    fn invalid_transitions_and_time_regressions() {
        let mut t = TurnController::new();
        assert!(t.pause_detected(0).is_err());
        t.speech_started(50).unwrap();
        assert_eq!(t.pause_detected(40).unwrap_err(), TurnError::NonMonotonicSample);
        assert_eq!(t.pause_detected(50).unwrap_err(), TurnError::EmptyTurn);
        assert_eq!(t.phase(), Phase::Speech);
        assert!(t.response_finished().is_err());
        t.abort_turn().unwrap();
        assert_eq!(t.phase(), Phase::Listening);
    }
    #[test]
    fn rejected_transcript_cannot_change_state() {
        let mut t = TurnController::new();
        assert!(t.transcript_ready(Transcript::new("", "fr", "mock")).is_err());
        assert_eq!(t.phase(), Phase::Listening);
    }
}
