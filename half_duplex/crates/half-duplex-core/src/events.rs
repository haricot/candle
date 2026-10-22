#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase { Listening, Speech, MaybeEnd, Transcribing, Responding }
impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Listening => "LISTENING",
            Self::Speech => "SPEECH",
            Self::MaybeEnd => "MAYBE_END",
            Self::Transcribing => "TRANSCRIBING",
            Self::Responding => "RESPONDING",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    SpeechStarted { turn_id: u64, sample: u64 },
    PauseDetected { turn_id: u64, sample: u64 },
    TurnContinued { turn_id: u64, sample: u64 },
    TurnCompleted { turn_id: u64, start_sample: u64, end_sample: u64 },
    TranscriptReady { turn_id: u64, text: String, backend_id: String, language: String },
    ResponseFinished { turn_id: u64 },
}
impl Event {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SpeechStarted { .. } => "speech_started",
            Self::PauseDetected { .. } => "pause_detected",
            Self::TurnContinued { .. } => "turn_continued",
            Self::TurnCompleted { .. } => "turn_completed",
            Self::TranscriptReady { .. } => "transcript_ready",
            Self::ResponseFinished { .. } => "response_finished",
        }
    }
}
