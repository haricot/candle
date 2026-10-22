# Interfaces for HDB1/HDB2

Audio domain: f32 mono samples, 16_000 Hz for the synthetic `simulate-turn` action. Every `AudioChunk` carries a monotonic `start_sample: u64`. No wall-clock sample synchronization has been implemented.

Pipeline contract (HDB1):
1. `AudioCapture` emits PCM chunks to a bounded channel and ring buffer without blocking on ASR.
2. Streaming VAD emits speech onset and pause candidates; the VAD implementation must preserve its recurrent state and report all actual frames consumed.
3. At candidate pauses, Smart Turn reads the most recent context and accepts/denies end-of-turn. An incomplete decision resumes the same turn ID; no PCM samples are discarded.
4. On accepted endpoint, extract the full turn from ring and send `PcmAudio` to the selected `AsrBackend`; overrun is an error, never silently truncated.
5. On transcription success, emit `TranscriptReady`; on inference failure emit an explicit error event at HDB2. Do not synthesize placeholder text as if it were recognized speech.
6. After LLM/TTS response completion, resume capture into `Listening`. Expose barge-in as a future optional feature, not implicitly in HDB0.

`TurnController` is deliberately a synchronous, deterministic state machine; HDB1 owns async tasks, scheduling, cancellation, channel backpressure, and actual endpoint timing.

Backend contract (HDB2):

```rust
pub trait AsrBackend: Send {
    fn id(&self) -> &str;
    fn transcribe(&mut self, audio: &PcmAudio) -> Result<Transcript, AsrError>;
}
```

No compatibility assumptions about TranscribePsy's FastConformer dimensions, decoding layout, or NeMo token mappings are made by this scaffold. Confirm model layout from the actual QVAC checkpoint before writing Candle loaders.
