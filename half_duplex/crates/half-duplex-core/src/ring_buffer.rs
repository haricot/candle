use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use crate::audio::{AudioChunk, PcmAudio};

#[derive(Debug, PartialEq, Eq)]
pub enum RingError { InvalidCapacity, RateMismatch, NonContiguous, NotAvailable, Overflow }
impl fmt::Display for RingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl Error for RingError {}

/// Absolute-indexed ring buffer retaining the most recent mono PCM samples.
#[derive(Debug)]
pub struct PcmRingBuffer {
    capacity: usize,
    sample_rate: u32,
    first_sample: u64,
    next_sample: u64,
    initialized: bool,
    queue: VecDeque<f32>,
}
impl PcmRingBuffer {
    pub fn new(sample_rate: u32, capacity: usize) -> Result<Self, RingError> {
        if sample_rate == 0 || capacity == 0 { return Err(RingError::InvalidCapacity); }
        Ok(Self { capacity, sample_rate, first_sample: 0, next_sample: 0, initialized: false, queue: VecDeque::with_capacity(capacity.min(65_536)) })
    }
    pub fn append(&mut self, chunk: &AudioChunk) -> Result<(), RingError> {
        if chunk.sample_rate != self.sample_rate { return Err(RingError::RateMismatch); }
        let expected = if self.initialized { self.next_sample } else { chunk.start_sample };
        if chunk.start_sample != expected { return Err(RingError::NonContiguous); }
        let end_sample = chunk.start_sample.checked_add(chunk.samples.len() as u64).ok_or(RingError::Overflow)?;
        if !self.initialized { self.first_sample = chunk.start_sample; self.initialized = true; }
        self.queue.extend(chunk.samples.iter().copied());
        self.next_sample = end_sample;
        while self.queue.len() > self.capacity {
            self.queue.pop_front();
            self.first_sample += 1;
        }
        Ok(())
    }
    pub fn retained_range(&self) -> (u64, u64) { (self.first_sample, self.next_sample) }
    pub fn len(&self) -> usize { self.queue.len() }
    pub fn is_empty(&self) -> bool { self.queue.is_empty() }
    /// Inclusive start / exclusive end. Fails instead of silently dropping a turn's beginning.
    pub fn extract(&self, start: u64, end: u64) -> Result<PcmAudio, RingError> {
        if start < self.first_sample || end > self.next_sample || end <= start {
            return Err(RingError::NotAvailable);
        }
        let start_offset = usize::try_from(start - self.first_sample).map_err(|_| RingError::Overflow)?;
        let take = usize::try_from(end - start).map_err(|_| RingError::Overflow)?;
        let samples = self.queue.iter().skip(start_offset).take(take).copied().collect::<Vec<_>>();
        if samples.len() != take { return Err(RingError::NotAvailable); }
        PcmAudio::new(self.sample_rate, samples).map_err(|_| RingError::NotAvailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chunk(start: u64, samples: &[f32]) -> AudioChunk {
        AudioChunk::new(start, 16_000, samples.to_vec()).unwrap()
    }
    #[test]
    fn ring_keeps_newest_samples() {
        let mut r = PcmRingBuffer::new(16_000, 4).unwrap();
        r.append(&chunk(100, &[1., 2., 3.])).unwrap();
        r.append(&chunk(103, &[4., 5.])).unwrap();
        assert_eq!(r.retained_range(), (101, 105));
        assert_eq!(r.extract(102, 105).unwrap().samples(), &[3., 4., 5.]);
        assert!(matches!(r.extract(100, 105), Err(RingError::NotAvailable)));
    }
    #[test]
    fn discontinuity_is_rejected_without_change() {
        let mut r = PcmRingBuffer::new(16_000, 10).unwrap();
        r.append(&chunk(0, &[1., 2.])).unwrap();
        assert_eq!(r.append(&chunk(4, &[3.])), Err(RingError::NonContiguous));
        assert_eq!(r.retained_range(), (0, 2));
    }
    #[test]
    fn rate_mismatch_is_rejected() {
        let mut r = PcmRingBuffer::new(16_000, 10).unwrap();
        let c = AudioChunk::new(0, 48_000, vec![0.1]).unwrap();
        assert_eq!(r.append(&c), Err(RingError::RateMismatch));
    }
}
