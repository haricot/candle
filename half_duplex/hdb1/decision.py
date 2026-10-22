"""HDB1-R1 endpoint policy: cancellable pause before Smart Turn inference.

No voice/ASR model output is simulated by this module.
All timing is in 16 kHz PCM samples, not wall clock.
"""
from dataclasses import dataclass
from typing import Optional

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Config:
    threshold: float = 0.50
    min_turn_ms: int = 300
    # Allow normal intra-sentence pauses; only test the semantic endpoint
    # after this much uninterrupted *post-VAD* silence.
    endpoint_hold_ms: int = 1200
    max_hesitation_ms: int = 3200
    max_turn_ms: int = 25000
    pre_roll_ms: int = 300
    max_pcm_ms: int = 28000

    def validate(self):
        if not (0 <= self.threshold <= 1): raise ValueError('threshold out of range')
        if not (0 < self.min_turn_ms < self.max_turn_ms <= self.max_pcm_ms):
            raise ValueError('invalid turn bounds')
        if not (0 < self.endpoint_hold_ms < self.max_hesitation_ms < self.max_turn_ms):
            raise ValueError('invalid hold/hesitation bound')
        if not (0 <= self.pre_roll_ms < self.max_pcm_ms):
            raise ValueError('invalid pre-roll')


class EndpointPolicy:
    """Turn transitions only; Smart Turn is called by the adapter when due.

    A FireRed speech_end is a candidate. The hold is cancellable if FireRed
    reports speech_start before it expires. Smart Turn runs only after a
    continuous post-VAD pause. A declined Smart Turn cannot authorize ASR;
    a separately-labelled hesitation timeout may force an endpoint later.
    """
    def __init__(self, config: Config = Config()):
        config.validate()
        self.cfg = config
        self.started: Optional[int] = None
        self.last_vad_end: Optional[int] = None
        self.pending = False
        self.eligible = False
        self.smart_checked = False
        self.finished = False
        self.last_sample = 0

    def _tick(self, sample: int):
        if sample < self.last_sample: raise ValueError('sample regression')
        self.last_sample = sample

    def advance(self, sample: int) -> Optional[str]:
        self._tick(sample)
        if self.finished or self.started is None: return None
        if sample - self.started >= self.cfg.max_turn_ms * 16:
            self.finished = True
            return 'MAX_TURN_TIMEOUT'
        if self.pending and self.last_vad_end is not None:
            silent = sample - self.last_vad_end
            if silent >= self.cfg.max_hesitation_ms * 16:
                self.finished = True
                return 'HESITATION_TIMEOUT'
            if (self.eligible and not self.smart_checked
                    and silent >= self.cfg.endpoint_hold_ms * 16):
                return 'SMART_TURN_CHECK'
        return None

    def speech_start(self, sample: int) -> bool:
        """Return whether a candidate endpoint was cancelled by resumed speech.

        This must NOT call advance(): FireRed reports starts with delay and a
        possible new onset has priority over a pending endpoint timeout.
        """
        self._tick(sample)
        if self.finished: return False
        cancelled = self.pending
        if self.started is None: self.started = sample
        self.last_vad_end = None
        self.pending = False
        self.eligible = False
        self.smart_checked = False
        return cancelled

    def speech_end(self, sample: int) -> bool:
        self._tick(sample)
        if self.finished or self.started is None: return False
        self.last_vad_end = sample
        self.pending = True
        self.eligible = sample - self.started >= self.cfg.min_turn_ms * 16
        self.smart_checked = False
        return self.eligible

    def smart_result(self, probability: float) -> bool:
        if self.finished or not self.pending or not self.eligible or self.smart_checked:
            raise ValueError('no unchecked pending candidate')
        if self.last_vad_end is None or self.last_sample - self.last_vad_end < self.cfg.endpoint_hold_ms * 16:
            raise ValueError('Smart Turn invoked before endpoint hold')
        if not (0 <= probability <= 1): raise ValueError('invalid probability')
        self.smart_checked = True
        if probability >= self.cfg.threshold:
            self.finished = True
            return True
        return False

    def eof(self) -> Optional[str]:
        if self.finished or self.started is None: return None
        self.finished = True
        return 'EOF_FORCED_UNCONFIRMED' if self.pending else 'EOF_WHILE_SPEAKING'

    def reset(self):
        self.__init__(self.cfg)
