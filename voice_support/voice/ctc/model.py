"""Parakeet's CTC log-probabilities from its transducer encoder and CTC head.

NeMo's hybrid TDT-CTC model shares one encoder between its two heads. The
CTC head is only a 1x1 convolution to the vocabulary and a log-softmax, so it
is kept beside the encoder (``ctc_head.npz``) and applied here: that gives the
CTC model's log-probabilities from the encoder alone, without the separately
exported CTC model loading the same encoder a second time. This is
adapt-ctc's ``hybrid.HybridModel`` without the transducer's decoder and
joiner, which only write transcripts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .acoustic import SAMPLE_RATE, MelFeaturizer, Vocabulary
from .runtime import session_options

ENCODER = "encoder.onnx"
CTC_HEAD = "ctc_head.npz"
TOKENS = "tokens.txt"


class CtcModel:
    """``vocab`` and ``log_probs`` for 16 kHz mono audio."""

    def __init__(self, model_dir: Path, num_threads: int = 1) -> None:
        import onnxruntime as ort  # pylint: disable=import-outside-toplevel

        model_dir = Path(model_dir)
        self._encoder = ort.InferenceSession(
            str(model_dir / ENCODER),
            sess_options=session_options(num_threads),
            providers=["CPUExecutionProvider"],
        )
        with np.load(model_dir / CTC_HEAD) as head:
            self._weight = np.ascontiguousarray(head["weight"], dtype=np.float32)
            self._bias = np.ascontiguousarray(head["bias"], dtype=np.float32)
        self.vocab = Vocabulary.from_tokens_txt(model_dir / TOKENS)
        self._featurizer = MelFeaturizer()

    def log_probs(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """``[T, V]`` natural-log probabilities and the seconds per output frame."""
        feats = self._featurizer(np.ascontiguousarray(audio, dtype=np.float32))
        encoded = self._encoder.run(
            None,
            {
                "audio_signal": feats[np.newaxis],
                "length": np.array([feats.shape[1]], dtype=np.int64),
            },
        )[0][0]
        hop = self._featurizer.hop_length / self._featurizer.sample_rate
        seconds = hop * max(1, round(feats.shape[1] / max(1, encoded.shape[1])))
        logits = (self._weight @ encoded + self._bias[:, np.newaxis]).T
        top = logits.max(axis=1, keepdims=True)
        log_probs = logits - top - np.log(np.exp(logits - top).sum(axis=1, keepdims=True))
        return np.ascontiguousarray(log_probs, dtype=np.float32), seconds

    def warm_up(self) -> None:
        """One decode, so that the first wake word does not pay for the arena."""
        self.log_probs(np.zeros(SAMPLE_RATE, dtype=np.float32))
