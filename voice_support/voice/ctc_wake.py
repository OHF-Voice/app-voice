"""What Parakeet's CTC head adds to the Kaldi verifier.

Two things, both from voice-incubator, where they run on every call:

* **a second opinion** on a wake word Kaldi refused on its `prefix` or `last`
  test (``wake_ratio``): those ask whether the window is the wake word or
  just speech, which a speech model answers well and Kaldi's thresholds
  answer badly for some voices. A `confusable` refusal stands.
* **where the wake phrase ends**, found again in the CTC evidence, so that
  speech-to-text starts where the command does.

Measured on the office-multicap corpus (four Voice PEs, six speakers, the
detector at 0.40, 1.5 s of pre-roll), Kaldi alone verified 69% of the wake
words the detector caught, a child's 25%; with the second opinion at 1.2,
91% (88%). Of near misses the detector fired on ("okay natalie", "okay
now", in a room, synthesized), Kaldi accepted 10.1% and the two together
21.9%; of 1.5 s windows of overheard speech and television, 0.19% and 0.29%.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ctc.model import CtcModel
from .ctc.runtime import release_memory
from .ctc.wake import LEAD, WakeConfig, WakeVerifier
from .wake_ratio import RatioVerdict, WakeRatio

_LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WakeWordCtc:
    """What the CTC head knows of one wake word."""

    spellings: tuple[str, ...]
    """How Parakeet writes it, for finding where it ends. Placement only --
    Kaldi has decided it was the wake word -- so these are the readings it
    prefers, not confusables. It has no word "nabu"."""

    partial: tuple[str, ...] = ()
    """Spellings that may be only the start of the wake word's last word
    ("okay now" of "okay na|bu"): there the end is the later of where they
    end and where Kaldi ended the phrase."""

    threshold: float | None = None
    """The second opinion's (``wake_ratio``); None for none."""

    rivals: tuple[str, ...] = field(default_factory=tuple)
    """Near misses that keep a rejection the score alone would overturn."""


WAKE_WORDS: dict[str, WakeWordCtc] = {
    "okay_nabu": WakeWordCtc(
        spellings=("okay nabu", "okay naboo", "ok nabu", "ok naboo", "okay now"),
        partial=("okay now",),
        # 1.3 in voice-incubator. Here 1.2 kept 91.4% of wake words against
        # 92.1%, and let through 21.6% of near misses against 24.7% -- 5.8% of
        # "okay now" against 9.5% (with SciPy's FFT, as voice-incubator has it).
        threshold=1.2,
        # The verifier's confusables (wake.yaml) less the 25 that beat "okay
        # nabu" by more than 0.3 per frame on over 1% of the real wake words
        # the score overturns ("now", "new", "near", "night", "boo", "maybe").
        rivals=tuple(
            f"okay {word}"
            for word in (
                "bamboo", "dabu", "knobby", "labu", "mabu", "naba", "nabil",
                "nabo", "nadia", "nadu", "nagu", "nahbu", "nalu", "namu",
                "nancy", "nanu", "naomi", "napkin", "napu", "naru", "naruto",
                "nasu", "natalie", "natasha", "nature", "navu", "nebula",
                "neighbor", "nemo", "nina", "nolan", "noodle", "north", "nova",
                "nozzle", "nugget", "taboo",
            )
        ),
    ),
    # voice-incubator's, not measured here.
    "hey_jarvis": WakeWordCtc(
        spellings=("hey jarvis", "hi jarvis", "hey jervis", "a jarvis"),
        threshold=1.0,
    ),
    "hey_mycroft": WakeWordCtc(
        spellings=(
            "hey mycroft", "hey my craft", "hey microsoft", "hi mycroft",
            "hey my croft",
        ),
        threshold=1.2,
    ),
}

PLACE_AFTER_SECONDS = 0.3
"""Audio past the detection that placement is given. With none, the phrase
cannot be found ending after the detection, which it often does; past
0.3 s, nothing changed."""

SPLIT_BOUND = 0.3
"""Seconds past Kaldi's end of the phrase that the CTC head may still end it."""

KALDI_LEAD = 0.02
"""Seconds added to Kaldi's end when the CTC head does not end the phrase
later. Speech-to-text started cold there lost the command's first word less
often than at the detection, and gained a stray word less often than at
Kaldi's end itself."""


def _released[T](function: Callable[..., T], *args: Any) -> T:
    """``function``, then the encoder's activations handed back to the system:
    glibc would otherwise keep them, about 60 MB."""
    try:
        return function(*args)
    finally:
        release_memory()


def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class CtcWake:
    """The CTC model and, per wake word, its second opinion and placer.

    Every use of the model runs in a thread, one at a time: it is a 110M
    parameter encoder, and the event loop is serving satellites.
    """

    def __init__(self, model: CtcModel, wake_words: dict[str, WakeWordCtc]) -> None:
        self.model = model
        self.wake_words = wake_words
        self._ratios: dict[str, WakeRatio] = {}
        self._placers: dict[str, WakeVerifier] = {}
        for wake_id, wake in wake_words.items():
            if wake.threshold is not None:
                ratio = WakeRatio(wake.spellings[0], threshold=wake.threshold, rivals=wake.rivals)
                ratio.train(model)
                self._ratios[wake_id] = ratio
            self._placers[wake_id] = WakeVerifier(
                WakeConfig(wake.spellings, scoring="best"), model.vocab, "en"
            )
        self._lock = asyncio.Lock()

    @classmethod
    def load(cls, model_dir: Path, wake_ids: list[str], threads: int = 2) -> CtcWake:
        """Load the model and build the graphs; blocking, a second or two."""
        model = CtcModel(model_dir, num_threads=threads)
        model.warm_up()
        release_memory()
        return cls(model, {w: WAKE_WORDS[w] for w in wake_ids if w in WAKE_WORDS})

    def knows(self, wake_id: str) -> bool:
        return wake_id in self._placers

    async def second_opinion(self, wake_id: str, rejected_by: str | None, pcm: bytes) -> RatioVerdict | None:
        """Whether a Kaldi rejection is overturned; None when it may not be."""
        ratio = self._ratios.get(wake_id)
        if ratio is None:
            return None
        async with self._lock:
            return await asyncio.to_thread(_released, ratio.judge, rejected_by, pcm)

    async def place(
        self, wake_id: str, pcm: bytes, wake_start: float | None, kaldi_end: float
    ) -> tuple[float, str | None]:
        """Where the command starts, in seconds from the start of ``pcm``, and
        the spelling the wake phrase was found as (None: Kaldi's end).

        ``pcm`` should run ``PLACE_AFTER_SECONDS`` past the detection.
        voice-incubator's placement (engine.py ``_place``), then never before
        Kaldi's end plus ``KALDI_LEAD``.
        """
        placer = self._placers[wake_id]
        wake = self.wake_words[wake_id]
        async with self._lock:
            return await asyncio.to_thread(
                _released, self._place, placer, wake, pcm, wake_start, kaldi_end
            )

    def _place(
        self,
        placer: WakeVerifier,
        wake: WakeWordCtc,
        pcm: bytes,
        wake_start: float | None,
        kaldi_end: float,
    ) -> tuple[float, str | None]:
        anchor = wake_start if wake_start is not None else max(0.0, kaldi_end - 1.0)
        offset = max(0.0, anchor - LEAD)
        log_probs, frame_seconds = self.model.log_probs(to_float(pcm)[int(offset * SAMPLE_RATE) :])
        bound = int((kaldi_end - offset + SPLIT_BOUND) / frame_seconds)
        found = placer.verify_log_probs(log_probs, frame_seconds, max_end=bound)
        end = int((kaldi_end - offset) / frame_seconds) - 1
        if found.phrase is not None:
            end = (
                max(found.end, end) if found.phrase in wake.partial else found.end
            )
        placed = offset + (end + 1) * frame_seconds
        return max(placed, kaldi_end + KALDI_LEAD), found.phrase
