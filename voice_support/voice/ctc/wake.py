"""Where the wake phrase is in the CTC evidence.

Phrases are spelled as the model writes them. Parakeet has no word "Nabu", so
it writes "Naboo", "Nabu" or "now"; all are spellings of the wake word, not
confusables. Confusables should be words the model can write: near-homophone
nonsense ("okay napu") is just another spelling of the wake word to it.

Vendored from adapt-ctc's wake.py with the whole-call command recognizer left
out: this app uses it to place the end of the wake phrase, and the command is
heard by Home Assistant's speech-to-text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic import Vocabulary
from .graph import Meaning, PhraseGraph
from .spotter import _logsumexp, forward_numpy, relative_log_probs, viterbi
from .text import name_units

WAKE = "wake"
CONFUSABLE = "confusable"

LEAD = 0.5
"""Seconds kept before the wake phrase when decoding the rest of the call.
With less, the model loses its footing: from 0.1 s before "okay nabu" it
heard "Okay, nowab do you turn on" where the whole call reads "Okay, Nab
turn on"."""

SPLIT_SLACK = 2.0
"""Nats. The command starts after the latest wake placement, of any spelling,
within this much of the best one: "okay now" may fit best while "okay nabu",
nearly as good, also covers the "-bu" the model heard afterwards."""


@dataclass(frozen=True)
class WakeConfig:
    """Second-stage wake word verification on the call's first seconds."""

    phrases: tuple[str, ...]
    """Every way the model writes the wake word ("okay nabu", "okay naboo")."""

    confusables: tuple[str, ...] = ()
    """Phrases the device's detector also fires on; the wake word must beat them."""

    scoring: str = "sum"
    """``sum``: evidence summed over every alignment and spelling (the CTC
    forward score), with the wake word's spellings pooled. ``best``: the single
    best path, as the command recognizer scores phrases."""

    min_margin: float = -0.5
    """Reject when the strongest confusable has more evidence than the wake
    phrase by more than this many nats."""

    max_cost_per_token: float | None = None
    """Optionally also require the wake phrase's best path to fit this well."""


@dataclass
class WakeResult:
    accepted: bool
    reason: str = ""
    phrase: str | None = None
    cost: float = float("inf")
    """Total cost (nats) of the best wake phrase placement."""

    cost_per_token: float = float("inf")
    start: int = 0
    end: int = 0
    """Frames of the wake phrase (inclusive)."""

    confusable: str | None = None
    confusable_cost: float = float("inf")
    margin: float = float("inf")
    """Wake evidence minus the strongest confusable's (nats): negative when the
    confusable is better supported. Summed or best-path, per ``scoring``."""

    evidence: float | None = None
    """Summed log-evidence for the wake word, relative to the greedy path."""

    frame_seconds: float = 0.08


class WakeVerifier:
    def __init__(self, config: WakeConfig, vocab: Vocabulary, language: str = "en") -> None:
        self.config = config
        self.vocab = vocab
        graph = PhraseGraph(vocab)
        for kind, phrases in ((WAKE, config.phrases), (CONFUSABLE, config.confusables)):
            for phrase in phrases:
                for units in name_units(phrase, language):
                    graph.add(units, Meaning("keyword", phrase, name_type=kind), flexible=True)
        self.unspeakable = graph.unspeakable
        self.graph = graph.compile()
        kinds = [m[0].name_type for m in self.graph.meanings]
        self._wake = np.asarray([i for i, k in enumerate(kinds) if k == WAKE], dtype=np.int64)
        self._confusable = np.asarray([i for i, k in enumerate(kinds) if k == CONFUSABLE], dtype=np.int64)

    def verify_log_probs(
        self, log_probs: np.ndarray, frame_seconds: float, max_end: int | None = None
    ) -> WakeResult:
        """Verdict on log-probabilities whose first frames hold the wake word.
        ``max_end`` limits where the wake phrase may end."""
        config = self.config
        rel_tokens, rel_blank = relative_log_probs(log_probs, self.vocab)
        score, start, tokens = viterbi(self.graph, rel_tokens, rel_blank)
        if max_end is not None:
            score = score[: max_end + 1]
        penalty = np.where(np.isfinite(score), -score.astype(np.float64), np.inf)

        result = WakeResult(accepted=False, frame_seconds=frame_seconds)
        best = self._best(penalty, self._wake)
        if best is None:
            result.reason = "no wake phrase"
            return result
        end, column = best
        result.phrase = self.graph.meanings[column][0].phrase
        result.cost = float(penalty[end, column])
        result.cost_per_token = result.cost / max(1, int(tokens[end, column]))
        result.start, result.end = int(start[end, column]), end
        # Where the wake word ends, for the split: the latest good placement.
        block = penalty[:, self._wake]
        near = np.nonzero((block <= result.cost + SPLIT_SLACK).any(axis=1))[0]
        if near.size:
            result.end = max(end, int(near.max()))

        if config.scoring == "sum":
            self._sum_margin(result, rel_tokens, rel_blank, max_end)
        else:
            rival = self._best(penalty, self._confusable)
            if rival is not None:
                result.confusable = self.graph.meanings[rival[1]][0].phrase
                result.confusable_cost = float(penalty[rival])
                result.margin = result.confusable_cost - result.cost

        # A confusable that wins says more about a rejection than a poor fit.
        if result.margin < config.min_margin:
            result.reason = (
                f"'{result.confusable}' is better supported than the wake word by "
                f"{-result.margin:.2f} (limit {-config.min_margin:g})"
            )
        elif (
            config.max_cost_per_token is not None
            and result.cost_per_token > config.max_cost_per_token
        ):
            result.reason = (
                f"'{result.phrase}' fits poorly: {result.cost_per_token:.2f}/token "
                f"(limit {config.max_cost_per_token:g})"
            )
        else:
            result.accepted = True
        return result

    def _sum_margin(
        self,
        result: WakeResult,
        rel_tokens: np.ndarray,
        rel_blank: np.ndarray,
        max_end: int | None,
    ) -> None:
        """Margin on summed evidence: every alignment and spelling of the wake
        word pooled at each end frame, against each confusable's own sum."""
        evidence = forward_numpy(self.graph, rel_tokens, rel_blank)
        if max_end is not None:
            evidence = evidence[: max_end + 1]
        result.evidence = float(_logsumexp(evidence[:, self._wake], axis=1).max())
        if self._confusable.size:
            per_confusable = evidence[:, self._confusable].max(axis=0)
            strongest = int(per_confusable.argmax())
            result.confusable = self.graph.meanings[self._confusable[strongest]][0].phrase
            result.confusable_cost = -float(per_confusable[strongest])
            result.margin = result.evidence + result.confusable_cost

    @staticmethod
    def _best(penalty: np.ndarray, columns: np.ndarray) -> tuple[int, int] | None:
        """Placement with the lowest total cost among these phrases."""
        if columns.size == 0 or penalty.shape[0] == 0:
            return None
        block = penalty[:, columns]
        end, index = np.unravel_index(np.argmin(block), block.shape)
        if not np.isfinite(block[end, index]):
            return None
        return int(end), int(columns[index])
