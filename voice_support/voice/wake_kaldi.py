"""Verifying a wake word with the vendored Kaldi decoder.

The detector on the device is a small streaming classifier. This re-decodes
the window it fired on against explicit grammars -- the wake phrase, free
background speech, the phrase minus its last word, and the phrase's known
confusables -- and accepts only when the phrase beats background on its
prefix and on its last word, and no confusable beats it by much. Those are
three separate decisions, and they fail separately: "okay now" loses on the
last word, "okay naruto" loses on the confusable margin.

The same decode aligns the phrase, so it also says where the wake word ends.
That is a better boundary for speech-to-text than the detection itself,
which is only where a threshold was crossed; Parakeet's CTC head, once it is
loaded (``ctc_wake``), places it better still, and gives a second opinion on
a `prefix` or `last` refusal.

The decoding itself is vendored in ``kaldi/`` rather than depended on. What
this module adds is the app's shape around it: the graphs and the resident
decoders brought up once at startup, the retry, the second opinion and the
boundary.

A detector that fires before the speaker has finished leaves the last
phoneme outside the window, and the thresholds are then applied to a truncated phrase. When
the alignment puts the end of the phrase hard against the end of the window,
that is what happened, and the fix is to slide the window forward over audio
from after the detection. The experimental app measured this both ways
across four speakers: sliding always cost two of them (-1.4 and -2.8 points)
what it gained the one being truncated, while sliding only on a pinned split
gained +3.8 and +0.7 and took nothing from anyone. So the window grows only
when the alignment asks for it, and the app waits for that much more of the
run to arrive rather than asking Home Assistant for it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .kaldi import (
    ResidentDecoder,
    ServiceConfig,
    SharedDecoder,
    SpeechTools,
    Verifier as KaldiDecode,
    VerifyResult,
    load_config,
    maybe_start_decoder,
    maybe_start_shared_decoder,
    prepare_model,
    train_wake_word,
)
from .protocol import (
    SAMPLE_CHANNELS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    WakeWordInfo,
    bytes_to_ms,
    ms_to_bytes,
)
from .verify import (
    SPLIT_KALDI,
    SPLIT_NONE,
    SPLIT_PLACED,
    MoreAudio,
    Verdict,
    Verifier,
    VerifyRequest,
)

from .ctc_wake import KALDI_LEAD
from .wake_ratio import OVERRIDABLE

if TYPE_CHECKING:
    from .ctc_wake import CtcWake

_LOGGER = logging.getLogger(__name__)

TAIL_MS = 300
"""How far the window slides when the wake phrase ran off its end, and how
far past the detection the end of the phrase is looked for."""

EDGE_SECONDS = 0.03
"""The alignment is "at the end" of the window within this much of it."""

RATIO_WINDOW_MS = 1500
"""The second opinion's window, ending where Kaldi's did: the pre-roll it
was calibrated on."""

DECODES_KEPT = 16
"""Windows whose decodes are remembered, for a verifier asked again once more
audio has arrived."""

class KaldiVerifier(Verifier):
    """The vendored decoder, with one decode per configured wake word.

    Everything is brought up at startup -- the tools checked, the acoustic
    model prepared, a graph built or found in the cache for each wake word,
    and a resident decoder started for it -- so the first wake word of the
    day costs what the rest do, and ``/info`` is true as soon as it answers.
    """

    def __init__(
        self,
        config_path: Path,
        train_dir: Path | None = None,
        split: str = SPLIT_PLACED,
    ) -> None:
        self.config_path = Path(config_path)
        self.split = split
        self.config: ServiceConfig = load_config(self.config_path)
        if train_dir is not None:
            # Graphs are cached build output, not configuration: in the app
            # they belong in /data, which survives an update, and not in the
            # image beside the config that named them.
            self.config.train_dir = Path(train_dir)
        self._decodes: dict[str, KaldiDecode] = {}
        self._pools: list[ResidentDecoder] = []
        self._shared: list[SharedDecoder] = []
        self._decoding: set[asyncio.Task[VerifyResult]] = set()
        self._decoded: dict[tuple[str, bytes], VerifyResult] = {}
        self.ctc: CtcWake | None = None
        """Parakeet's CTC head, once it is downloaded and loaded."""

    @property
    def wake_words(self) -> list[WakeWordInfo]:
        return [
            WakeWordInfo(id=wake_id, phrase=wake.phrase_text)
            for wake_id, wake in self.config.wake_words.items()
        ]

    async def start(self) -> None:
        tools = SpeechTools.from_dir(self.config.tools_dir)
        models = {
            model_id: prepare_model(
                model_id, model_config.directory, self.config.train_dir
            )
            for model_id, model_config in self.config.models.items()
        }

        graph_dirs: dict[str, Path] = {}
        for wake_id, wake_word in self.config.wake_words.items():
            graph_dirs[wake_id] = await train_wake_word(
                wake_word, models[wake_word.model], self.config.train_dir, tools
            )

        # One decoder for every wake word, where the models allow it: the
        # acoustic model is nearly all of what a decoder costs resident, and
        # it is the same one for all of them. Wake words from different
        # models keep their own, since the model is what is being shared.
        shared: dict[str, SharedDecoder] = {}
        if self.config.resident_decoders:
            by_model: dict[str, dict[str, Path]] = {}
            for wake_id, wake_word in self.config.wake_words.items():
                by_model.setdefault(wake_word.model, {})[wake_id] = graph_dirs[
                    wake_id
                ]
            for model_id, dirs in by_model.items():
                pool = await maybe_start_shared_decoder(
                    None, models[model_id], dirs, tools, self.config.decoder
                )
                if pool is not None:
                    self._shared.append(pool)
                    shared.update(dict.fromkeys(dirs, pool))

        for wake_id, wake_word in self.config.wake_words.items():
            model = models[wake_word.model]
            graph_dir = graph_dirs[wake_id]
            pool: object | None = None
            if wake_id in shared:
                pool = shared[wake_id].view(wake_id)
            elif self.config.resident_decoders:
                # Without the daemon every decode spawns a Kaldi process
                # instead, which works and is slower: on one window of
                # "okay nabu", 42 ms against roughly 200 ms.
                pool = await maybe_start_decoder(
                    None, model, graph_dir, tools, self.config.decoder
                )
                if pool is not None:
                    self._pools.append(pool)
            self._decodes[wake_id] = KaldiDecode(
                wake_word, model, graph_dir, tools, pool, self.config.decoder
            )
            _LOGGER.debug("Ready to verify %s", wake_id)

    async def stop(self) -> None:
        pools, self._pools = self._pools, []
        shared, self._shared = self._shared, []
        self._decodes = {}
        await asyncio.gather(
            *(pool.stop() for pool in (*pools, *shared)), return_exceptions=True
        )

    async def verify(
        self, request: VerifyRequest, pcm: bytes, final: bool
    ) -> Verdict | MoreAudio:
        assert self._decodes, "start() first"
        wake = self.config.resolve(request.wake_word_phrase)
        if wake is None:
            # Home Assistant asked about a wake word /info never offered.
            return Verdict(accepted=True, reason="unknown_wake_word")

        start, end = self._window(request, len(pcm))
        tail = ms_to_bytes(TAIL_MS)
        result = await self._decode(wake.id, pcm[start:end])
        if result.error is not None:
            return self._unverified(result)
        offset = start
        if (not result.accepted) and self._ran_into_edge(result, end - start):
            # The phrase ended at the edge of the window, which means the
            # detector fired before the speaker had finished and the
            # thresholds were applied to a phrase that had not finished
            # inside it. Slide the window forward over audio from after the
            # detection.
            if len(pcm) < end + tail and not final:
                return MoreAudio(ms=TAIL_MS - bytes_to_ms(len(pcm) - end))
            # A satellite that stopped streaming is judged on what there is.
            if len(pcm) >= end + tail:
                retry = await self._decode(wake.id, pcm[start + tail : end + tail])
                if (retry.error is None) and (retry.accepted or (retry.split is not None)):
                    result, offset = retry, start + tail

        ctc = self.ctc if (self.ctc is not None and self.ctc.knows(wake.id)) else None
        second_opinion = (
            ctc is not None and (not result.accepted) and result.rejected_by in OVERRIDABLE
        )
        placing = (
            ctc is not None
            and self.split == SPLIT_PLACED
            and result.split is not None
            and (result.accepted or second_opinion)
        )
        if placing and len(pcm) < end + tail and not final:
            # Placement hears past the detection, as far as the retry does.
            return MoreAudio(ms=TAIL_MS - bytes_to_ms(len(pcm) - end))

        verdict = self._verdict(result, offset)
        if second_opinion:
            assert ctc is not None
            window_end = offset + (end - start)
            opinion = await ctc.second_opinion(
                wake.id,
                result.rejected_by,
                pcm[max(0, window_end - ms_to_bytes(RATIO_WINDOW_MS)) : window_end],
            )
            if opinion is not None:
                verdict.detail["second_opinion"] = {
                    "overturned": opinion.overturned,
                    "score": opinion.score,
                    "rival": opinion.rival,
                    "rival_margin": opinion.rival_margin,
                }
                if opinion.overturned:
                    verdict = replace(
                        verdict, accepted=True, reason=f"overturned_{result.rejected_by}"
                    )
        if verdict.accepted and result.split is not None:
            verdict = await self._boundary(
                verdict, wake.id, result, offset, pcm[: end + tail], ctc
            )
        return verdict

    # ------------------------------------------------------------------

    def _window(self, request: VerifyRequest, available: int) -> tuple[int, int]:
        """Which bytes of the window to decode.

        It ends at the detection, because that is where the wake word ends.
        It starts at most ``max_window_seconds`` earlier: decode time is
        roughly linear in window length, and the cap is also what the
        configured confusables were measured against. So what is dropped
        from a long pre-roll is its oldest audio, not the wake word.
        """
        end = min(
            ms_to_bytes(request.wake_word_audio_ms)
            if request.wake_word_audio_ms > 0
            else available,
            available,
        )
        return max(0, end - ms_to_bytes(self.config.max_window_seconds * 1000)), end

    async def _boundary(
        self,
        verdict: Verdict,
        wake_id: str,
        result: VerifyResult,
        offset: int,
        pcm: bytes,
        ctc: CtcWake | None,
    ) -> Verdict:
        """Where speech-to-text should start, for an accepted wake word."""
        if self.split == SPLIT_NONE:
            return verdict
        assert result.split is not None
        offset_s = bytes_to_ms(offset) / 1000
        kaldi_end = offset_s + result.split
        if self.split == SPLIT_KALDI:
            return replace(verdict, wake_word_audio_ms=round(kaldi_end * 1000))
        if ctc is None:
            # The CTC head is not there yet (or knows no such wake word).
            return replace(verdict, wake_word_audio_ms=round((kaldi_end + KALDI_LEAD) * 1000))

        wake_start = None if result.wake_start is None else offset_s + result.wake_start
        placed, spelled = await ctc.place(wake_id, pcm, wake_start, kaldi_end)
        verdict.detail["placed_ms"] = round(placed * 1000)
        verdict.detail["placed_as"] = spelled
        return replace(verdict, wake_word_audio_ms=round(placed * 1000))

    def attach_ctc(self, ctc: CtcWake) -> None:
        self.ctc = ctc

    async def _decode(self, wake_id: str, window: bytes) -> VerifyResult:
        # A verifier asked to wait for more audio is asked again with all of
        # it, and decodes the same windows: those are remembered.
        key = (wake_id, hashlib.blake2b(window, digest_size=16).digest())
        if (known := self._decoded.get(key)) is not None:
            return known
        # A resident decoder answers requests in the order they came, so one
        # given up halfway leaves its answer to be read by the next request as
        # its own, and deletes the file it was still to read. The run that
        # wanted it may be cancelled; the decode runs to its end regardless.
        task = asyncio.ensure_future(self._decode_window(wake_id, window))
        self._decoding.add(task)
        task.add_done_callback(self._decoding.discard)
        result = await asyncio.shield(task)
        if result.error is None:
            self._decoded[key] = result
            while len(self._decoded) > DECODES_KEPT:
                self._decoded.pop(next(iter(self._decoded)))
        return result

    async def _decode_window(self, wake_id: str, window: bytes) -> VerifyResult:
        # The decoder reads a WAV file, not a buffer.
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
            with wave.open(temp.name, "wb") as wav_file:
                wav_file.setnchannels(SAMPLE_CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(window)
            return await self._decodes[wake_id].verify(Path(temp.name))

    @staticmethod
    def _ran_into_edge(result: VerifyResult, window_bytes: int) -> bool:
        """Did the wake phrase end at the very end of the window?

        Then the aligner had nowhere else to put it, and the thresholds were
        applied to a phrase that had not finished inside the window.
        """
        if result.split is None:
            return False
        return result.split >= (window_bytes / (SAMPLE_RATE * SAMPLE_WIDTH)) - EDGE_SECONDS

    @staticmethod
    def _unverified(result: VerifyResult) -> Verdict:
        """A decode that failed says nothing about the audio, and refusing on
        its behalf would silence the satellite."""
        return Verdict(accepted=True, reason="error", detail=result.to_dict())

    def _verdict(self, result: VerifyResult, offset: int) -> Verdict:
        """The library's result, with its times moved back to the window's start."""
        split_ms = result.split_ms
        if split_ms is not None:
            split_ms += bytes_to_ms(offset)
        detail = result.to_dict()
        if offset:
            detail["offset_ms"] = bytes_to_ms(offset)
        if split_ms is not None:
            # Always reported, so the alignment can be compared against
            # whatever boundary is sent.
            detail["split_ms"] = split_ms
        return Verdict(
            accepted=result.accepted,
            reason=result.rejected_by,
            confusable=(result.confusable_words or None)
            if result.rejected_by == "confusable"
            else None,
            detail=detail,
        )
