"""What the Kaldi backend decides around the decoder.

What the four decodes mean is the vendored code's business and is tested
where it came from. What is tested here is what this app does with them:
when the alignment puts the end of the wake phrase against the end of the
window, the detector fired before the speaker finished, and the window has
to slide rather than the phrase be judged truncated; a `prefix` or `last`
refusal goes to the CTC head for a second opinion; and an accepted wake word
says where speech-to-text should start.

The decodes and the CTC head are scripted, so this runs without the 255 MB
of Kaldi binaries and model data that staging brings in, or the CTC model.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice.protocol import ms_to_bytes  # noqa: E402
from voice.verify import (  # noqa: E402
    SPLIT_KALDI,
    SPLIT_NONE,
    SPLIT_PLACED,
    MoreAudio,
    Verdict,
    VerifyRequest,
)
from voice.wake_kaldi import RATIO_WINDOW_MS, TAIL_MS, KaldiVerifier  # noqa: E402
from voice.wake_ratio import RatioVerdict  # noqa: E402

PREROLL_MS = 1000
PREROLL = ms_to_bytes(PREROLL_MS)
TAIL = ms_to_bytes(TAIL_MS)
REQUEST = VerifyRequest(wake_word_phrase="okay nabu", wake_word_audio_ms=PREROLL_MS)


def result(
    accepted, split=None, rejected_by=None, error=None, wake_start=None, confusable_words=""
):
    """One of the library's verdicts, as far as this module reads it."""
    return SimpleNamespace(
        accepted=accepted,
        rejected_by=rejected_by,
        confusable_words=confusable_words,
        split=split,
        split_ms=None if split is None else int(split * 1000),
        wake_start=wake_start,
        error=error,
        to_dict=lambda: {"accepted": accepted, "rejected_by": rejected_by},
    )


class Verifier(KaldiVerifier):
    """A KaldiVerifier whose decodes are scripted."""

    def __init__(self, results, split=SPLIT_KALDI, ctc=None):
        self._results = list(results)
        self.split = split
        self.ctc = ctc
        self.windows = []
        self.decoded = []
        self._decodes = {"okay_nabu": None}
        self._decoded = {}
        self._decoding = set()
        self._pools = []
        self.config = SimpleNamespace(
            max_window_seconds=1.4,
            wake_words={},
            resolve=lambda name: SimpleNamespace(id="okay_nabu"),
        )

    async def _decode(self, wake_id, window):
        self.windows.append(len(window))
        self.decoded.append(window)
        return self._results.pop(0)


async def verify(verifier, audio_ms, final=False):
    return await verifier.verify(REQUEST, bytes(ms_to_bytes(audio_ms)), final)


@pytest.mark.asyncio
async def test_accepted_at_once():
    verifier = Verifier([result(True, split=0.5)])
    answer = await verify(verifier, PREROLL_MS)
    assert isinstance(answer, Verdict)
    assert answer.accepted
    assert answer.wake_word_audio_ms == 500
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_rejection_that_did_not_run_off_the_end_stands():
    verifier = Verifier(
        [result(False, split=0.5, rejected_by="confusable", confusable_words="naruto")]
    )
    answer = await verify(verifier, PREROLL_MS)
    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert answer.reason == "confusable"
    assert answer.confusable == "naruto"
    # No retry: the phrase finished inside the window.
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_rejection_with_no_alignment_stands():
    verifier = Verifier([result(False, split=None, rejected_by="prefix")])
    answer = await verify(verifier, PREROLL_MS)
    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_phrase_against_the_edge_asks_for_more():
    verifier = Verifier([result(False, split=1.0, rejected_by="last")])
    answer = await verify(verifier, PREROLL_MS)
    assert answer == MoreAudio(ms=TAIL_MS)


@pytest.mark.asyncio
async def test_the_window_slides_once_the_audio_is_there():
    verifier = Verifier(
        [result(False, split=1.0, rejected_by="last"), result(True, split=0.8)]
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert isinstance(answer, Verdict)
    assert answer.accepted
    # Same length of window, started a tail later, and the split with it
    assert verifier.windows == [PREROLL, PREROLL]
    assert answer.wake_word_audio_ms == 800 + TAIL_MS
    assert answer.detail["offset_ms"] == TAIL_MS


@pytest.mark.asyncio
async def test_the_first_rejection_stands_if_the_retry_also_fails():
    verifier = Verifier(
        [
            result(False, split=1.0, rejected_by="last"),
            result(False, split=None, rejected_by="prefix"),
        ]
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert answer.reason == "last"


@pytest.mark.asyncio
async def test_audio_that_is_not_coming_is_not_waited_for():
    verifier = Verifier([result(False, split=1.0, rejected_by="last")])
    answer = await verify(verifier, PREROLL_MS, final=True)

    assert isinstance(answer, Verdict)
    assert not answer.accepted


@pytest.mark.asyncio
async def test_the_window_is_capped_and_ends_at_the_detection():
    verifier = Verifier([result(True, split=1.2)])
    answer = await verifier.verify(
        VerifyRequest(wake_word_phrase="okay nabu", wake_word_audio_ms=3000),
        # Count the samples, so the window's position shows in them.
        b"".join(i.to_bytes(2, "little") for i in range(ms_to_bytes(3000) // 2)),
        False,
    )

    assert verifier.windows == [ms_to_bytes(1400)]
    # It is the last 1.4 s of the pre-roll, not the first
    assert verifier.decoded[0][:2] == (ms_to_bytes(1600) // 2).to_bytes(2, "little")
    # ...and the split it reports is still measured from the audio's start
    assert isinstance(answer, Verdict)
    assert answer.wake_word_audio_ms == 1200 + 1600


@pytest.mark.asyncio
async def test_a_wake_word_it_was_never_offered():
    verifier = Verifier([])
    verifier.config.resolve = lambda name: None
    answer = await verify(verifier, PREROLL_MS)

    assert isinstance(answer, Verdict)
    assert answer.accepted
    assert answer.reason == "unknown_wake_word"


@pytest.mark.asyncio
async def test_no_split_keeps_the_detection():
    """The alignment is reported, but the pipeline keeps its own boundary."""
    verifier = Verifier([result(True, split=0.5)], split=SPLIT_NONE)
    answer = await verify(verifier, PREROLL_MS)
    assert answer.accepted
    assert answer.wake_word_audio_ms is None
    assert answer.detail["split_ms"] == 500


@pytest.mark.asyncio
async def test_placed_without_the_ctc_head_is_just_after_kaldi():
    verifier = Verifier([result(True, split=0.5)], split=SPLIT_PLACED)
    answer = await verify(verifier, PREROLL_MS)
    assert answer.accepted
    assert answer.wake_word_audio_ms == 520


@pytest.mark.asyncio
async def test_a_refusal_never_offers_a_split():
    """A wake word that was refused has no boundary worth sending."""
    verifier = Verifier([result(False, split=0.5, rejected_by="last")], split=SPLIT_KALDI)
    answer = await verify(verifier, PREROLL_MS)
    assert not answer.accepted
    assert answer.wake_word_audio_ms is None
    assert answer.detail["split_ms"] == 500


@pytest.mark.asyncio
async def test_a_failed_decode_is_not_a_refusal():
    verifier = Verifier([result(False, rejected_by="error", error="decoder: ERR")])
    answer = await verify(verifier, PREROLL_MS)

    assert isinstance(answer, Verdict)
    assert answer.accepted
    assert answer.reason == "error"
    assert answer.wake_word_audio_ms is None


@pytest.mark.asyncio
async def test_a_failed_retry_leaves_the_first_verdict():
    verifier = Verifier(
        [
            result(False, split=1.0, rejected_by="last"),
            result(False, rejected_by="error", error="decoder: ERR"),
        ]
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert isinstance(answer, Verdict)
    assert not answer.accepted
    assert answer.reason == "last"


class SlowVerifier(KaldiVerifier):
    """A KaldiVerifier whose one decode waits to be released."""

    def __init__(self):
        self.release = asyncio.Event()
        self.finished = False
        self._decoding = set()
        self._decoded = {}

    async def _decode_window(self, wake_id, window):
        await self.release.wait()
        self.finished = True
        return result(True, split=0.5)


@pytest.mark.asyncio
async def test_a_cancelled_run_still_finishes_its_decode():
    """The decoder answers in order, so a decode given up halfway would leave
    its answer for the next request to read as its own."""
    verifier = SlowVerifier()
    waiting = asyncio.create_task(verifier._decode("okay_nabu", b""))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    verifier.release.set()
    await asyncio.gather(*verifier._decoding)
    assert verifier.finished
    assert not verifier._decoding


class Ctc:
    """The CTC head, scripted: a second opinion and a placement."""

    def __init__(self, overturned=False, placed=0.9):
        self.overturned = overturned
        self.placed = placed
        self.judged = []
        self.places = []

    def knows(self, wake_id):
        return wake_id == "okay_nabu"

    async def second_opinion(self, wake_id, rejected_by, pcm):
        self.judged.append((rejected_by, len(pcm)))
        return RatioVerdict(self.overturned, score=1.0 if self.overturned else 2.0)

    async def place(self, wake_id, pcm, wake_start, kaldi_end):
        self.places.append((len(pcm), wake_start, kaldi_end))
        return self.placed, "okay nabu"


@pytest.mark.asyncio
async def test_placement_hears_past_the_detection():
    ctc = Ctc(placed=1.05)
    accepted = result(True, split=0.9, wake_start=0.2)
    verifier = Verifier([accepted, accepted], split=SPLIT_PLACED, ctc=ctc)

    assert await verify(verifier, PREROLL_MS) == MoreAudio(ms=TAIL_MS)
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert answer.accepted
    assert answer.wake_word_audio_ms == 1050
    # Kaldi's times are from the start of the run, not of its window
    assert ctc.places == [(PREROLL + TAIL, 0.2 + 0.0, 0.9)]


class Remembering(Verifier):
    """Scripted below the memo, so that what is remembered shows."""

    async def _decode(self, wake_id, window):
        return await KaldiVerifier._decode(self, wake_id, window)

    async def _decode_window(self, wake_id, window):
        self.windows.append(len(window))
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_a_second_try_does_not_decode_again():
    """Asked again with more audio, the windows already decoded are not."""
    verifier = Remembering([result(True, split=0.9)], split=SPLIT_PLACED, ctc=Ctc())
    audio = bytes(range(256)) * (ms_to_bytes(PREROLL_MS + TAIL_MS) // 256 + 1)

    assert isinstance(await verifier.verify(REQUEST, audio[:PREROLL], False), MoreAudio)
    answer = await verifier.verify(REQUEST, audio[: PREROLL + TAIL], False)

    assert answer.accepted
    assert verifier.windows == [PREROLL]


@pytest.mark.asyncio
async def test_a_second_opinion_overturns_a_last_refusal():
    ctc = Ctc(overturned=True, placed=0.95)
    verifier = Verifier(
        [result(False, split=0.8, rejected_by="last")], split=SPLIT_PLACED, ctc=ctc
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert answer.accepted
    assert answer.reason == "overturned_last"
    assert answer.wake_word_audio_ms == 950
    assert answer.detail["second_opinion"]["overturned"]
    # The pre-roll it was calibrated on, ending where Kaldi's window did
    assert ctc.judged == [("last", ms_to_bytes(min(RATIO_WINDOW_MS, PREROLL_MS)))]


@pytest.mark.asyncio
async def test_a_second_opinion_can_keep_a_refusal():
    ctc = Ctc(overturned=False)
    verifier = Verifier(
        [result(False, split=0.8, rejected_by="prefix")], split=SPLIT_PLACED, ctc=ctc
    )
    answer = await verify(verifier, PREROLL_MS + TAIL_MS)

    assert not answer.accepted
    assert answer.reason == "prefix"
    assert answer.wake_word_audio_ms is None
    assert ctc.places == []


@pytest.mark.asyncio
async def test_a_confusable_is_not_second_guessed():
    """Telling "nabu" from "now" is what Kaldi's lexicon is for."""
    ctc = Ctc(overturned=True)
    verifier = Verifier(
        [result(False, split=0.8, rejected_by="confusable")], split=SPLIT_PLACED, ctc=ctc
    )
    # No waiting either: there is nothing to place
    answer = await verify(verifier, PREROLL_MS)

    assert not answer.accepted
    assert answer.reason == "confusable"
    assert ctc.judged == []
