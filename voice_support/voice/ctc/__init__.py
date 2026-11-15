"""Parakeet's CTC head, for a second opinion on a wake word and for where it ends.

Vendored from adapt-ctc (commit 07ef328, as voice-incubator carries it at
4694538): ``acoustic``, ``runtime``, ``graph``, ``spotter`` and ``text`` as
they are, except that ``text`` does not spell out digits (wake phrases have
none) and ``graph`` keeps the one number helper it needs, so neither needs
ICU's number rules; ``wake`` without the whole-call command recognizer.
``model`` is adapt-ctc's hybrid model without its transducer. The native
Viterbi module is not built: the numpy search it falls back to gives the
same results, in a few milliseconds on graphs this size.

Nor is SciPy installed, though its FFT is NeMo's and numpy's differs from it
in the last bit: SciPy is 139 MB, and on the office-multicap corpus the
difference moved 29 of 7,570 second opinions across the threshold, either
way, leaving the wake words kept (91.3%) and the near misses let through
(21.9% against 21.6%) where they were.
"""
