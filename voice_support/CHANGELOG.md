# Changelog

## 0.2.0

- A second opinion from Parakeet's CTC head on a wake word Kaldi refused as
  not the wake word (`prefix` or `last`): of the wake words the detector
  caught on the office-multicap corpus, 91% are now accepted instead of 69%.
  The model (174 MB) is downloaded the first time the app starts.
- Speech-to-text now starts where Parakeet's CTC head finds the wake phrase
  ending, so a command said straight after the wake word keeps its first
  word more often.
- The Kaldi decoder no longer gets out of step when a run ends while its
  wake word is being verified, which made every later verdict the previous
  one's; a decode that fails no longer refuses the wake word.

## 0.1.0

- First release. Home Assistant's `voice_support` integration streams a whole
  interaction's audio over a websocket, and the app answers three things: was
  that really the wake word, where does the command begin, and where does it
  end.
