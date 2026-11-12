# Home Assistant App: Voice

Source repository of the [Voice][docs] app for Home Assistant.
Users install the app from the [official app repository][index].

A second opinion on what your voice satellites hear. Home Assistant streams
the app a whole interaction's audio, and it answers three questions: was that
really the wake word, where does the command begin, and where does it end.

- [voice/README.md](voice/README.md) — how it works, and the measurements behind it
- [voice/PROTOCOL.md](voice/PROTOCOL.md) — what Home Assistant and the app say to each other
- [voice/DOCS.md](voice/DOCS.md) — what a user sees

## Releases

- Every push to `main` builds and publishes the image tagged `dev` and with the short commit SHA.
  These builds can be installed from the [development app repository][dev].
- Publishing a GitHub release builds the image tagged with the release version and `stable`.
  The release notes become the app changelog in the [official app repository][index].

## Two images

The app image is what Home Assistant installs, built and published by the
shared app workflows. A **standalone** image is published alongside it on
every tagged release, for running the app by hand or beside a Home Assistant
that is not an appliance:

```sh
docker run -d -p 10800:10800 ghcr.io/ohf-voice/app-voice-standalone:latest
```

It needs no arguments and no Supervisor: it serves `/info`, `/run` and
`/health` on 10800, the same as the app does.

## Local build

```sh
docker build -t local/app-voice voice                      # the app
docker build --target standalone -t local/app-voice-sa voice   # standalone
```

The Kaldi and OpenFST binaries and the acoustic model are fetched by the
build itself, from the [rhasspy-speech][assets] dataset and pinned to a
revision, so the build context stays small and every build — a release, CI,
somebody's laptop — gets the same bytes.

Running the app _outside_ Docker still needs them staged by hand:

```sh
voice/script/stage_build [path to a wyoming-wake-kaldi checkout]
```

The build is two stages. The first has the whole Kaldi tree and uses it once,
to build the decoding graphs for every wake word in `wake.yaml` and the
resident decoder. The second keeps only what the decoder links and opens —
about 62 MB of the 250 MB staged — so the image carries no compiler, no Kaldi
binaries, no OpenFST and no training FSTs.

The trade is that **the wake words are fixed at build time**: adding one means
editing `wake.yaml` and rebuilding, not restarting.

## Running it while working on it

```sh
cd voice && python3 -m voice --port 10800 --wake-config wake.yaml --train-dir var/wake-train
```

Then add it in Home Assistant from **Settings → Devices & services → Add
integration → Voice**, which asks for a host and port when there is no
Supervisor to install an app from.

## Tests

```sh
cd voice && python3 -m pytest
```

[assets]: https://huggingface.co/datasets/rhasspy/rhasspy-speech
[docs]: https://www.home-assistant.io/integrations/voice_support
[index]: https://github.com/home-assistant/apps
[dev]: https://github.com/home-assistant/apps-dev
