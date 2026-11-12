ARG BUILD_FROM

# ---------------------------------------------------------------------------
# Builder: the full Kaldi tree, used once to build the decoding graphs and
# the resident decoder, and then left behind.
# ---------------------------------------------------------------------------
FROM ${BUILD_FROM} AS builder
ARG BUILD_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src/voice

# perl is Kaldi's: its egs scripts validate the lexicon with it. Only the
# builder needs it, which is the point of building the graphs here.
COPY requirements.txt ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        curl \
        perl \
        python3 \
        python3-pip \
        python3-venv \
    && python3 -m venv /usr/src/.venv \
    && /usr/src/.venv/bin/pip3 install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/src/.venv/bin:${PATH}"

# The Kaldi and OpenFST binaries and the acoustic model: ~130 MB that no
# package index serves, published as part of rhasspy-speech. Fetched here
# rather than carried in the build context so that the context stays small
# and any build -- a release, CI, somebody's laptop -- gets the same bytes.
#
# Pinned to a revision: this is 130 MB of executables run on the user's
# machine, and "whatever main says today" is not a thing to inherit.
ARG TARGETARCH
ARG RHASSPY_SPEECH_REVISION=9e6bcf3af07909f39f71674f62372508d00dd4a9
ARG RHASSPY_SPEECH_URL=https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve
RUN \
    mkdir -p local/wake-kaldi-tools local/wake-kaldi-models \
    && curl -fsSL \
        "${RHASSPY_SPEECH_URL}/${RHASSPY_SPEECH_REVISION}/tools/rhasspy-speech_${TARGETARCH}.tar.gz" \
        | tar -xz -C local/wake-kaldi-tools \
    && curl -fsSL \
        "${RHASSPY_SPEECH_URL}/${RHASSPY_SPEECH_REVISION}/models/en_US-rhasspy.tar.gz" \
        | tar -xz -C local/wake-kaldi-models \
    && test -d local/wake-kaldi-tools/kaldi \
    && test -d local/wake-kaldi-models/en_US-rhasspy

COPY wake.yaml ./
COPY voice/ ./voice/
COPY config.yaml ./

# The resident decoder daemon, built from the vendored source against the
# staged Kaldi tools. Its build-time needs are not the app's, so they are
# installed and taken away again here; libopenblas0 is asked for by name so
# that --auto-remove keeps it, because the daemon links it at runtime.
#
# Skipping this is not a build failure: without the daemon every decode
# spawns its own Kaldi process, which works. On one window of "okay nabu",
# 42 ms with the daemon against roughly 200 ms without.
RUN \
    if [ -d local/wake-kaldi-tools ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            libgfortran5 \
            libopenblas0 \
        && apt-get install -y --no-install-recommends \
            g++ \
            libopenblas-dev \
        && bash voice/kaldi/decoderd/build.sh "$PWD/local/wake-kaldi-tools" \
        && rm -rf voice/kaldi/decoderd/.fst_shadow_include \
        && apt-get purge -y --auto-remove g++ libopenblas-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/* \
        && ldd voice/kaldi/decoderd/verify_decoderd > /dev/null; \
    else \
        echo "No Kaldi tools staged: decodes will each spawn a process"; \
    fi


# The graphs for every wake word in wake.yaml, built here so that the image
# below can ship without a trainer: Kaldi's binaries and egs scripts,
# OpenFST and the training FSTs are ~190 MB that nothing needs once the
# graphs exist. A wake word cannot be added without rebuilding the image.
RUN \
    if [ -f voice/kaldi/decoderd/verify_decoderd ]; then \
        /usr/src/.venv/bin/python3 -m voice --build-graphs; \
    fi

# What the decoder links and opens, and nothing else.
COPY script/ ./script/
RUN /usr/src/.venv/bin/python3 script/prune_runtime local /usr/src/lean-local


# ---------------------------------------------------------------------------
# Standalone: the same app without the Supervisor around it, for anyone
# running it by hand or beside a Home Assistant that is not an appliance.
#
# Built with --target standalone. It must come before the app stage, because
# the shared app workflows build this file with no target and take the last.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS standalone

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /usr/src/voice

# libopenblas0 and libgfortran5 are what the decoder links outside its own
# tree; curl is the health check's.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgfortran5 \
        libopenblas0 \
        python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/.venv /usr/src/.venv
ENV PATH="/usr/src/.venv/bin:${PATH}"

COPY --from=builder /usr/src/lean-local/ ./local/
COPY --from=builder /usr/src/voice/var/ ./var/
COPY --from=builder /usr/src/voice/voice/kaldi/decoderd/verify_decoderd ./voice/kaldi/decoderd/verify_decoderd

COPY wake.yaml config.yaml ./
COPY voice/ ./voice/

RUN \
    if [ -f voice/kaldi/decoderd/verify_decoderd ]; then \
        ldd voice/kaldi/decoderd/verify_decoderd > /dev/null; \
    fi

EXPOSE 10900
HEALTHCHECK --start-period=2m \
    CMD curl -sf http://127.0.0.1:10900/health > /dev/null || exit 1

ENTRYPOINT ["python3", "-m", "voice"]


# ---------------------------------------------------------------------------
# The app: no compiler, no Kaldi binaries, no OpenFST, no training FSTs.
# ---------------------------------------------------------------------------
FROM ${BUILD_FROM}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /usr/src/voice

# libopenblas0 and libgfortran5 are what the decoder links outside its own
# tree; curl is the health check's.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgfortran5 \
        libopenblas0 \
        python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/.venv /usr/src/.venv
ENV PATH="/usr/src/.venv/bin:${PATH}"

COPY --from=builder /usr/src/lean-local/ ./local/
COPY --from=builder /usr/src/voice/var/ ./var/
COPY --from=builder /usr/src/voice/voice/kaldi/decoderd/verify_decoderd ./voice/kaldi/decoderd/verify_decoderd

COPY wake.yaml config.yaml ./
COPY voice/ ./voice/

RUN \
    if [ -f voice/kaldi/decoderd/verify_decoderd ]; then \
        ldd voice/kaldi/decoderd/verify_decoderd > /dev/null; \
    fi

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=2m \
    CMD curl -sf http://127.0.0.1:10900/health > /dev/null || exit 1
