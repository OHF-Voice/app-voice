ARG BUILD_FROM

# ---------------------------------------------------------------------------
# Builder: the full Kaldi tree, used once to build the decoding graphs and
# the resident decoder, and then left behind.
# ---------------------------------------------------------------------------
FROM ${BUILD_FROM} AS builder
ARG BUILD_ARCH

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /usr/src/voice

COPY requirements.txt ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
        python3-venv \
    && python3 -m venv /usr/src/.venv \
    && /usr/src/.venv/bin/pip3 install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/usr/src/.venv/bin:${PATH}"

# The Kaldi and OpenFST binaries and the acoustic model are staged into
# local/ by script/stage_build rather than fetched here: they are ~255 MB
# that no package index serves. The verifier's code is not among them -- it
# is vendored in voice/kaldi/ -- so there is nothing to install, and an
# image built without local/ is a working app that verifies nothing, which
# is what a machine with no model should look like.
COPY local/ ./local/
RUN test -d local/wake-kaldi-models \
    || echo "No acoustic model staged: this image will verify nothing"

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
        /usr/src/.venv/bin/python3 -m voice \
            --wake-config wake.yaml \
            --train-dir /usr/src/voice/graphs \
            --build-graphs; \
    else \
        mkdir -p /usr/src/voice/graphs; \
    fi

# What the decoder links and opens, and nothing else.
COPY script/ ./script/
RUN /usr/src/.venv/bin/python3 script/prune_runtime local /usr/src/lean-local


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
COPY --from=builder /usr/src/voice/graphs/ ./graphs/
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
    CMD curl -sf http://127.0.0.1:10800/health > /dev/null || exit 1
