"""The CTC model, downloaded the first time the app starts.

Parakeet TDT-CTC 110M's encoder (int8) and CTC head, as voice-incubator
downloads them: about 174 MB, pinned to a revision and checked file by file
against its SHA-256. In the app it goes in ``/data/models``, which survives
updates; elsewhere beside the app (``var/models``), or wherever
``--models-dir`` says.

Until it is there the verifier works without it: Kaldi alone, and the
command starting just after Kaldi's end of the wake phrase.

Parakeet TDT-CTC 110M is NVIDIA's (https://huggingface.co/nvidia/parakeet-tdt_ctc-110m),
licensed CC-BY-4.0.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import aiohttp

_LOGGER = logging.getLogger(__name__)

REPO = (
    "https://huggingface.co/rhasspy/voice-incubator-models/resolve/"
    "e359089e2dfe430415b24410a089fdcf36fd5bf9"
)
MODEL = "parakeet-tdt_ctc-110m-en-int8"
RETRY_SECONDS = 60.0
CHUNK = 1 << 20


@dataclass(frozen=True)
class File:
    name: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        return f"{REPO}/{MODEL}/{self.name}"


FILES = (
    File("encoder.onnx", "8e19f44028ee520c9734b0cd85582a1b2aab992e4fcafac0dbd835fba4ed445c", 171777188),
    File("ctc_head.npz", "5459bf21d35af241ac604ef0007805f2b0ddc91ef6f3c2f04eb1ab5851d9cda8", 2104188),
    File("tokens.txt", "450e56bd2f036fe5b6aa821865838cc5aa9d8b0106134ce9a9ba0664abe6cd10", 9953),
)


def default_models_dir(app_dir: Path) -> Path:
    """The app's data folder when there is one (the Supervisor's ``/data``)."""
    data = Path("/data")
    return data / "models" if data.is_dir() else app_dir / "var" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(path: Path, file: File) -> bool:
    return path.is_file() and path.stat().st_size == file.size and _sha256(path) == file.sha256


async def _download(session: aiohttp.ClientSession, file: File, path: Path) -> None:
    partial = path.with_name(path.name + ".part")
    digest = hashlib.sha256()
    size = 0
    async with session.get(file.url, raise_for_status=True) as response:
        with open(partial, "wb") as out:
            async for chunk in response.content.iter_chunked(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    if size != file.size or digest.hexdigest() != file.sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{file.name}: got {size} bytes that do not match its SHA-256")
    partial.replace(path)


async def ensure_model(models_dir: Path) -> Path:
    """The model's folder, once every file in it is there and intact.

    Downloads what is missing or wrong, and keeps trying: a home without
    internet when the app starts may have it an hour later.
    """
    folder = models_dir / MODEL
    while True:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            missing = [
                file
                for file in FILES
                if not await asyncio.to_thread(_is_complete, folder / file.name, file)
            ]
            if not missing:
                return folder
            total = sum(file.size for file in missing) / 1e6
            _LOGGER.info("Downloading %s (%.0f MB) to %s", MODEL, total, folder)
            timeout = aiohttp.ClientTimeout(total=None, sock_read=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for file in missing:
                    await _download(session, file, folder / file.name)
            _LOGGER.info("Downloaded %s", MODEL)
        except (aiohttp.ClientError, OSError, ValueError, TimeoutError) as err:
            _LOGGER.warning(
                "Cannot download %s (%s); trying again in %.0f s", MODEL, err, RETRY_SECONDS
            )
            await asyncio.sleep(RETRY_SECONDS)
