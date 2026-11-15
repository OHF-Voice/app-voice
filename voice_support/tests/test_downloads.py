"""Downloading the CTC model: checked, kept, and not fetched twice."""

import asyncio
import hashlib
import sys
from pathlib import Path

import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parent.parent))

from voice import downloads  # noqa: E402
from voice.downloads import File, ensure_model  # noqa: E402

CONTENT = {"encoder.onnx": b"encoder" * 1000, "tokens.txt": b"<blk> 0\n"}


def files() -> tuple[File, ...]:
    return tuple(
        File(name, hashlib.sha256(data).hexdigest(), len(data))
        for name, data in CONTENT.items()
    )


@pytest.fixture
async def server(monkeypatch):
    """A model repository that serves CONTENT (or what a test puts there)."""
    served = dict(CONTENT)
    requests = []

    async def handle(request: web.Request) -> web.Response:
        name = request.match_info["name"]
        requests.append(name)
        return web.Response(body=served[name])

    app = web.Application()
    app.router.add_get("/model/{name}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # pylint: disable=protected-access

    monkeypatch.setattr(downloads, "REPO", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(downloads, "MODEL", "model")
    monkeypatch.setattr(downloads, "FILES", files())
    monkeypatch.setattr(downloads, "RETRY_SECONDS", 0.01)
    yield served, requests
    await runner.cleanup()


@pytest.mark.asyncio
async def test_the_model_is_downloaded(server, tmp_path):
    folder = await ensure_model(tmp_path)

    assert folder == tmp_path / "model"
    for name, data in CONTENT.items():
        assert (folder / name).read_bytes() == data
    assert not list(folder.glob("*.part"))


@pytest.mark.asyncio
async def test_a_damaged_download_is_not_kept(server, tmp_path):
    served, requests = server
    served["encoder.onnx"] = b"not the encoder"

    task = asyncio.create_task(ensure_model(tmp_path))
    while requests.count("encoder.onnx") < 2:
        await asyncio.sleep(0.01)
    assert not (tmp_path / "model" / "encoder.onnx").exists()

    # Fixed upstream: the next try gets it
    served["encoder.onnx"] = CONTENT["encoder.onnx"]
    folder = await asyncio.wait_for(task, 5)
    assert (folder / "encoder.onnx").read_bytes() == CONTENT["encoder.onnx"]


@pytest.mark.asyncio
async def test_what_is_there_is_not_downloaded_again(server, tmp_path):
    _, requests = server
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "encoder.onnx").write_bytes(CONTENT["encoder.onnx"])
    (folder / "tokens.txt").write_bytes(b"something else")

    await ensure_model(tmp_path)

    assert requests == ["tokens.txt"]
    assert (folder / "tokens.txt").read_bytes() == CONTENT["tokens.txt"]
