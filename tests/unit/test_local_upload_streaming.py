import asyncio
import hashlib
import threading

import pytest
from domain import local_processor
from fastapi import HTTPException
from kb_scope import AmbitoKB
from infra.db.sqlite import create_pool
from routes import local_upload
from services import chunker


class ChunkedUpload:
    def __init__(self, data: bytes, filename: str = "document.bin"):
        self._data = data
        self._offset = 0
        self.filename = filename
        self.read_sizes: list[int] = []

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        chunk = self._data[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


KB = "99999999-9999-4999-8999-999999999999"


async def _scope_with_db(tmp_path):
    """L'ambito della KB al posto della richiesta: la rotta non va piu' a
    prendersi connessione e radice dallo stato dell'app."""
    db = await create_pool(str(tmp_path / "local.db"))
    return AmbitoKB(kb_id=KB, radice=tmp_path, db=db), db


def _scope_senza_db(tmp_path):
    """Per i casi che devono fallire PRIMA di toccare il database."""
    return AmbitoKB(kb_id=KB, radice=tmp_path, db=None)


async def test_stream_upload_writes_chunks_and_hashes(monkeypatch, tmp_path):
    monkeypatch.setattr(local_upload, "UPLOAD_CHUNK_BYTES", 4)
    payload = b"streamed payload"
    upload = ChunkedUpload(payload)
    dest = tmp_path / "nested" / "document.bin"

    size, digest = await local_upload._stream_upload_to_path(
        upload,
        dest,
        max_bytes=100,
    )

    assert dest.read_bytes() == payload
    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert len(upload.read_sizes) > 2
    assert set(upload.read_sizes) == {4}
    assert not list(dest.parent.glob("*.upload"))


async def test_stream_upload_rejects_oversize_without_replacing_existing_file(tmp_path):
    dest = tmp_path / "document.bin"
    dest.write_bytes(b"existing")

    with pytest.raises(HTTPException) as exc:
        await local_upload._stream_upload_to_path(
            ChunkedUpload(b"too-large"),
            dest,
            max_bytes=4,
        )

    assert exc.value.status_code == 413
    assert dest.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.upload"))


async def test_cancellation_waits_for_active_write_before_cleanup(monkeypatch, tmp_path):
    monkeypatch.setattr(local_upload, "UPLOAD_CHUNK_BYTES", 4)
    started = threading.Event()
    release = threading.Event()
    real_write = local_upload._write_chunk

    def blocking_write(output, chunk):
        started.set()
        assert release.wait(timeout=5)
        real_write(output, chunk)

    monkeypatch.setattr(local_upload, "_write_chunk", blocking_write)
    dest = tmp_path / "document.bin"
    task = asyncio.create_task(
        local_upload._stream_upload_to_path(ChunkedUpload(b"abcdefgh"), dest)
    )

    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not dest.exists()
    assert not list(tmp_path.glob(".*.upload"))


async def test_cancellation_waits_for_active_rename_before_cleanup(monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()
    real_replace = local_upload._replace_path

    def blocking_replace(source, destination):
        started.set()
        assert release.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(local_upload, "_replace_path", blocking_replace)
    dest = tmp_path / "document.bin"
    task = asyncio.create_task(
        local_upload._stream_upload_to_path(ChunkedUpload(b"payload"), dest)
    )

    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not dest.exists()
    assert not list(tmp_path.glob(".*.upload"))


async def test_duplicate_relative_path_preserves_first_file_and_row(monkeypatch, tmp_path):
    scope, db = await _scope_with_db(tmp_path)
    try:
        first = b"first payload"
        created = await local_upload.upload_file(
            ChunkedUpload(first, "same.bin"),
            path="/",
            user_id="local-user",
            scope=scope,
        )

        with pytest.raises(HTTPException) as exc:
            await local_upload.upload_file(
                ChunkedUpload(b"replacement", "same.bin"),
                path="/",
                user_id="local-user",
                scope=scope,
            )

        assert exc.value.status_code == 409
        assert (tmp_path / "same.bin").read_bytes() == first
        cursor = await db.execute(
            "SELECT id, content_hash FROM documents WHERE relative_path = 'same.bin'"
        )
        rows = await cursor.fetchall()
        assert rows == [(created["id"], hashlib.sha256(first).hexdigest())]
        assert not list(tmp_path.glob(".*.upload"))
    finally:
        await db.close()


async def test_image_is_scheduled_but_unknown_binary_is_ready(monkeypatch, tmp_path):
    processed = asyncio.Event()
    processed_ids = []

    async def fake_process(_workspace, doc_id):
        processed_ids.append(doc_id)
        processed.set()

    monkeypatch.setattr(local_processor, "process_document_isolated", fake_process)
    scope, db = await _scope_with_db(tmp_path)
    try:
        image = await local_upload.upload_file(
            ChunkedUpload(b"fake image", "photo.png"),
            path="/",
            user_id="local-user",
            scope=scope,
        )
        unknown = await local_upload.upload_file(
            ChunkedUpload(b"opaque", "archive.unknown"),
            path="/",
            user_id="local-user",
            scope=scope,
        )
        await asyncio.wait_for(processed.wait(), timeout=2)

        assert image["status"] == "pending"
        assert processed_ids == [image["id"]]
        assert unknown["status"] == "ready"
    finally:
        await db.close()


async def test_text_upload_uses_ten_mib_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(local_upload, "MAX_TEXT_CONTENT_BYTES", 4)

    with pytest.raises(HTTPException) as exc:
        await local_upload.upload_file(
            ChunkedUpload(b"12345", "large.md"),
            path="/",
            user_id="local-user",
            scope=_scope_senza_db(tmp_path),
        )

    assert exc.value.status_code == 413
    assert not (tmp_path / "large.md").exists()
    assert not list(tmp_path.glob(".*.upload"))


async def test_text_read_and_chunking_run_off_event_loop(monkeypatch, tmp_path):
    scope, db = await _scope_with_db(tmp_path)
    caller_thread = threading.get_ident()
    chunk_thread = None

    def recording_chunk_text(content):
        nonlocal chunk_thread
        chunk_thread = threading.get_ident()
        return []

    monkeypatch.setattr(chunker, "chunk_text", recording_chunk_text)
    try:
        await local_upload.upload_file(
            ChunkedUpload(b"small text", "notes.md"),
            path="/",
            user_id="local-user",
            scope=scope,
        )
    finally:
        await db.close()

    assert chunk_thread is not None
    assert chunk_thread != caller_thread


async def test_cancellation_during_commit_leaves_text_fully_indexed(monkeypatch, tmp_path):
    scope, db = await _scope_with_db(tmp_path)
    real_commit = db.commit
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()

    async def blocking_commit():
        commit_started.set()
        await release_commit.wait()
        await real_commit()

    monkeypatch.setattr(db, "commit", blocking_commit)
    payload = ("A sentence with enough searchable words. " * 100).encode()
    task = asyncio.create_task(
        local_upload.upload_file(
            ChunkedUpload(payload, "indexed.md"),
            path="/",
            user_id="local-user",
            scope=scope,
        )
    )
    try:
        await asyncio.wait_for(commit_started.wait(), timeout=2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release_commit.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        cursor = await db.execute(
            "SELECT id, content FROM documents WHERE relative_path = 'indexed.md'"
        )
        document = await cursor.fetchone()
        assert document is not None
        assert document[1] == payload.decode()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?",
            (document[0],),
        )
        assert (await cursor.fetchone())[0] > 0
        assert (tmp_path / "indexed.md").read_bytes() == payload
        assert not list(tmp_path.glob(".*.upload"))
    finally:
        release_commit.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await db.close()
