"""SQLite parity tests for private knowledge-base activity."""

import json
from uuid import UUID

import pytest
from fastapi import HTTPException
from tests.helpers import kb as kb_helpers


USER_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
KB_ID = "11111111-1111-1111-1111-111111111111"


async def rows(db):
    cursor = await db.execute(
        "SELECT event_type, subject_kind, subject_title, subject_path, metadata "
        "FROM knowledge_base_events ORDER BY id",
    )
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in await cursor.fetchall()]


@pytest.fixture
async def local_db(tmp_path):
    from infra.db.sqlite import create_pool

    db = await create_pool(str(tmp_path / "index.db"))
    await db.execute(
        "INSERT INTO workspace (id, name, user_id) VALUES (?, 'Local Wiki', ?)",
        (KB_ID, USER_ID),
    )
    await db.commit()
    yield db
    await db.close()


async def test_local_page_events_and_technical_writes(local_db):
    db = local_db
    doc_id = "22222222-2222-2222-2222-222222222222"
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, content, metadata, version, document_number) "
        "VALUES (?, ?, 'page.md', 'Page', '/wiki/', 'wiki/page.md', 'wiki', "
        "'md', 'ready', 'one', '{}', 1, 7)",
        (doc_id, USER_ID),
    )
    await db.execute(
        "UPDATE documents SET content = 'two', version = 2 WHERE id = ?",
        (doc_id,),
    )
    await db.execute(
        "UPDATE documents SET metadata = ? WHERE id = ?",
        (json.dumps({"course": {"status": "complete"}}), doc_id),
    )
    await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    await db.commit()

    events = await rows(db)
    assert [event["event_type"] for event in events] == [
        "wiki.created",
        "page.created",
        "page.updated",
        "page.deleted",
    ]
    assert json.loads(events[2]["metadata"])["changes"] == ["content"]
    assert events[-1]["subject_path"] == "/wiki/page.md"


async def test_local_source_extraction_and_hidden_assets_are_quiet(local_db):
    db = local_db
    source_id = "33333333-3333-3333-3333-333333333333"
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, content, metadata) "
        "VALUES (?, ?, 'paper.pdf', 'Paper', '/', 'paper.pdf', 'source', 'pdf', "
        "'pending', NULL, '{}')",
        (source_id, USER_ID),
    )
    await db.execute(
        "UPDATE documents SET content = 'Extracted', status = 'ready' WHERE id = ?",
        (source_id,),
    )
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, metadata) "
        "VALUES ('44444444-4444-4444-4444-444444444444', ?, 'asset.png', 'Asset', "
        "'/paper.assets/', 'paper.assets/asset.png', 'asset', 'png', 'ready', ?)",
        (USER_ID, json.dumps({"hidden": True, "asset": True})),
    )
    await db.commit()

    events = await rows(db)
    assert [event["event_type"] for event in events] == [
        "wiki.created",
        "source.added",
    ]


async def test_local_feed_endpoint_is_owner_scoped_and_cursor_paginated(local_db, tmp_path):
    from routes.events import list_knowledge_base_events

    db = local_db
    for index in range(3):
        await db.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
            "source_kind, file_type, status, content) VALUES (?, ?, ?, ?, '/wiki/', ?, "
            "'wiki', 'md', 'ready', '')",
            (
                f"77777777-7777-7777-7777-77777777777{index}",
                USER_ID,
                f"page-{index}.md",
                f"Page {index}",
                f"wiki/page-{index}.md",
            ),
        )
    await db.commit()

    # La rotta risolve la KB dalla richiesta: niente percorso e niente
    # testata significa «il legacy», e qui il legacy di questa radice base E'
    # KB_ID. Senza registrarlo, `apri_db` aprirebbe una SECONDA connessione
    # sullo stesso file e le due si contenderebbero il lock.
    base = tmp_path / "base"
    base.mkdir()
    request = kb_helpers.richiesta(base, kb_id=KB_ID, db=db)
    kb_helpers.registra(base, KB_ID, db)
    first = await list_knowledge_base_events(
        UUID(KB_ID), request, USER_ID, limit=2, before=None,
    )
    assert len(first["items"]) == 2
    assert first["next_cursor"] == first["items"][-1]["id"]
    second = await list_knowledge_base_events(
        UUID(KB_ID), request, USER_ID, limit=2, before=int(first["next_cursor"]),
    )
    assert len(second["items"]) == 2

    with pytest.raises(HTTPException) as error:
        await list_knowledge_base_events(
            UUID(KB_ID), request, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", limit=2, before=None,
        )
    assert error.value.status_code == 404

    # Una KB che la radice base non conosce non ripiega sul legacy: 404.
    ignota = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    with pytest.raises(HTTPException) as mancante:
        await list_knowledge_base_events(
            UUID(ignota),
            kb_helpers.richiesta(base, kb_id=ignota, db=db),
            USER_ID, limit=2, before=None,
        )
    assert mancante.value.status_code == 404

    kb_helpers.scorda(base)


async def test_existing_local_database_gets_honest_idempotent_backfill(tmp_path):
    from infra.db.sqlite import create_pool

    path = str(tmp_path / "existing.db")
    db = await create_pool(path)
    await db.execute(
        "INSERT INTO workspace (id, name, user_id) VALUES (?, 'Existing', ?)",
        (KB_ID, USER_ID),
    )
    await db.execute(
        "INSERT INTO documents (id, user_id, filename, title, path, relative_path, "
        "source_kind, file_type, status, content, metadata) "
        "VALUES ('55555555-5555-5555-5555-555555555555', ?, 'notes.md', 'Notes', "
        "'/', 'notes.md', 'source', 'md', 'ready', 'notes', '{}')",
        (USER_ID,),
    )
    await db.executescript(
        "DROP TRIGGER knowledge_base_activity_insert;"
        "DROP TRIGGER document_activity_insert;"
        "DROP TRIGGER document_activity_update;"
        "DROP TRIGGER document_activity_delete;"
        "DROP TABLE knowledge_base_events;"
    )
    await db.close()

    db = await create_pool(path)
    try:
        events = await rows(db)
        assert [event["event_type"] for event in events] == [
            "wiki.created",
            "source.added",
            "tracking.started",
        ]

        await db.close()
        db = await create_pool(path)
        assert len(await rows(db)) == 3
    finally:
        await db.close()
