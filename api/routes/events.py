"""Private knowledge-base activity feed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from deps import get_user_id

if TYPE_CHECKING:
    import aiosqlite
    import asyncpg

router = APIRouter(tags=["events"])

_EVENT_COLUMNS = (
    "id, event_type, subject_kind, document_id, document_number, "
    "document_version, subject_title, subject_path, metadata, occurred_at"
)


@router.get("/v1/knowledge-bases/{kb_id}/events")
async def list_knowledge_base_events(
    kb_id: UUID,
    request: Request,
    user_id: Annotated[str, Depends(get_user_id)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    before: Annotated[int | None, Query(ge=1, le=9_223_372_036_854_775_807)] = None,
) -> dict:
    """Return an owner-only, newest-first page of immutable raw events."""
    if request.app.state.mode == "local":
        # L'ambito si risolve QUI e non come dipendenza della firma: in
        # modalita' ospitata non esiste una radice di workspace, e una
        # dipendenza dichiarata girerebbe anche la'.
        from deps import get_kb_scope

        scope = await get_kb_scope(request)
        await _require_local_ownership(scope.db, user_id)
        rows = await _fetch_local_events(scope.db, user_id, limit, before)
    else:
        pool = request.app.state.pool
        await _require_hosted_ownership(pool, kb_id, user_id)
        rows = await _fetch_hosted_events(pool, kb_id, user_id, limit, before)

    has_more = len(rows) > limit
    page = rows[:limit]
    return {
        "items": [_serialize_event(row) for row in page],
        "next_cursor": str(page[-1]["id"]) if has_more and page else None,
    }


def _serialize_event(row: dict) -> dict:
    return {
        "id": str(row["id"]),
        "event_type": row["event_type"],
        "subject_kind": row["subject_kind"],
        "document_id": str(row["document_id"]) if row["document_id"] else None,
        "document_number": row["document_number"],
        "document_version": row["document_version"],
        "subject_title": row["subject_title"],
        "subject_path": row["subject_path"],
        "metadata": _decode_metadata(row["metadata"]),
        "occurred_at": _iso_timestamp(row["occurred_at"]),
    }


def _decode_metadata(value: dict | str) -> dict:
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _iso_timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat().replace("+00:00", "Z")
    text = str(value)
    if "T" not in text and " " in text:
        text = text.replace(" ", "T", 1)
    if not text.endswith(("Z", "+00:00")):
        text += "Z"
    return text.replace("+00:00", "Z")


async def _require_local_ownership(
    db: aiosqlite.Connection, user_id: str,
) -> None:
    """L'identita' della KB l'ha gia' provata l'ambito: questa connessione
    E' il database di quella wiki. Qui resta da verificare il PROPRIETARIO,
    e quindi non si filtra piu' per `id` — un id della riga diverso da
    quello della cartella svuoterebbe il feed senza dire niente."""
    cursor = await db.execute(
        "SELECT 1 FROM workspace WHERE user_id = ? LIMIT 1",
        (user_id,),
    )
    if await cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Wiki not found")


async def _require_hosted_ownership(
    pool: asyncpg.Pool, kb_id: UUID, user_id: str,
) -> None:
    owns_wiki = await pool.fetchval(
        "SELECT 1 FROM knowledge_bases WHERE id = $1 AND user_id = $2",
        str(kb_id), user_id,
    )
    if not owns_wiki:
        raise HTTPException(status_code=404, detail="Wiki not found")


async def _fetch_local_events(
    db: aiosqlite.Connection, user_id: str,
    limit: int, before: int | None,
) -> list[dict]:
    # Nessun filtro per knowledge_base_id: in modalita' locale QUESTO
    # database contiene gli eventi di UNA wiki sola, e filtrare per un id
    # che potrebbe non combaciare con la cartella renderebbe il feed vuoto
    # invece di sbagliato — un guasto che non da' errore.
    cursor_clause = " AND id < ?" if before is not None else ""
    params: list[Any] = [user_id]
    if before is not None:
        params.append(before)
    params.append(limit + 1)
    cursor = await db.execute(
        f"SELECT {_EVENT_COLUMNS} FROM knowledge_base_events "
        "WHERE user_id = ?"
        f"{cursor_clause} ORDER BY id DESC LIMIT ?",
        tuple(params),
    )
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in await cursor.fetchall()]


async def _fetch_hosted_events(
    pool: asyncpg.Pool, kb_id: UUID, user_id: str,
    limit: int, before: int | None,
) -> list[dict]:
    records = await pool.fetch(
        f"SELECT {_EVENT_COLUMNS} FROM knowledge_base_events "
        "WHERE knowledge_base_id = $1 AND user_id = $2 "
        "AND ($3::bigint IS NULL OR id < $3) "
        "ORDER BY id DESC LIMIT $4",
        str(kb_id), user_id, before, limit + 1,
    )
    return [dict(record) for record in records]
