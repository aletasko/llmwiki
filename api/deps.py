import asyncio
import inspect
import json
from pathlib import Path
from typing import Annotated, AsyncGenerator

import kb_scope
from fastapi import Depends, Request
from kb_scope import AmbitoKB
from scoped_db import ScopedDB

DB_ACQUIRE_TIMEOUT_SECONDS = 10.0


async def _acquire_connection(pool):
    """Acquire without allowing a saturated pool to hang a request forever."""
    try:
        parameters = inspect.signature(pool.acquire).parameters.values()
        supports_timeout = any(
            parameter.name == "timeout"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_timeout = False

    if supports_timeout:
        return await pool.acquire(timeout=DB_ACQUIRE_TIMEOUT_SECONDS)
    return await asyncio.wait_for(
        pool.acquire(),
        timeout=DB_ACQUIRE_TIMEOUT_SECONDS,
    )


async def get_pool(request: Request):
    return request.app.state.pool


async def get_user_id(request: Request) -> str:
    """Authenticate and return user_id.

    In local mode, auth_provider is a LocalAuthProvider (always returns the fixed user).
    In hosted mode, auth_provider is None and we fall through to Supabase JWKS.
    The auth path is determined at startup, not here.
    """
    auth_provider = request.app.state.auth_provider
    if auth_provider:
        return await auth_provider.get_current_user(request)
    from auth import get_current_user
    return await get_current_user(request)


def radice_base(request: Request) -> Path:
    """La cartella che contiene TUTTE le knowledge base locali.

    Viene dallo stato dell'app, scritto una volta all'avvio: non da
    `settings.WORKSPACE_PATH` letto per richiesta, perche' una richiesta non
    deve poter spostare la radice di un'altra.
    """
    return Path(request.app.state.workspace_base).resolve()


async def get_kb_scope(request: Request) -> AmbitoKB:
    """La KB di QUESTA richiesta: id nel percorso, altrimenti `X-Wiki-KB`,
    altrimenti il legacy.

    L'id si legge da `request.path_params` invece di dichiararlo come
    parametro: cosi' la stessa dipendenza vale per le rotte che hanno
    `{kb_id}` nel percorso e per quelle che hanno solo un id di documento, e
    non c'e' una seconda forma della regola da tenere allineata.
    """
    kb_id = kb_scope.kb_dalla_richiesta(request, request.path_params.get("kb_id"))
    return await kb_scope.ambito(kb_id, radice_base(request))


def _e_locale(request: Request) -> bool:
    return getattr(request.app.state, "mode", None) == "local"


async def get_user_service(request: Request):
    user_id = await get_user_id(request)
    if _e_locale(request):
        return request.app.state.factory.user_service(
            user_id, scope=await get_kb_scope(request),
        )
    return request.app.state.factory.user_service(user_id)


async def get_kb_service(request: Request):
    user_id = await get_user_id(request)
    return request.app.state.factory.kb_service(user_id)


async def get_document_service(request: Request):
    user_id = await get_user_id(request)
    if _e_locale(request):
        return request.app.state.factory.document_service(
            user_id, scope=await get_kb_scope(request),
        )
    return request.app.state.factory.document_service(user_id)


async def get_scoped_db(
    request: Request,
    pool: Annotated = Depends(get_pool),
) -> AsyncGenerator[ScopedDB, None]:
    """Scoped DB connection.

    In local mode (pool is None), returns a thin wrapper around the SQLite
    connection OF THE RESOLVED KB — no RLS, no transaction management.
    In hosted mode, returns a proper RLS-enforced Postgres connection.

    The branching is on app.state.pool being None, which is set once at startup.
    """
    if pool is None:
        scope = await get_kb_scope(request)
        user_id = await get_user_id(request)
        yield ScopedDB(None, scope.db, user_id)
        return

    from auth import get_current_user
    user_id = await get_current_user(request)
    conn = await _acquire_connection(pool)
    try:
        # The transaction context covers startup too. If BEGIN, RLS setup,
        # request handling, or cancellation fails, it rolls back before the
        # connection is returned to the pool.
        async with conn.transaction():
            claims = json.dumps({"sub": user_id})
            await conn.execute("SET LOCAL ROLE authenticated")
            await conn.execute("SELECT set_config('request.jwt.claims', $1, true)", claims)
            yield ScopedDB(pool, conn, user_id)
    finally:
        await asyncio.shield(pool.release(conn))
