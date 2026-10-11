import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware as _BaseCORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


class CORSMiddleware(_BaseCORSMiddleware):
    """CORS middleware that passes WebSocket connections through.

    WebSocket auth is handled by JWT verification in the handler, not by
    origin checks. HTTP requests still get full CORS protection.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)

from config import settings

logger = logging.getLogger(__name__)

if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        send_default_pii=True,
        traces_sample_rate=0.1,
        environment=settings.STAGE,
    )

if settings.LOGFIRE_TOKEN:
    import logfire
    logfire.configure(token=settings.LOGFIRE_TOKEN, service_name="supavault-api")
    logfire.instrument_asyncpg()

from routes.health import router as health_router
from routes.knowledge_bases import router as knowledge_bases_router
from routes.documents import router as documents_router
from routes.events import router as events_router
from routes.me import router as me_router
from routes.usage import router as usage_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.MODE == "local":
        async with _local_lifespan(app):
            yield
        return

    # ── Hosted mode ──
    # Prefetch the Supabase JWKS so the first authenticated request doesn't
    # pay the cold-cache cost and so a JWKS outage at boot is visible
    # immediately rather than masked behind the first auth error.
    from auth import prefetch_jwks
    await prefetch_jwks()

    import asyncpg
    pool = await asyncpg.create_pool(settings.DATABASE_URL, min_size=2, max_size=10)
    app.state.pool = pool
    app.state.mode = "hosted"

    s3_service = None
    ocr_service = None
    if settings.AWS_ACCESS_KEY_ID and settings.S3_BUCKET:
        from services.s3 import S3Service
        s3_service = S3Service()
    if s3_service:
        from services.ocr import OCRService
        ocr_service = OCRService(s3_service, pool)

    app.state.s3_service = s3_service
    app.state.ocr_service = ocr_service
    app.state.auth_provider = None  # Uses Supabase JWKS auth via deps.py

    from services.hosted import HostedServiceFactory
    app.state.factory = HostedServiceFactory(pool, s3_service, ocr_service)

    # Real-time document change notifications via WebSocket
    from routes.ws import setup_listener
    listener_task = await setup_listener(settings.listen_database_url)

    from infra.tus import cleanup_stale_uploads
    cleanup_task = asyncio.create_task(cleanup_stale_uploads())

    if ocr_service:
        rows = await pool.fetch(
            "SELECT id::text, user_id::text FROM documents "
            "WHERE status IN ('pending', 'processing') AND NOT archived"
        )
        for row in rows:
            logger.info("Recovering stuck document %s", row["id"][:8])
            asyncio.create_task(ocr_service.process_document(row["id"], row["user_id"]))

    yield

    cleanup_task.cancel()
    listener_task.cancel()
    await pool.close()


async def _local_lifespan_inner(app: FastAPI):
    """Local mode: SQLite + local filesystem + single-user auth.

    `WORKSPACE_PATH` non e' piu' «il workspace»: e' la RADICE BASE che
    contiene la KB legacy (nella radice stessa) e tutte le altre sotto
    `kb/<uuid>/`. Qui si apre solo il legacy; le altre si aprono alla prima
    richiesta che le nomina (vedi `kb_scope.apri_db`).
    """
    import uuid
    from pathlib import Path

    import kb_scope
    from infra.db.sqlite import create_pool as create_sqlite_pool
    from infra.storage.local import LocalStorageService
    from infra.auth.local import LocalAuthProvider

    workspace = Path(settings.WORKSPACE_PATH).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "wiki").mkdir(exist_ok=True)
    (workspace / ".llmwiki").mkdir(exist_ok=True)
    (workspace / ".llmwiki" / "cache").mkdir(exist_ok=True)

    db_path = str(workspace / ".llmwiki" / "index.db")
    db = await create_sqlite_pool(db_path)

    local_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "local"))
    auth_provider = LocalAuthProvider(local_user_id)
    storage = LocalStorageService(str(workspace), settings.API_URL)

    # L'id della KB legacy si LEGGE, non si indovina: la riga `workspace` di
    # questo database e' la sua identita', e un id scritto a mano nel codice
    # che non combacia con la riga sono due nomi per la stessa wiki — uno dei
    # due non risolve, e nessun errore dice quale.
    cursor = await db.execute("SELECT id FROM workspace LIMIT 1")
    riga = await cursor.fetchone()
    if riga:
        legacy_id = str(riga[0])
    else:
        # Workspace nuovo: l'id e' quello canonico del legacy, non un UUID a
        # caso. Cosi' una installazione fresca e la produzione risolvono la
        # KB base con lo stesso id, e «senza indicazione» significa la stessa
        # cosa in entrambe.
        legacy_id = kb_scope.KB_LEGACY
        await db.execute(
            "INSERT INTO workspace (id, name, description, user_id) VALUES (?, ?, '', ?)",
            (legacy_id, workspace.name, local_user_id),
        )
        await db.commit()
        logger.info("Initialized local workspace: %s", workspace)
    kb_scope.registra_legacy(workspace, legacy_id)
    kb_scope.registra_connessione(workspace, legacy_id, db)

    app.state.mode = "local"
    app.state.pool = None  # No asyncpg pool in local mode
    app.state.sqlite_db = db  # la connessione del LEGACY, non «la» connessione
    app.state.s3_service = None
    app.state.storage_service = storage
    app.state.ocr_service = None
    app.state.auth_provider = auth_provider
    app.state.workspace_path = str(workspace)
    app.state.workspace_base = str(workspace)
    app.state.legacy_kb_id = legacy_id

    from services.local import LocalServiceFactory
    app.state.factory = LocalServiceFactory(workspace, local_user_id)

    logger.info(
        "Local mode — radice base: %s · KB legacy: %s · altre KB: %d",
        workspace, legacy_id, len(kb_scope.kb_registrate(workspace)) - 1,
    )
    return db


@asynccontextmanager
async def _local_lifespan(app: FastAPI):
    await _local_lifespan_inner(app)
    from pathlib import Path
    from infra.db.sqlite import create_pool as create_sqlite_pool
    workspace = Path(app.state.workspace_path)
    db_path = str(workspace / ".llmwiki" / "index.db")

    # Each background writer gets its own connection so a commit can't flush
    # another writer's (or a request handler's) open transaction.
    reconcile_db = await create_sqlite_pool(db_path, init_schema=False)
    watcher_db = await create_sqlite_pool(db_path, init_schema=False)

    from domain.local_processor import reconcile_workspace
    reconcile_task = asyncio.create_task(reconcile_workspace(reconcile_db, workspace))

    watcher_task = None
    try:
        from domain.watcher import watch_workspace
        watcher_task = asyncio.create_task(watch_workspace(watcher_db, workspace))
        logger.info("File watcher started")
    except ImportError:
        logger.warning("watchfiles not installed — file watcher disabled")

    try:
        yield
    finally:
        reconcile_task.cancel()
        try:
            await reconcile_task
        except asyncio.CancelledError:
            pass
        if watcher_task:
            watcher_task.cancel()
            try:
                await watcher_task
            except asyncio.CancelledError:
                pass
        await reconcile_db.close()
        await watcher_db.close()
        # Il legacy sta nella cache di kb_scope come ogni altra KB: lo
        # chiude quella, insieme a tutte le KB aperte a richiesta. Un
        # `db.close()` qui lascerebbe aperte le altre.
        import kb_scope

        await kb_scope.chiudi_tutte(workspace)


app = FastAPI(title="LLM Wiki API", lifespan=lifespan)

# Rate limiting — applied as middleware so every authenticated route gets a
# broad ceiling. Hot endpoints can add tighter `@limiter.limit(...)` overrides.
# Skip in local mode where there's only one user.
if settings.MODE != "local":
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    from infra.rate_limit import limiter

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

local_extension_origin_regex = None
if settings.MODE == "local":
    from infra.local_http import LocalHTTPBoundaryMiddleware, extension_origin_regex

    local_extension_origin_regex = extension_origin_regex(
        settings.local_extension_origins,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.APP_URL],
    # Local clipping may be initiated by the popup or extension background
    # worker. Hosted mode keeps its existing, web-app-only CORS policy.
    allow_origin_regex=local_extension_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Location", "Upload-Offset", "Upload-Length",
        "Tus-Resumable", "Tus-Version", "Tus-Max-Size", "Tus-Extension",
        "X-Document-Id",
    ],
    # Default preflight cache is 600s — on high-latency links every endpoint
    # re-pays an OPTIONS round-trip every 10 minutes. Browsers cap this
    # (Chrome 7200s, Firefox 86400s), so ask for the max.
    max_age=86400,
)

if settings.MODE == "local":
    # This is deliberately outside CORS so Host validation also covers
    # preflights. Local mode has no auth and supports loopback binding only.
    app.add_middleware(
        LocalHTTPBoundaryMiddleware,
        app_origin=settings.APP_URL,
        extension_origins=settings.local_extension_origins,
    )

    # Aggiunto per ULTIMO, quindi e' il piu' esterno: una richiesta senza
    # gettone muore prima di toccare CORS, le rotte e il database. Copre
    # anche `/v1/files/...` e il WebSocket, che sono le due vie per cui i
    # dati escono senza passare dai servizi. `/health` resta anonimo.
    from infra.service_token import ServiceTokenMiddleware, gettone_configurato

    if not gettone_configurato():
        logger.error(
            "LLMWIKI_SERVICE_TOKEN non configurato: le rotte dati "
            "risponderanno 503 finche' non c'e' un gettone",
        )
    app.add_middleware(ServiceTokenMiddleware)

if settings.LOGFIRE_TOKEN:
    import logfire
    logfire.instrument_fastapi(app)

app.include_router(health_router)
app.include_router(me_router)
app.include_router(usage_router)
app.include_router(knowledge_bases_router)
app.include_router(documents_router)
app.include_router(events_router)

if settings.MODE == "local":
    from routes.local_upload import router as local_upload_router
    from routes.files import router as files_router
    from routes.local_graph import router as local_graph_router
    app.include_router(local_upload_router)
    app.include_router(files_router)
    app.include_router(local_graph_router)
    # Nessun `set_workspace_root`: la radice non e' piu' una per processo.
    # La rotta dei file prende quella della KB risolta per la richiesta.
else:
    from routes.api_keys import router as api_keys_router
    from routes.graph import router as graph_router
    from routes.ws import router as ws_router
    from routes.public import router as public_router
    from routes.quiz import router as quiz_router
    from infra.tus import router as tus_router
    app.include_router(api_keys_router)
    app.include_router(tus_router)
    app.include_router(graph_router)
    app.include_router(ws_router)
    app.include_router(public_router)
    app.include_router(quiz_router)
