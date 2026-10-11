"""Local service implementations — SQLite + filesystem."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path

import kb_scope
from config import settings
from domain.watcher import mark_written
from fastapi import HTTPException
from kb_scope import AmbitoKB
from infra.db.sqlite import SQLiteChunkRepository, SQLiteDocumentRepository, serialized_write
from services.chunker import chunk_text
from services.webclip_assets import materialize_webclip_assets

from .base import DocumentService, KBService, ServiceFactory, UserService
from .highlight_merge import merge_highlights_by_id
from .parsers import extract_tags, parse_frontmatter, title_from_filename
from .types import MAX_TEXT_CONTENT_BYTES

logger = logging.getLogger(__name__)

# Mirrors the web's lesson definition (wikiTree.ts): wiki pages minus the structural hub/log/index.
_LESSON_FILTER = (
    "source_kind = 'wiki' AND status != 'failed' "
    "AND (path || filename) NOT IN ('/wiki/overview.md', '/wiki/log.md', '/wiki/index.json')"
)


class LocalUserService(UserService):

    def __init__(self, db, user_id: str):
        self.db = db
        self.user_id = user_id

    async def get_profile(self) -> dict:
        return {
            "id": self.user_id,
            "email": "local@localhost",
            "display_name": "Local User",
            "onboarded": True,
        }

    async def complete_onboarding(self) -> None:
        pass

    async def get_usage(self) -> dict:
        cursor = await self.db.execute(
            "SELECT count(*) as doc_count, "
            "COALESCE(SUM(page_count), 0) as total_pages, "
            "COALESCE(SUM(file_size), 0) as total_storage "
            "FROM documents WHERE status != 'failed'",
        )
        row = await cursor.fetchone()
        return {
            "total_pages": row[1] if row else 0,
            "total_storage_bytes": row[2] if row else 0,
            "document_count": row[0] if row else 0,
            "max_pages": 999999,
            "max_storage_bytes": 999999999999,
        }


# I conteggi stanno NELLO stesso database della KB, quindi non hanno nessun
# `WHERE knowledge_base_id`: e' l'isolamento strutturale di kb_scope. Questa
# query si esegue su una connessione per volta, quella della KB giusta.
_QUERY_KB = (
    "SELECT w.id, w.user_id, w.name, w.name as slug, w.description, w.kind, "
    "w.created_at, w.created_at as updated_at, "
    "(SELECT count(*) FROM documents WHERE source_kind = 'source' AND status != 'failed') as source_count, "
    "(SELECT count(*) FROM documents WHERE source_kind = 'wiki' AND status != 'failed') as wiki_page_count, "
    f"(SELECT count(*) FROM documents WHERE {_LESSON_FILTER}) as lesson_count, "
    f"(SELECT count(*) FROM documents WHERE {_LESSON_FILTER} "
    "AND json_extract(metadata, '$.course.status') = 'complete') as lessons_completed "
    "FROM workspace w LIMIT 1"
)

# Le cartelle che una KB nuova deve avere perche' la prima scrittura non
# fallisca: le stesse che il lifespan crea per il legacy.
_CARTELLE_KB = ("wiki", ".llmwiki", ".llmwiki/cache")


def _prepara_cartelle(radice: Path) -> None:
    for relativa in _CARTELLE_KB:
        (radice / relativa).mkdir(parents=True, exist_ok=True)


class LocalKBService(KBService):
    """Le knowledge base come RIGHE DIVERSE, non come la prima del workspace.

    Prima questo servizio leggeva un workspace solo: `get()` ignorava l'id e
    tornava la prima riga, `create()` restituiva l'esistente, `delete()`
    rifiutava sempre. Qui ogni metodo lavora sulla KB CHIESTA, e la radice
    base e' l'unico stato che il servizio tiene.
    """

    def __init__(self, radice_base: Path, user_id: str):
        self.radice_base = Path(radice_base).resolve()
        self.user_id = user_id

    async def _leggi(self, kb_id: str) -> dict | None:
        db = await kb_scope.apri_db(kb_id, self.radice_base)
        cursor = await db.execute(_QUERY_KB)
        riga = await cursor.fetchone()
        if riga is None:
            return None
        colonne = [d[0] for d in cursor.description]
        # L'id della riga combacia con la cartella: lo pretende
        # `kb_scope.apri_db` all'apertura, una volta, invece di «correggerlo»
        # a ogni lettura (vedi `_esigi_identita`).
        return dict(zip(colonne, riga))

    async def list(self) -> list[dict]:
        voci: list[dict] = []
        for kb_id in kb_scope.kb_registrate(self.radice_base):
            try:
                voce = await self._leggi(kb_id)
            except Exception:  # noqa: BLE001 — una KB rotta non deve svuotare l'elenco
                logger.warning("KB %s non leggibile: esclusa dall'elenco", kb_id[:8])
                continue
            if voce is not None:
                voci.append(voce)
        return voci

    async def get(self, kb_id: str) -> dict | None:
        kb = kb_scope.valida_kb_id(kb_id)
        if not kb_scope.kb_pronta(kb, self.radice_base):
            return None
        return await self._leggi(kb)

    async def create(
        self,
        name: str,
        description: str | None,
        kind: str | None = None,
        kb_id: str | None = None,
    ) -> dict:
        """Crea una KB vera: cartella, database, riga di identita'.

        Con `kb_id` e' un UPSERT: il ponte di AGNO manda un id deterministico e
        rimanda lo stesso POST a ogni riconciliazione. Fallire sul secondo giro
        fermerebbe la riconciliazione; ignorare nome e descrizione farebbe
        perdere i rinomini. Quindi: se c'e' si aggiorna, se non c'e' si crea.
        """
        kb, creata = await kb_scope.crea_kb(
            name, self.radice_base, kb_id,
            user_id=self.user_id, descrizione=description, kind=kind,
        )
        if creata:
            await asyncio.to_thread(
                _prepara_cartelle, kb_scope.radice_di(kb, self.radice_base),
            )
        voce = await self._leggi(kb)
        if voce is None:
            raise HTTPException(status_code=500, detail="Knowledge base created without identity row")
        return voce

    async def update(self, kb_id: str, name: str | None, description: str | None, kind: str | None = None) -> dict | None:
        kb = kb_scope.valida_kb_id(kb_id)
        if not kb_scope.kb_pronta(kb, self.radice_base):
            return None
        sets = []
        params: list[object] = []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if kind is not None:
            sets.append("kind = ?")
            params.append(kind)
        if not sets:
            return None
        db = await kb_scope.apri_db(kb, self.radice_base)
        async with serialized_write():
            try:
                # Nessun `WHERE id = ?`: questo database contiene UNA sola riga
                # workspace, e il suo id combacia per forza con la cartella
                # (lo pretende `kb_scope.apri_db`). Il filtro sarebbe rumore.
                await db.execute(f"UPDATE workspace SET {', '.join(sets)}", tuple(params))
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        return await self._leggi(kb)

    async def update_sharing(
        self, kb_id: str, visibility: str, public_slug: str | None,
    ) -> dict | None:
        raise HTTPException(
            status_code=400,
            detail="Wiki sharing isn't available in local mode. Run the hosted "
            "version (or use llmwiki.app) to publish a wiki.",
        )

    async def delete(self, kb_id: str) -> bool:
        """Toglie la KB: connessione chiusa, marcatore via, cartella cancellata.

        Il legacy lo rifiuta `kb_scope.elimina_kb` con un 400: dentro c'e' il
        corpus che esisteva prima del multi-KB.
        """
        return await kb_scope.elimina_kb(kb_id, self.radice_base)


_WEBCLIP_ROOT = "/webclipper/"


def _normalize_webclip_path(path: str | None) -> str:
    missing = path is None or not path.strip()
    raw = _WEBCLIP_ROOT if missing else path.strip()
    if "\\" in raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="Invalid folder path")
    raw = "/" + raw.strip("/") + "/"
    raw = re.sub(r"/+", "/", raw)
    parts = [p for p in raw.split("/") if p]
    if not parts and not missing:
        raise HTTPException(status_code=400, detail="Web clips must be stored under /webclipper/")
    if any(p in {".", ".."} for p in parts):
        raise HTTPException(status_code=400, detail="Invalid folder path")
    normalized = "/" + "/".join(parts) + "/" if parts else _WEBCLIP_ROOT
    if normalized != _WEBCLIP_ROOT and not normalized.startswith(_WEBCLIP_ROOT):
        raise HTTPException(status_code=400, detail="Web clips must be stored under /webclipper/")
    return normalized


def _merge_text_anchors(payloads: list[dict], mapped) -> list[dict]:
    """Same as the hosted helper — merge parser-computed text_anchors back
    onto incoming highlight payloads."""
    out: list[dict] = []
    for i, h in enumerate(payloads):
        if not isinstance(h, dict):
            continue
        entry = dict(h)
        if i < len(mapped) and mapped[i].text_anchor is not None:
            ta = mapped[i].text_anchor
            entry["textAnchor"] = {
                "textStart": ta.text_start,
                "textEnd": ta.text_end,
                "textContent": ta.text_content,
                "prefix": ta.prefix,
                "suffix": ta.suffix,
            }
        out.append(entry)
    return out


def _parse_web_clip(html: str, url: str, highlights: list[dict]):
    """Run the synchronous HTML parser from a worker thread."""
    from html_parser import Parser

    parser = Parser(html, url=url, content_only=True)
    return parser.parse(highlights=highlights)


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _move_file(old_path: Path, new_path: Path) -> None:
    new_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.rename(new_path)


def _unlink_file(path: Path) -> None:
    path.unlink(missing_ok=True)


class LocalDocumentService(DocumentService):

    def __init__(self, db, user_id: str, radice: Path, kb_id: str):
        self.db = db
        self.user_id = user_id
        # La radice della KB di QUESTA richiesta, passata per argomento. Prima
        # si leggeva `settings.WORKSPACE_PATH`: un valore solo per tutto il
        # processo, cioe' tutte le wiki nello stesso albero di file.
        self.radice = Path(radice).resolve()
        self.kb_id = kb_id
        self.doc_repo = SQLiteDocumentRepository(db)
        self.chunk_repo = SQLiteChunkRepository(db)

    def _risolvi(self, relative: str) -> Path:
        """Un percorso relativo dentro la radice di questa KB.

        DUE controlli, non uno:
          - fuori dalla radice -> 400 (un `../` leggerebbe un'altra wiki);
          - dentro una cartella RISERVATA -> 400. Qui sta il buco vero: la
            radice del legacy CONTIENE `kb/<altra-wiki>/`, quindi
            `kb/<altra>/wiki/pagina.md` e' davvero sotto la sua radice e il
            primo controllo lo lascia passare. Vale anche per `.llmwiki/`,
            dove sta l'indice.
        """
        resolved = (self.radice / relative).resolve()
        if not resolved.is_relative_to(self.radice):
            raise HTTPException(status_code=400, detail="Path escapes workspace")
        if kb_scope.e_riservato(resolved, self.radice):
            raise HTTPException(
                status_code=400,
                detail="Path points into a reserved directory (kb/ or .llmwiki/)",
            )
        return resolved

    def _percorso_su_disco(self, doc: dict) -> Path | None:
        relative = (doc["path"].rstrip("/") + "/" + doc["filename"]).lstrip("/")
        resolved = (self.radice / relative).resolve()
        if not resolved.is_relative_to(self.radice):
            return None
        if kb_scope.e_riservato(resolved, self.radice):
            # Una riga che punta a una cartella riservata non si tocca sul
            # disco: cancellarla o riscriverla rovinerebbe l'indice o i file
            # di un'altra wiki. La riga si gestisce, il file no.
            logger.warning(
                "Documento %s punta a un percorso riservato: file non toccato",
                str(doc.get("id"))[:8],
            )
            return None
        return resolved

    async def list(self, kb_id: str, path: str | None = None) -> list[dict]:
        return await self.doc_repo.list_by_kb(kb_id, path=path)

    async def get(self, doc_id: str) -> dict | None:
        return await self.doc_repo.get(doc_id)

    async def get_content(self, doc_id: str) -> dict | None:
        return await self.doc_repo.get_content(doc_id)

    async def get_url(self, doc_id: str) -> dict | None:
        doc = await self.doc_repo.get(doc_id)
        if not doc:
            return None
        # L'URL porta la KB NEL PERCORSO, non in una testata.
        #
        # IL GUASTO CHE QUESTO CHIUDE
        # Prima tornava `/v1/files/<percorso relativo>`. Quel percorso
        # (`wiki/overview.md`, `local/<id>/converted.pdf`) esiste in OGNI
        # wiki: chi seguiva l'URL senza rimettere `X-Wiki-KB` riceveva il file
        # del legacy — con il nome giusto, la dimensione plausibile e nessun
        # errore. Un URL deve bastare da solo.
        api_url = settings.API_URL.rstrip("/")
        prefisso = f"{api_url}/v1/knowledge-bases/{self.kb_id}/files"
        ext = doc["filename"].rsplit(".", 1)[-1].lower() if "." in doc["filename"] else doc.get("file_type", "")

        for check_ext, cache_suffix in [
            ({"pptx", "ppt", "docx", "doc"}, "converted.pdf"),
            ({"html", "htm"}, "tagged.html"),
        ]:
            if ext in check_ext:
                cache_key = f"{doc.get('user_id', 'local')}/{doc['id']}/{cache_suffix}"
                if (self.radice / ".llmwiki" / "cache" / cache_key).is_file():
                    return {"url": f"{prefisso}/{cache_key}"}

        relative = doc.get("relative_path") or (doc["path"].rstrip("/") + "/" + doc["filename"]).lstrip("/")
        return {"url": f"{prefisso}/{relative}"}

    async def create_note(self, kb_id: str, filename: str, path: str, content: str) -> dict:
        if len(content.encode("utf-8")) > MAX_TEXT_CONTENT_BYTES:
            raise HTTPException(status_code=413, detail="Text content exceeds the 10 MiB limit")

        meta = await asyncio.to_thread(parse_frontmatter, content)
        title = meta.get("title", "").strip() or title_from_filename(filename)
        tags = extract_tags(meta)

        existing = await self.doc_repo.find_by_path(kb_id, self.user_id, filename, path)
        if existing:
            raise HTTPException(status_code=409, detail=f"'{filename}' already exists at {path}")

        relative = (path.rstrip("/") + "/" + filename).lstrip("/")
        file_path = self._risolvi(relative)
        mark_written(str(file_path))
        await asyncio.to_thread(_write_text_file, file_path, content or "")

        row = await self.doc_repo.create_note(kb_id, self.user_id, filename, path, title, content, tags)

        if content:
            chunks = await asyncio.to_thread(chunk_text, content)
            await self.chunk_repo.store(str(row["id"]), self.user_id, kb_id, chunks)

        return row

    async def create_web_clip(
        self, kb_id: str, url: str, title: str, html: str,
        highlights: list[dict] | None = None, path: str = "/webclipper/",
    ) -> dict:
        path = _normalize_webclip_path(path)
        result = await asyncio.to_thread(
            _parse_web_clip,
            html,
            url,
            highlights or [],
        )
        markdown = result.content

        base_name = re.sub(r"[^\w\s\-.]", "", title.lower().replace(" ", "-"))[:80].strip("-._")
        filename = f"{base_name or 'web-clip'}.md"

        relative_dir = path.strip("/")
        relative = f"{relative_dir}/{filename}" if relative_dir else filename
        file_path = self._risolvi(relative)
        if file_path.exists():
            base = filename.rsplit(".", 1)[0]
            for i in range(2, 100):
                candidate = f"{base}-{i}.md"
                candidate_relative = f"{relative_dir}/{candidate}" if relative_dir else candidate
                candidate_path = self._risolvi(candidate_relative)
                if not candidate_path.exists():
                    filename = candidate
                    file_path = candidate_path
                    break

        stem = filename.rsplit(".", 1)[0]
        asset_dir_name = f"{stem}.assets"
        markdown, assets = await materialize_webclip_assets(markdown, result.images, asset_dir_name)

        existing = await self.doc_repo.get_by_source_url(url)
        if existing:
            parent_id = str(existing["id"])
            parent_doc = await self.doc_repo.get(parent_id)
            if parent_doc and (parent_doc.get("path") != path or parent_doc.get("title") != title):
                await self.update_metadata(parent_id, {"path": path, "title": title})

            await self.update_content(parent_id, markdown or "")

            asset_metadata: list[dict] = []
            for asset in assets:
                asset_id = str(uuid.uuid4())
                asset.document_id = asset_id
                asset_path = f"{path}{asset_dir_name}/"
                relative_asset = f"{relative_dir}/{asset.src}" if relative_dir else asset.src
                local_asset = self._risolvi(relative_asset)
                mark_written(str(local_asset))
                await asyncio.to_thread(_write_bytes_file, local_asset, asset.data)
                await self.doc_repo.create_asset(
                    asset_id,
                    self.user_id,
                    asset.filename,
                    asset_path,
                    asset.filename,
                    asset.file_type,
                    len(asset.data),
                    {
                        "asset": True,
                        "hidden": True,
                        "parent_document_id": parent_id,
                        "source_url": url,
                        **asset.metadata(),
                    },
                )
                asset_metadata.append(asset.metadata())

            await self.doc_repo.set_metadata_field(parent_id, "source_url", url)
            await self.doc_repo.set_metadata_field(parent_id, "clip_kind", "web")
            await self.doc_repo.set_metadata_field(parent_id, "assets", asset_metadata)

            current = await self.doc_repo.get_highlights(parent_id)
            current_highlights = current["highlights"] if current else []
            enriched = _merge_text_anchors(highlights or [], result.highlights)
            merged_highlights = merge_highlights_by_id(current_highlights, enriched)
            if merged_highlights or current_highlights:
                await self.doc_repo.replace_highlights(parent_id, self.user_id, merged_highlights)

            doc = await self.doc_repo.get(parent_id) or existing
            fresh_highlights = await self.doc_repo.get_highlights(parent_id)
            if fresh_highlights:
                doc["version"] = fresh_highlights["version"]
                doc["highlights"] = fresh_highlights["highlights"]
            return doc

        mark_written(str(file_path))
        await asyncio.to_thread(_write_text_file, file_path, markdown or "")

        try:
            row = await self.doc_repo.create_note(kb_id, self.user_id, filename, path, title, markdown, [])
        except Exception:
            await asyncio.to_thread(_unlink_file, file_path)
            raise

        parent_id = str(row["id"])
        asset_metadata: list[dict] = []
        for asset in assets:
            asset_id = str(uuid.uuid4())
            asset.document_id = asset_id
            asset_path = f"{path}{asset_dir_name}/"
            relative_asset = f"{relative_dir}/{asset.src}" if relative_dir else asset.src
            local_asset = self._risolvi(relative_asset)
            mark_written(str(local_asset))
            await asyncio.to_thread(_write_bytes_file, local_asset, asset.data)
            await self.doc_repo.create_asset(
                asset_id,
                self.user_id,
                asset.filename,
                asset_path,
                asset.filename,
                asset.file_type,
                len(asset.data),
                {
                    "asset": True,
                    "hidden": True,
                    "parent_document_id": parent_id,
                    "source_url": url,
                    **asset.metadata(),
                },
            )
            asset_metadata.append(asset.metadata())

        await self.doc_repo.set_metadata_field(parent_id, "source_url", url)
        await self.doc_repo.set_metadata_field(parent_id, "clip_kind", "web")
        if asset_metadata:
            await self.doc_repo.set_metadata_field(parent_id, "assets", asset_metadata)

        if markdown:
            chunks = await asyncio.to_thread(chunk_text, markdown)
            await self.chunk_repo.store(parent_id, self.user_id, kb_id, chunks)

        # replace_highlights AFTER chunks exist so the repo-level
        # _recompute_chunks_for_doc can materialize annotations onto the
        # freshly-created chunks. Running it before chunks exist would be a
        # no-op materialization.
        if highlights:
            enriched = _merge_text_anchors(highlights, result.highlights)
            await self.doc_repo.replace_highlights(parent_id, self.user_id, enriched)

        doc = await self.doc_repo.get(parent_id) or row
        fresh_highlights = await self.doc_repo.get_highlights(parent_id)
        if fresh_highlights:
            doc["version"] = fresh_highlights["version"]
            doc["highlights"] = fresh_highlights["highlights"]
        return doc

    async def get_by_source_url(self, url: str) -> dict | None:
        return await self.doc_repo.get_by_source_url(url)

    async def get_highlights(self, doc_id: str) -> dict | None:
        return await self.doc_repo.get_highlights(doc_id)

    async def replace_highlights(
        self, doc_id: str, highlights: list[dict],
        expected_version: int | None = None,
    ) -> dict | None:
        return await self.doc_repo.replace_highlights(
            doc_id, self.user_id, highlights, expected_version,
        )

    async def upsert_highlight(
        self, doc_id: str, highlight: dict,
        expected_version: int | None = None,
    ) -> dict | None:
        return await self.doc_repo.upsert_highlight(
            doc_id, self.user_id, highlight, expected_version,
        )

    async def delete_highlight(
        self, doc_id: str, highlight_id: str,
        expected_version: int | None = None,
    ) -> dict | None:
        return await self.doc_repo.delete_highlight(
            doc_id, self.user_id, highlight_id, expected_version,
        )

    async def update_content(self, doc_id: str, content: str) -> dict | None:
        if len(content.encode("utf-8")) > MAX_TEXT_CONTENT_BYTES:
            raise HTTPException(status_code=413, detail="Text content exceeds the 10 MiB limit")

        doc = await self.doc_repo.get(doc_id)
        if not doc:
            return None

        file_path = self._percorso_su_disco(doc)
        if file_path:
            mark_written(str(file_path))
            await asyncio.to_thread(_write_text_file, file_path, content)

        row = await self.doc_repo.update_content(doc_id, self.user_id, content)

        kb_id = await self.doc_repo.get_kb_id(doc_id)
        if kb_id:
            chunks = await asyncio.to_thread(chunk_text, content) if content else []
            await self.chunk_repo.store(doc_id, self.user_id, kb_id, chunks)

        return row

    async def update_metadata(self, doc_id: str, fields: dict) -> dict | None:
        doc = await self.doc_repo.get(doc_id)
        if not doc:
            return None

        # Spostare un documento fra due KB vorrebbe dire muovere il file da una
        # radice all'altra E spostare la riga da un database all'altro: non e'
        # un UPDATE, e' una migrazione. Il campo si scarta invece di far
        # fallire la chiamata, come prima.
        fields = {k: v for k, v in fields.items() if k != "knowledge_base_id"}
        if not fields:
            return doc

        old_path = self._percorso_su_disco(doc)
        needs_move = "filename" in fields or "path" in fields

        if needs_move and old_path and old_path.is_file():
            new_filename = fields.get("filename", doc["filename"])
            new_dir = fields.get("path", doc["path"])
            new_relative = (new_dir.rstrip("/") + "/" + new_filename).lstrip("/")
            new_path = self._risolvi(new_relative)
            mark_written(str(old_path))
            mark_written(str(new_path))
            await asyncio.to_thread(_move_file, old_path, new_path)
            fields["relative_path"] = new_relative
            fields["source_kind"] = "wiki" if new_dir.strip("/").startswith("wiki") else "source"

        return await self.doc_repo.update_metadata(doc_id, self.user_id, **fields)

    async def delete(self, doc_id: str) -> bool:
        doc = await self.doc_repo.get(doc_id)
        if doc:
            file_path = self._percorso_su_disco(doc)
            if file_path and file_path.is_file():
                mark_written(str(file_path))
                await asyncio.to_thread(_unlink_file, file_path)
        return await self.doc_repo.archive(doc_id, self.user_id)

    async def bulk_delete(self, doc_ids: list[str]) -> int:
        for doc_id in doc_ids:
            doc = await self.doc_repo.get(doc_id)
            if doc:
                file_path = self._percorso_su_disco(doc)
                if file_path and file_path.is_file():
                    mark_written(str(file_path))
                    await asyncio.to_thread(_unlink_file, file_path)
        return await self.doc_repo.bulk_archive(doc_ids, self.user_id)


class LocalServiceFactory(ServiceFactory):
    """Costruisce i servizi SULL'AMBITO della richiesta.

    La fabbrica non tiene piu' UNA connessione e UNA radice: tiene la radice
    BASE, e ogni servizio riceve l'ambito (id, radice, connessione) risolto
    per quella richiesta. E' il punto in cui «una wiki per processo» diventa
    «una wiki per richiesta» senza che niente di globale cambi valore.
    """

    def __init__(self, radice_base: Path, user_id: str = ""):
        self.radice_base = Path(radice_base).resolve()
        self.user_id = user_id

    @staticmethod
    def _esigi(scope: AmbitoKB | None) -> AmbitoKB:
        if scope is None:
            # Un servizio locale senza ambito lavorerebbe su una radice
            # indovinata: meglio un errore qui che dati della wiki sbagliata.
            raise HTTPException(
                status_code=500,
                detail="Local services require a resolved knowledge base scope",
            )
        return scope

    def user_service(self, user_id: str, scope: AmbitoKB | None = None) -> LocalUserService:
        return LocalUserService(self._esigi(scope).db, user_id)

    def kb_service(self, user_id: str) -> LocalKBService:
        return LocalKBService(self.radice_base, user_id)

    def document_service(self, user_id: str, scope: AmbitoKB | None = None) -> LocalDocumentService:
        ambito = self._esigi(scope)
        return LocalDocumentService(ambito.db, user_id, ambito.radice, ambito.kb_id)

    def storage_service(self, scope: AmbitoKB | None = None):
        """La cache su disco DELLA KB chiesta. Una cache sola per piu' wiki
        mescolerebbe gli artefatti derivati, che hanno nomi per id documento e
        quindi non collidono per caso — collidono per progetto sbagliato."""
        from infra.storage.local import LocalStorageService

        return LocalStorageService(str(self._esigi(scope).radice), settings.API_URL)

    def public_wiki_service(self):
        raise HTTPException(status_code=404, detail="Public wikis aren't available in local mode.")
