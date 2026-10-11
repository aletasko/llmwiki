"""Local file serving route with Range support and path traversal protection.

Only active in local mode. Serves files from the RESOLVED knowledge base:
`<radice KB>/.llmwiki/cache/` (derived artifacts) and the KB root (source
files).

PERCHE' L'AMBITO ANCHE QUI
Questa rotta non ha `{kb_id}` nel percorso: la chiave e' un percorso
relativo, e lo stesso percorso (`wiki/overview.md`) esiste in OGNI wiki. Con
una radice sola per processo serviva sempre il file del legacy, e due wiki con
lo stesso nome di file si leggevano a vicenda senza alcun errore. La KB
arriva dalla testata `X-Wiki-KB`, la stessa che usano le rotte per id di
documento.
"""

import mimetypes
import platform
import subprocess
from pathlib import Path
from typing import Annotated

import kb_scope
from deps import get_kb_scope
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from kb_scope import AmbitoKB
from pydantic import BaseModel

router = APIRouter(tags=["files"])


def _resolve_safe(radice: Path, key: str) -> Path:
    """Resolve a key to a safe path under the KB root. Rejects traversal.

    DUE radici, due controlli diversi:
      - la cache (`.llmwiki/cache/`) deve restare DENTRO LA CACHE, non
        genericamente sotto la radice: un `../index.db` risolto sotto la
        radice passerebbe il controllo largo e servirebbe l'indice;
      - l'albero dei documenti non puo' finire in una cartella riservata. La
        radice del legacy contiene `kb/<altra-wiki>/`, quindi senza questo
        controllo `kb/<altra>/wiki/pagina.md` e' un file «suo» che appartiene
        a un'altra wiki. Nessun errore, il file giusto della wiki sbagliata.
    """
    # Try .llmwiki/cache/ first (derived artifacts), then the KB root (source files)
    cache = (radice / ".llmwiki" / "cache").resolve()
    cache_path = (cache / key).resolve()
    if cache_path.is_file() and cache_path.is_relative_to(cache):
        return cache_path

    root_path = (radice / key).resolve()
    if (
        root_path.is_file()
        and root_path.is_relative_to(radice)
        and not kb_scope.e_riservato(root_path, radice)
    ):
        return root_path

    raise HTTPException(status_code=404, detail="File not found")


@router.get("/v1/knowledge-bases/{kb_id}/files/{key:path}")
async def serve_kb_file(
    kb_id: str,
    key: str,
    request: Request,
    scope: Annotated[AmbitoKB, Depends(get_kb_scope)],
):
    """La forma da usare: la KB sta nel percorso, quindi l'URL basta da solo.

    E' quella che `get_url` restituisce. La forma senza `kb_id` resta per i
    client che mandano `X-Wiki-KB` (il ponte di AGNO), ma un URL che ha
    bisogno di una testata per significare qualcosa non si puo' incollare in
    un `<img src>`."""
    return await _serve(scope, key, request)


@router.get("/v1/files/{key:path}")
async def serve_file(
    key: str,
    request: Request,
    scope: Annotated[AmbitoKB, Depends(get_kb_scope)],
):
    """Forma storica: la KB viene da `X-Wiki-KB`, altrimenti e' il legacy."""
    return await _serve(scope, key, request)


async def _serve(scope: AmbitoKB, key: str, request: Request):
    path = _resolve_safe(scope.radice, key)

    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    file_size = path.stat().st_size
    mtime = path.stat().st_mtime
    etag = f'"{hash((path, mtime, file_size)):x}"'

    # Check If-None-Match for caching
    if_none_match = request.headers.get("if-none-match")
    if if_none_match == etag:
        return StreamingResponse(content=iter([]), status_code=304, headers={"ETag": etag})

    range_header = request.headers.get("range")

    if range_header and range_header.startswith("bytes="):
        ranges = range_header[6:]
        start_str, end_str = ranges.split("-", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def iter_range():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            content=iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
                "ETag": etag,
            },
        )

    return FileResponse(
        path=str(path),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "ETag": etag,
            "Content-Length": str(file_size),
        },
    )


class OpenFileRequest(BaseModel):
    path: str


@router.post("/v1/files/open")
async def open_file_externally(
    body: OpenFileRequest,
    scope: Annotated[AmbitoKB, Depends(get_kb_scope)],
):
    """Open a file in the user's default application."""
    resolved = _resolve_safe(scope.radice, body.path)

    system = platform.system()
    if system == "Darwin":
        cmd = ["open", str(resolved)]
    elif system == "Linux":
        cmd = ["xdg-open", str(resolved)]
    elif system == "Windows":
        cmd = ["start", "", str(resolved)]
    else:
        raise HTTPException(status_code=501, detail=f"Unsupported platform: {system}")

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to open file: {e}")

    return {"status": "opened", "path": body.path}
