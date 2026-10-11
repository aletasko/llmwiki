"""Local graph routes — thin HTTP layer, delegates to services/graph.py.

Il grafo si costruisce dai documenti e dai riferimenti fra loro: sta tutto
NELLA connessione della KB, quindi la rotta deve prendere quella risolta e non
l'unica connessione del processo. Con `app.state.sqlite_db` il grafo di una
wiki veniva disegnato con i documenti del legacy.
"""

from typing import Annotated

from deps import get_kb_scope, get_user_id
from fastapi import APIRouter, Depends
from kb_scope import AmbitoKB
from services.graph import get_graph_local, rebuild_local

router = APIRouter(tags=["graph"])


@router.get("/v1/knowledge-bases/{kb_id}/graph")
async def get_kb_graph(
    kb_id: str,
    scope: Annotated[AmbitoKB, Depends(get_kb_scope)],
    user_id: Annotated[str, Depends(get_user_id)],
):
    return await get_graph_local(scope.db, user_id)


@router.post("/v1/knowledge-bases/{kb_id}/graph/rebuild")
async def rebuild_references(
    kb_id: str,
    scope: Annotated[AmbitoKB, Depends(get_kb_scope)],
    user_id: Annotated[str, Depends(get_user_id)],
):
    return await rebuild_local(scope.db, user_id)
