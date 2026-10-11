"""Aiuti per i test che lavorano su una KB senza montare tutta l'app.

`kb_scope` tiene un registro DI MODULO (connessioni e id del legacy per
radice base): sopravvive da un test all'altro, quindi un test che registra e
non scorda fa servire al prossimo il database del precedente. Per questo ogni
aiuto qui ha il suo `scorda`, e la fixture `ambito_legacy` lo chiama sempre.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kb_scope
from kb_scope import AmbitoKB


def registra(base: Path, kb_id: str, db) -> AmbitoKB:
    """Dichiara `kb_id` come KB della radice base e ci attacca `db`.

    Serve ai test che hanno gia' una connessione aperta: senza
    `registra_connessione`, `apri_db` ne aprirebbe una seconda sullo stesso
    file e le due si contenderebbero il lock di scrittura.
    """
    base = Path(base).resolve()
    kb_scope.registra_legacy(base, kb_id)
    kb_scope.registra_connessione(base, kb_id, db)
    return AmbitoKB(kb_id=kb_id, radice=base, db=db)


def ambito(base: Path, kb_id: str, db) -> AmbitoKB:
    """Un ambito e nient'altro: nessuna registrazione globale."""
    return AmbitoKB(kb_id=kb_id, radice=Path(base).resolve(), db=db)


def richiesta(
    base: Path,
    *,
    kb_id: str | None = None,
    testate: dict | None = None,
    mode: str = "local",
    db=None,
    pool=None,
) -> SimpleNamespace:
    """Una richiesta finta con quello che le dipendenze leggono davvero:
    testate, parametri di percorso e lo stato dell'app."""
    return SimpleNamespace(
        headers=testate or {},
        path_params={"kb_id": kb_id} if kb_id else {},
        app=SimpleNamespace(state=SimpleNamespace(
            mode=mode,
            pool=pool,
            sqlite_db=db,
            workspace_base=str(Path(base).resolve()),
            auth_provider=None,
        )),
    )


def scorda(base: Path) -> None:
    kb_scope.scorda(Path(base).resolve())
