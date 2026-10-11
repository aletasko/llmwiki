"""Una knowledge base, una radice, un database — dentro UN processo solo.

IL GUASTO CHE QUESTO FILE CHIUDE
`LocalKBService` lavorava su un workspace unico: `get(kb_id)` IGNORAVA l'id e
tornava `kbs[0]`, `create()` restituiva l'esistente invece di crearne una, e i
conteggi di `list()` facevano `SELECT count(*) FROM documents` senza alcun
filtro — perche' la tabella `documents` NON HA una colonna di knowledge base.
Due wiki sulla stessa istanza finivano negli stessi file e nello stesso
indice: il prefisso dei percorsi separava i nomi, non i dati.

PERCHE' UNA RADICE PER KB E NON UN `kb_id` IN OGNI QUERY
La seconda strada vorrebbe una colonna nuova su `documents` e `chunks`, una
migrazione sui dati vivi, e un `WHERE` in una trentina di punti. L'isolamento
dipenderebbe dal RICORDARSI quel `WHERE`: il giorno che qualcuno aggiunge una
query e lo dimentica, due wiki si mescolano e nessun errore lo dice. Qui
l'isolamento e' STRUTTURALE — due KB non condividono il file, quindi non c'e'
nessun filtro da ricordare, e la revisione si fa su UN punto (questa
risoluzione) invece che su trenta query.

QUELLO CHE NON SI FA, E PERCHE'
Non si muta la configurazione globale per richiesta: un processo solo serve
piu' richieste insieme, e una `settings.WORKSPACE_PATH` riscritta per
richiesta e' una corsa in cui due wiki si leggono a vicenda — senza errore, e
nessun test la vedrebbe. La radice si passa come argomento, sempre.

IL LEGACY
La KB 9efbe904-1b10-455c-aefc-cb6e95e42766 sta nella radice che ha gia' e non
si sposta. Le KB nuove vivono sotto `<radice>/kb/<uuid>/`.

SENZA INDICAZIONE SI RISOLVE SOLO IL LEGACY, MAI «LA PRIMA»: tornare la prima
KB e' il guasto di partenza con un altro nome.
"""
from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

KB_LEGACY = "9efbe904-1b10-455c-aefc-cb6e95e42766"
SOTTOCARTELLA_KB = "kb"
NOME_DB = ".llmwiki/index.db"

_RX_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# una connessione per KB, aperta a richiesta e tenuta: aprire un SQLite a ogni
# chiamata costa, e due connessioni sullo stesso file si contendono il lock
_connessioni: dict[str, object] = {}
_lucchetto = asyncio.Lock()


def valida_kb_id(kb_id: str | None) -> str:
    """Un id di KB e' un UUID minuscolo e nient'altro.

    Serve perche' quell'id diventa un PEZZO DI PERCORSO: senza questo controllo
    un `../` o un nome assoluto uscirebbero dalla radice, e la richiesta
    leggerebbe file che non sono di nessuna wiki.
    """
    valore = (kb_id or "").strip().lower()
    if not _RX_UUID.match(valore):
        raise HTTPException(status_code=400, detail="Invalid knowledge base id")
    return valore


def radice_di(kb_id: str, radice_base: Path) -> Path:
    """La radice sul disco di una KB. Il legacy resta dov'e'."""
    kb = valida_kb_id(kb_id)
    if kb == KB_LEGACY:
        return radice_base.resolve()
    radice = (radice_base / SOTTOCARTELLA_KB / kb).resolve()
    base = radice_base.resolve()
    # il percorso DEVE restare sotto la base: cintura oltre alla validazione
    if base != radice and base not in radice.parents:
        raise HTTPException(status_code=400, detail="Invalid knowledge base id")
    return radice


def percorso_db(kb_id: str, radice_base: Path) -> Path:
    return radice_di(kb_id, radice_base) / NOME_DB


def kb_esistenti(radice_base: Path) -> list[str]:
    """Le KB che esistono DAVVERO sul disco: il legacy piu' le cartelle sotto
    `kb/` che hanno un database. Una cartella senza database non e' una KB —
    e' spazzatura, e offrirla farebbe fallire la prima lettura."""
    trovate = []
    if (radice_base.resolve() / NOME_DB).exists():
        trovate.append(KB_LEGACY)
    cartella = radice_base.resolve() / SOTTOCARTELLA_KB
    if cartella.is_dir():
        for figlia in sorted(cartella.iterdir()):
            if figlia.is_dir() and _RX_UUID.match(figlia.name) and (figlia / NOME_DB).exists():
                trovate.append(figlia.name)
    return trovate


def kb_dalla_richiesta(request, kb_id_percorso: str | None = None) -> str:
    """Quale KB sta chiedendo questa richiesta.

    TRE REGOLE, e la terza e' quella che chiude il buco:
      1. l'id nel percorso vince, se c'e';
      2. altrimenti la testata `X-Wiki-KB`, che il ponte mette dopo aver
         risolto la KB voluta;
      3. senza ne' l'uno ne' l'altra si risolve SOLO il legacy — mai «la
         prima», che e' il guasto di partenza con un altro nome.

    E se percorso e testata DISACCORDANO si rifiuta: uno dei due e' sbagliato,
    e indovinare quale significa servire i dati di una wiki a chi ne ha
    chiesta un'altra.
    """
    testata = (request.headers.get("X-Wiki-KB") or "").strip().lower() or None
    nel_percorso = (kb_id_percorso or "").strip().lower() or None
    if nel_percorso and testata and nel_percorso != testata:
        raise HTTPException(
            status_code=400,
            detail="Knowledge base mismatch between path and X-Wiki-KB header")
    scelto = nel_percorso or testata
    if not scelto:
        return KB_LEGACY
    return valida_kb_id(scelto)


async def db_di(kb_id: str, radice_base: Path):
    """La connessione SQLite di quella KB, aperta se serve.

    Un id che non ha un database sul disco NON si risolve: 404. Prima si
    tornava la prima KB disponibile, che e' il modo in cui una wiki leggeva i
    documenti di un'altra senza che niente lo segnalasse.
    """
    kb = valida_kb_id(kb_id)
    async with _lucchetto:
        if kb in _connessioni:
            return _connessioni[kb]
        percorso = percorso_db(kb, radice_base)
        if not percorso.exists():
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        # Si riusa `create_pool`, la stessa funzione con cui nasce il database
        # del legacy: PRAGMA, schema e backfill sono i suoi. Scriverne una
        # copia qui vorrebbe dire che due database dello stesso prodotto
        # nascono diversi, e la differenza si scoprirebbe mesi dopo.
        from infra.db.sqlite import create_pool  # ritardato: evita un ciclo di import
        conn = await create_pool(str(percorso), init_schema=True)
        _connessioni[kb] = conn
        return conn


async def crea_kb(nome: str, radice_base: Path, kb_id: str | None = None) -> str:
    """Crea la radice e il database di una KB nuova. Idempotente.

    `kb_id` si puo' imporre (il ponte lo fa per avere un id deterministico):
    se quella KB esiste gia' si torna il suo id senza toccare niente, invece
    di fallire — una creazione ripetuta non deve rompere un giro di
    riconciliazione.
    """
    kb = valida_kb_id(kb_id) if kb_id else str(uuid.uuid4())
    radice = radice_di(kb, radice_base)
    percorso = radice / NOME_DB
    if percorso.exists():
        return kb                     # c'e' gia': idempotente
    percorso.parent.mkdir(parents=True, exist_ok=True)
    from infra.db.sqlite import create_pool
    conn = await create_pool(str(percorso), init_schema=True)
    # La riga `workspace` E' la knowledge base per il codice locale: senza, le
    # letture tornerebbero vuote e `list()` non vedrebbe la KB appena creata.
    # L'id lo imponiamo noi, perche' deve combaciare con la cartella: se il
    # database si inventasse un id suo, la stessa KB avrebbe due identita' —
    # una sul disco e una nei dati.
    async with _lucchetto:
        _connessioni[kb] = conn
    try:
        cur = await conn.execute("SELECT id FROM workspace LIMIT 1")
        riga = await cur.fetchone()
        if riga is None:
            await conn.execute(
                "INSERT INTO workspace (id, user_id, name, created_at) "
                "VALUES (?, 'local', ?, datetime('now'))", (kb, nome or kb))
        elif str(riga[0]) != kb:
            await conn.execute("UPDATE workspace SET id = ?, name = COALESCE(?, name)",
                               (kb, nome))
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    return kb


async def elimina_kb(kb_id: str, radice_base: Path, *, consenti_legacy: bool = False) -> bool:
    """Toglie una KB. Idempotente: se non c'era, non e' un errore.

    Il legacy NON si cancella senza un consenso esplicito: dentro c'e' il
    corpus che esisteva prima del multi-KB, e una cancellazione per sbaglio
    non si disfa.
    """
    kb = valida_kb_id(kb_id)
    if kb == KB_LEGACY and not consenti_legacy:
        raise HTTPException(status_code=400,
                            detail="Refusing to delete the legacy knowledge base")
    async with _lucchetto:
        conn = _connessioni.pop(kb, None)
    if conn is not None:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001 — chiudere e' un bonus, non un requisito
            pass
    radice = radice_di(kb, radice_base)
    if not (radice / NOME_DB).exists():
        return False                  # non c'era: idempotente
    import shutil
    shutil.rmtree(radice, ignore_errors=True)
    return True


def fuori_dal_sottoalbero_kb(percorso: Path, radice_base: Path) -> bool:
    """Questo file appartiene alla radice legacy e NON a una KB nuova?

    Serve al watcher e alla riconciliazione del legacy: senza questo controllo
    indicizzerebbero i file delle KB nuove come se fossero suoi, e il corpus
    di una wiki comparirebbe nell'indice di un'altra. E' il guasto peggiore di
    tutto il giro, perche' non da' errore: da' risultati.
    """
    try:
        risolto = percorso.resolve()
        base = radice_base.resolve()
        cartella_kb = base / SOTTOCARTELLA_KB
        return cartella_kb != risolto and cartella_kb not in risolto.parents
    except OSError:
        return True
