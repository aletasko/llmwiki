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
La KB che vive nella radice base (in produzione
9efbe904-1b10-455c-aefc-cb6e95e42766) sta dove ha sempre stato e non si
sposta. Le KB nuove vivono sotto `<radice base>/kb/<uuid>/`. L'id del legacy
non e' indovinato: `registra_legacy()` lo PRENDE dalla riga `workspace` del
database base all'avvio, perche' un id scritto a mano qui e una riga diversa
nel database sono la stessa wiki con due identita'.

SENZA INDICAZIONE SI RISOLVE SOLO IL LEGACY, MAI «LA PRIMA»: tornare la prima
KB e' il guasto di partenza con un altro nome.

PRONTA E' UN FILE, NON L'ESISTENZA DEL DATABASE
`create_pool` crea il file del database come PRIMA cosa, poi ci scrive lo
schema e la riga `workspace`. Fra i due istanti il database esiste ed e'
vuoto: elencare una KB in quella finestra significa offrire una wiki che alla
prima lettura non ha nemmeno la sua riga di identita'. Per questo la KB si
considera pronta solo quando esiste `.llmwiki/pronta`, scritto DOPO il commit
della riga. Il legacy non ha marcatore: il suo database esisteva prima di
questa regola e cancellarlo per riscriverlo non e' una migrazione, e' una
perdita.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# L'id che la KB legacy ha in produzione. Resta solo come RIPIEGO per quando
# nessuno ha ancora chiamato `registra_legacy` (test che non montano il
# lifespan): la verita' e' la riga `workspace` del database base.
KB_LEGACY = "9efbe904-1b10-455c-aefc-cb6e95e42766"
SOTTOCARTELLA_KB = "kb"
CARTELLA_INTERNA = ".llmwiki"
NOME_DB = f"{CARTELLA_INTERNA}/index.db"
NOME_PRONTA = f"{CARTELLA_INTERNA}/pronta"

# I primi segmenti che NESSUN percorso di documento puo' avere.
#
# IL GUASTO CHE QUESTO CHIUDE
# La radice del legacy CONTIENE `kb/<uuid>/` — le altre wiki stanno dentro il
# suo albero. Un controllo che chiede solo «il percorso risolto sta sotto la
# mia radice?» dice SI' a `kb/<altra-wiki>/wiki/pagina.md` chiesto dal
# legacy: il file e' davvero sotto la sua radice, ed e' davvero di un'altra
# wiki. Lo stesso vale per `.llmwiki/`, dove stanno l'indice e la cache: un
# documento che si scrive sopra `index.db` non da' errore, cancella l'indice.
SEGMENTI_RISERVATI = frozenset({SOTTOCARTELLA_KB, CARTELLA_INTERNA})

_RX_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

# Una connessione per (radice base, KB), aperta a richiesta e tenuta: aprire un
# SQLite a ogni chiamata costa, e due connessioni sullo stesso file si
# contendono il lock. La radice base E' NELLA CHIAVE: due basi diverse (il
# processo vero e un test con tmp_path) hanno KB con gli stessi id, e una cache
# con la sola KB nella chiave servirebbe il database dell'una all'altra.
_connessioni: dict[tuple[str, str], object] = {}
# Un lucchetto per KB, non uno globale: la creazione tiene il lucchetto per
# TUTTA la sua durata (cartella, pool, riga, marcatore), e con un lucchetto
# globale una creazione lenta fermerebbe ogni altra KB.
_lucchetti: dict[tuple[str, str], asyncio.Lock] = {}
# Protegge i due dizionari qui sopra, nient'altro: non si tiene mai attraverso
# un await lungo.
_lucchetto_registro = asyncio.Lock()
# Id della KB che vive nella radice base, per radice base.
_legacy_per_base: dict[str, str] = {}


@dataclass(frozen=True)
class AmbitoKB:
    """Tutto quello che una richiesta deve sapere della sua KB.

    Si passa per argomento ai servizi: niente stato globale da riscrivere, e
    quindi niente corsa fra due richieste che servono wiki diverse.
    """

    kb_id: str
    radice: Path
    db: object

    @property
    def percorso_db(self) -> Path:
        return self.radice / NOME_DB


def _chiave(base: Path, kb_id: str) -> tuple[str, str]:
    return (str(base.resolve()), kb_id)


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


def registra_legacy(radice_base: Path, kb_id: str) -> str:
    """Dichiara quale id ha la KB che vive NELLA radice base.

    Lo chiama il lifespan dopo aver letto (o creato) la riga `workspace` del
    database base. Senza questo, la risoluzione «senza indicazione» punterebbe
    a un id che in quel database non esiste, e ogni lettura tornerebbe 404 su
    una wiki che c'e'.
    """
    kb = valida_kb_id(kb_id)
    _legacy_per_base[str(radice_base.resolve())] = kb
    return kb


def id_legacy(radice_base: Path) -> str:
    return _legacy_per_base.get(str(radice_base.resolve()), KB_LEGACY)


def e_legacy(kb_id: str, radice_base: Path) -> bool:
    return valida_kb_id(kb_id) == id_legacy(radice_base)


def radice_di(kb_id: str, radice_base: Path) -> Path:
    """La radice sul disco di una KB. Il legacy resta dov'e'."""
    kb = valida_kb_id(kb_id)
    base = radice_base.resolve()
    if kb == id_legacy(radice_base):
        return base
    radice = (base / SOTTOCARTELLA_KB / kb).resolve()
    # il percorso DEVE restare sotto la base: cintura oltre alla validazione
    if base != radice and base not in radice.parents:
        raise HTTPException(status_code=400, detail="Invalid knowledge base id")
    return radice


def percorso_db(kb_id: str, radice_base: Path) -> Path:
    return radice_di(kb_id, radice_base) / NOME_DB


def percorso_pronta(kb_id: str, radice_base: Path) -> Path:
    return radice_di(kb_id, radice_base) / NOME_PRONTA


def kb_pronta(kb_id: str, radice_base: Path) -> bool:
    """Questa KB e' utilizzabile?

    Il legacy: basta il database, non ha marcatore (vedi testata del file).
    Le altre: serve il marcatore, scritto solo dopo il commit della riga di
    identita'. L'esistenza del solo database NON basta — e' esattamente lo
    stato in cui resta una creazione interrotta a meta'.
    """
    kb = valida_kb_id(kb_id)
    radice = radice_di(kb, radice_base)
    if not (radice / NOME_DB).exists():
        return False
    if kb == id_legacy(radice_base):
        return True
    return (radice / NOME_PRONTA).exists()


def kb_registrate(radice_base: Path) -> list[str]:
    """Le KB che esistono DAVVERO e sono pronte: il legacy piu' le cartelle
    sotto `kb/` che hanno database E marcatore. Una cartella a meta' non e' una
    KB — offrirla farebbe fallire la prima lettura."""
    base = radice_base.resolve()
    trovate: list[str] = []
    legacy = id_legacy(radice_base)
    if (base / NOME_DB).exists():
        trovate.append(legacy)
    cartella = base / SOTTOCARTELLA_KB
    if cartella.is_dir():
        for figlia in sorted(cartella.iterdir()):
            if not figlia.is_dir() or not _RX_UUID.match(figlia.name):
                continue
            if figlia.name == legacy:
                continue
            if (figlia / NOME_DB).exists() and (figlia / NOME_PRONTA).exists():
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
        return id_legacy(_radice_dalla_richiesta(request))
    return valida_kb_id(scelto)


def _radice_dalla_richiesta(request) -> Path:
    base = getattr(request.app.state, "workspace_base", None)
    return Path(base).resolve() if base else Path(".").resolve()


async def _lucchetto_per(base: Path, kb_id: str) -> asyncio.Lock:
    chiave = _chiave(base, kb_id)
    async with _lucchetto_registro:
        lucchetto = _lucchetti.get(chiave)
        if lucchetto is None:
            lucchetto = asyncio.Lock()
            _lucchetti[chiave] = lucchetto
        return lucchetto


def registra_connessione(radice_base: Path, kb_id: str, conn) -> None:
    """Mette in cache una connessione aperta da qualcun altro (il lifespan apre
    il database del legacy prima che arrivi una richiesta). Senza questo le
    richieste aprirebbero una SECONDA connessione sullo stesso file, e le due
    si contenderebbero il lock di scrittura."""
    _connessioni[_chiave(radice_base, valida_kb_id(kb_id))] = conn


async def apri_db(kb_id: str, radice_base: Path):
    """La connessione SQLite di quella KB, aperta se serve.

    Un id che non e' PRONTO non si risolve: 404. Prima si tornava la prima KB
    disponibile, che e' il modo in cui una wiki leggeva i documenti di
    un'altra senza che niente lo segnalasse.

    Il lucchetto si tiene per tutta l'apertura, non solo per la lettura della
    cache: due richieste in parallelo sulla stessa KB fredda aprivano DUE
    connessioni e la seconda sovrascriveva la prima in cache — la prima
    restava aperta per sempre, e le due si contendevano il lock.
    """
    kb = valida_kb_id(kb_id)
    conn = _connessioni.get(_chiave(radice_base, kb))
    if conn is not None:
        return conn
    lucchetto = await _lucchetto_per(radice_base, kb)
    async with lucchetto:
        return await _apri_tenendo_il_lucchetto(kb, radice_base)


async def _apri_tenendo_il_lucchetto(kb: str, radice_base: Path):
    """Il corpo di `apri_db`, da chiamare con il lucchetto della KB GIA' preso.

    Esiste perche' `crea_kb` tiene quel lucchetto per tutta la creazione: se
    richiamasse `apri_db` si bloccherebbe su se stessa."""
    chiave = _chiave(radice_base, kb)
    conn = _connessioni.get(chiave)
    if conn is not None:
        return conn
    if not kb_pronta(kb, radice_base):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    # Si riusa `create_pool`, la stessa funzione con cui nasce il database del
    # legacy: PRAGMA, schema e backfill sono i suoi. Scriverne una copia qui
    # vorrebbe dire che due database dello stesso prodotto nascono diversi, e
    # la differenza si scoprirebbe mesi dopo.
    from infra.db.sqlite import create_pool  # ritardato: evita un ciclo di import
    conn = await create_pool(str(percorso_db(kb, radice_base)), init_schema=True)
    try:
        await _esigi_identita(conn, kb)
    except BaseException:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    _connessioni[chiave] = conn
    return conn


async def _esigi_identita(conn, kb: str) -> None:
    """La riga `workspace` deve dire lo STESSO id della cartella.

    Serve a una KB IMPORTATA: una cartella spostata sotto `kb/<uuid>` con
    dentro il database di un'altra wiki passerebbe ogni controllo di percorso
    e servirebbe i suoi documenti sotto l'id sbagliato. Nessun errore, solo
    dati di qualcun altro. Quindi si rifiuta qui, una volta, all'apertura —
    non si «corregge» l'id in lettura, perche' i documenti e gli eventi
    dentro quel database portano comunque l'altro id.
    """
    cur = await conn.execute("SELECT id FROM workspace LIMIT 1")
    riga = await cur.fetchone()
    if riga is None:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge base {kb} has no workspace identity row",
        )
    if str(riga[0]).strip().lower() != kb:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Knowledge base {kb} holds a database whose workspace id is "
                f"{str(riga[0])}; refusing to serve it under the wrong id"
            ),
        )


async def ambito(kb_id: str, radice_base: Path) -> AmbitoKB:
    """La KB risolta: id, radice sul disco, connessione. Niente altro serve a
    una richiesta, e niente di questo e' globale."""
    kb = valida_kb_id(kb_id)
    db = await apri_db(kb, radice_base)
    return AmbitoKB(kb_id=kb, radice=radice_di(kb, radice_base), db=db)


def _mkdir(radice: Path) -> None:
    (radice / NOME_DB).parent.mkdir(parents=True, exist_ok=True)


async def crea_kb(
    nome: str,
    radice_base: Path,
    kb_id: str | None = None,
    *,
    user_id: str,
    descrizione: str | None = None,
    kind: str | None = None,
) -> tuple[str, bool]:
    """Crea la radice e il database di una KB nuova. Torna (id, creata_ora).

    `kb_id` si puo' imporre (il ponte lo fa per avere un id deterministico):
    se quella KB esiste gia' si torna il suo id aggiornando i campi passati,
    invece di fallire — una creazione ripetuta non deve rompere un giro di
    riconciliazione, e un rinomino dal ponte deve arrivare.

    Il lucchetto copre TUTTO: cartella, pool, riga `workspace`, marcatore. Due
    POST identici in parallelo producono una sola KB; il secondo aspetta e
    trova la prima pronta.
    """
    kb = valida_kb_id(kb_id) if kb_id else str(uuid.uuid4())
    lucchetto = await _lucchetto_per(radice_base, kb)
    async with lucchetto:
        radice = radice_di(kb, radice_base)
        if kb_pronta(kb, radice_base):
            conn = await _apri_tenendo_il_lucchetto(kb, radice_base)
            await _scrivi_identita(conn, kb, user_id, nome, descrizione, kind)
            return kb, False

        from infra.db.sqlite import create_pool
        await asyncio.to_thread(_mkdir, radice)
        conn = await create_pool(str(radice / NOME_DB), init_schema=True)
        try:
            await _scrivi_identita(conn, kb, user_id, nome, descrizione, kind)
        except BaseException:
            # Non si lascia in cache una connessione di una KB che non e'
            # nata: la prossima richiesta la riaprirebbe e la troverebbe senza
            # riga di identita'.
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                logger.warning("Chiusura della KB %s fallita dopo un errore", kb[:8])
            raise
        # Il marcatore per ULTIMO: da qui la KB e' elencabile e apribile.
        await asyncio.to_thread((radice / NOME_PRONTA).write_text, kb, encoding="utf-8")
        _connessioni[_chiave(radice_base, kb)] = conn
        return kb, True


async def _scrivi_identita(
    conn, kb: str, user_id: str,
    nome: str | None, descrizione: str | None, kind: str | None,
) -> None:
    """La riga `workspace` E' la knowledge base per il codice locale: senza, le
    letture tornerebbero vuote e `list()` non vedrebbe la KB appena creata.
    L'id lo imponiamo noi, perche' deve combaciare con la cartella: se il
    database si inventasse un id suo, la stessa KB avrebbe due identita' — una
    sul disco e una nei dati."""
    from infra.db.sqlite import serialized_write

    async with serialized_write():
        try:
            cur = await conn.execute("SELECT id FROM workspace LIMIT 1")
            riga = await cur.fetchone()
            if riga is None:
                await conn.execute(
                    # user_id VERO, non la stringa 'local': il controllo di
                    # proprieta' (eventi, grafo) confronta questa riga con
                    # l'utente autenticato, e con 'local' ogni rotta che lo
                    # fa rispondeva 404 su una wiki appena creata.
                    "INSERT INTO workspace (id, user_id, name, description, kind, created_at) "
                    "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                    (kb, user_id, nome or kb, descrizione or "", kind or "wiki"),
                )
            else:
                campi = ["id = ?"]
                valori: list[object] = [kb]
                if nome is not None:
                    campi.append("name = ?")
                    valori.append(nome)
                if descrizione is not None:
                    campi.append("description = ?")
                    valori.append(descrizione)
                if kind is not None:
                    campi.append("kind = ?")
                    valori.append(kind)
                await conn.execute(
                    f"UPDATE workspace SET {', '.join(campi)}", tuple(valori),
                )
            await conn.commit()
        except BaseException:
            try:
                await conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise


def _rmtree_severo(radice: Path) -> list[str]:
    """Cancella e DICE cosa non e' riuscito a cancellare.

    `ignore_errors=True` tace: una cartella rimasta per un permesso o un
    handle aperto tornerebbe «cancellata» e il giro dopo la KB ricomparirebbe
    senza che niente lo abbia detto."""
    rimasti: list[str] = []

    def annota(funzione, percorso, errore):  # noqa: ARG001
        rimasti.append(str(percorso))

    shutil.rmtree(radice, onexc=annota)
    return rimasti


async def elimina_kb(kb_id: str, radice_base: Path, *, consenti_legacy: bool = False) -> bool:
    """Toglie una KB. Idempotente: se non c'era, non e' un errore.

    Il legacy NON si cancella senza un consenso esplicito: dentro c'e' il
    corpus che esisteva prima del multi-KB, e una cancellazione per sbaglio
    non si disfa.

    Ordine: prima si toglie il marcatore (da quell'istante la KB non e' piu'
    elencabile ne' apribile), poi si chiude la connessione, poi si cancella.
    L'ordine opposto lascerebbe una finestra in cui una richiesta apre una KB
    che sta sparendo sotto i piedi.
    """
    kb = valida_kb_id(kb_id)
    if e_legacy(kb, radice_base) and not consenti_legacy:
        raise HTTPException(status_code=400,
                            detail="Refusing to delete the legacy knowledge base")
    lucchetto = await _lucchetto_per(radice_base, kb)
    async with lucchetto:
        radice = radice_di(kb, radice_base)
        esisteva = (radice / NOME_DB).exists()
        marcatore = radice / NOME_PRONTA
        if marcatore.exists():
            await asyncio.to_thread(marcatore.unlink, True)
        conn = _connessioni.pop(_chiave(radice_base, kb), None)
        if conn is not None:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                logger.warning("Chiusura della KB %s fallita in cancellazione", kb[:8])
        if not esisteva:
            return False                  # non c'era: idempotente
        rimasti = await asyncio.to_thread(_rmtree_severo, radice)
        if rimasti:
            raise HTTPException(
                status_code=500,
                detail=f"Knowledge base partially deleted; {len(rimasti)} path(s) remain",
            )
        return True


async def chiudi_tutte(radice_base: Path | None = None) -> None:
    """Chiude le connessioni tenute in cache. La chiama il lifespan in chiusura:
    senza, un riavvio a caldo lascerebbe i file SQLite aperti e il WAL non
    ripiegato."""
    async with _lucchetto_registro:
        if radice_base is None:
            chiavi = list(_connessioni)
        else:
            prefisso = str(radice_base.resolve())
            chiavi = [c for c in _connessioni if c[0] == prefisso]
        connessioni = [(c, _connessioni.pop(c)) for c in chiavi]
        for chiave, _ in connessioni:
            _lucchetti.pop(chiave, None)
    for chiave, conn in connessioni:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001
            logger.warning("Chiusura della KB %s fallita in arresto", chiave[1][:8])


def scorda(radice_base: Path | None = None) -> None:
    """Svuota il registro SENZA chiudere le connessioni.

    Lo usano i test, che aprono e chiudono i database da soli: `chiudi_tutte`
    chiuderebbe una connessione che il test sta ancora usando (o che ha gia'
    chiuso), e la cache sopravvive da un test all'altro perche' e' di modulo —
    cioe' un test servirebbe il database di quello prima.
    """
    if radice_base is None:
        _connessioni.clear()
        _lucchetti.clear()
        _legacy_per_base.clear()
        return
    prefisso = str(radice_base.resolve())
    for chiave in [c for c in _connessioni if c[0] == prefisso]:
        _connessioni.pop(chiave, None)
        _lucchetti.pop(chiave, None)
    _legacy_per_base.pop(prefisso, None)


def e_riservato(risolto: Path, radice: Path) -> bool:
    """Questo percorso GIA' RISOLTO punta a una cartella riservata della KB?

    Si chiama DOPO `resolve()`, non prima: un link simbolico che dalla radice
    del legacy punta dentro `kb/<altra>` ha un nome qualunque e un bersaglio
    riservato. Guardare la stringa chiesta dal client lo lascerebbe passare.
    """
    try:
        relativo = risolto.resolve().relative_to(radice.resolve())
    except ValueError:
        return True          # fuori dalla radice: riservato a maggior ragione
    except OSError:
        return True
    parti = relativo.parts
    return bool(parti) and parti[0] in SEGMENTI_RISERVATI


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
