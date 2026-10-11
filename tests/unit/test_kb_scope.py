"""La risoluzione della KB, provata sui suoi casi cattivi.

Ogni test qui corrisponde a un modo in cui il multi-KB puo' RIUSCIRE dando i
dati sbagliati: nessuno di questi guasti produce un errore da solo, e per
questo vanno provati invece che riletti.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import kb_scope
import pytest
from fastapi import HTTPException

UTENTE = "e6b612bf-d985-5024-a244-4206e5dd869d"
KB_1 = "11111111-1111-4111-8111-111111111111"
KB_2 = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
async def base(tmp_path):
    """Una radice base con il legacy gia' dentro, e il registro ripulito dopo.

    Il registro di `kb_scope` e' di modulo: senza la pulizia, il test dopo
    riceverebbe le connessioni di questo."""
    from infra.db.sqlite import create_pool

    radice = tmp_path / "base"
    (radice / ".llmwiki").mkdir(parents=True)
    legacy_db = await create_pool(str(radice / ".llmwiki" / "index.db"))
    await legacy_db.execute(
        "INSERT INTO workspace (id, name, user_id) VALUES (?, 'Legacy', ?)",
        (kb_scope.KB_LEGACY, UTENTE),
    )
    await legacy_db.commit()
    kb_scope.registra_legacy(radice, kb_scope.KB_LEGACY)
    kb_scope.registra_connessione(radice, kb_scope.KB_LEGACY, legacy_db)
    yield radice
    await kb_scope.chiudi_tutte(radice)
    kb_scope.scorda(radice)


def richiesta(testate: dict | None = None, radice: Path | None = None):
    return SimpleNamespace(
        headers=testate or {},
        app=SimpleNamespace(state=SimpleNamespace(
            workspace_base=str(radice) if radice else None,
        )),
    )


# ═══════════════════ gli id ═══════════════════

def test_un_id_che_non_e_un_uuid_non_diventa_un_pezzo_di_percorso():
    for cattivo in ["", None, "../../etc", "/absolute", "11111111-1111-1111-1111",
                    "11111111-1111-4111-8111-11111111111G", "  ",
                    "11111111-1111-4111-8111-111111111111/.."]:
        with pytest.raises(HTTPException) as errore:
            kb_scope.valida_kb_id(cattivo)
        assert errore.value.status_code == 400
    # Maiuscolo: si normalizza, non si rifiuta.
    assert kb_scope.valida_kb_id(KB_1.upper()) == KB_1


def test_la_radice_di_una_kb_non_esce_dalla_base(tmp_path):
    base = tmp_path / "b"
    assert kb_scope.radice_di(KB_1, base) == (base / "kb" / KB_1).resolve()
    kb_scope.registra_legacy(base, KB_2)
    assert kb_scope.radice_di(KB_2, base) == base.resolve()
    kb_scope.scorda(base)


# ═══════════════════ la risoluzione ═══════════════════

def test_senza_indicazione_e_il_legacy_mai_la_prima(base):
    assert kb_scope.kb_dalla_richiesta(richiesta(radice=base)) == kb_scope.KB_LEGACY


def test_il_percorso_vince_la_testata_lo_segue_e_il_disaccordo_si_rifiuta(base):
    # solo il percorso
    assert kb_scope.kb_dalla_richiesta(richiesta(radice=base), KB_1) == KB_1
    # solo la testata
    assert kb_scope.kb_dalla_richiesta(
        richiesta({"X-Wiki-KB": KB_1}, base)) == KB_1
    # d'accordo
    assert kb_scope.kb_dalla_richiesta(
        richiesta({"X-Wiki-KB": KB_1}, base), KB_1) == KB_1
    # in disaccordo: 400, non si indovina quale dei due e' sbagliato
    with pytest.raises(HTTPException) as errore:
        kb_scope.kb_dalla_richiesta(richiesta({"X-Wiki-KB": KB_2}, base), KB_1)
    assert errore.value.status_code == 400
    assert "mismatch" in errore.value.detail.lower()


async def test_una_kb_che_non_esiste_non_ripiega_su_nessuna(base):
    with pytest.raises(HTTPException) as errore:
        await kb_scope.apri_db(KB_1, base)
    assert errore.value.status_code == 404


# ═══════════════════ pronta ═══════════════════

async def test_il_database_da_solo_non_basta_serve_il_marcatore(base):
    """Lo stato di una creazione interrotta: il file del database c'e' perche'
    `create_pool` lo crea per primo, ma la riga di identita' no."""
    (base / "kb" / KB_1 / ".llmwiki").mkdir(parents=True)
    (base / "kb" / KB_1 / ".llmwiki" / "index.db").write_bytes(b"")

    assert kb_scope.kb_pronta(KB_1, base) is False
    assert kb_scope.kb_registrate(base) == [kb_scope.KB_LEGACY]
    with pytest.raises(HTTPException) as errore:
        await kb_scope.apri_db(KB_1, base)
    assert errore.value.status_code == 404


async def test_una_kb_creata_e_pronta_e_si_elenca(base):
    kb, creata = await kb_scope.crea_kb("Prima", base, KB_1, user_id=UTENTE)
    assert (kb, creata) == (KB_1, True)
    assert kb_scope.kb_pronta(KB_1, base) is True
    assert kb_scope.kb_registrate(base) == [kb_scope.KB_LEGACY, KB_1]
    assert (base / "kb" / KB_1 / ".llmwiki" / "pronta").read_text() == KB_1


# ═══════════════════ concorrenza ═══════════════════

async def test_otto_creazioni_identiche_insieme_fanno_una_kb_e_una_connessione(base):
    """Il difetto di partenza: il lucchetto si prendeva solo per leggere la
    cache, poi `create_pool` girava fuori. Due richieste sulla stessa KB
    fredda aprivano DUE connessioni, la seconda sovrascriveva la prima in
    cache, e la prima restava aperta per sempre contendendosi il lock."""
    esiti = await asyncio.gather(*[
        kb_scope.crea_kb(f"Tentativo {indice}", base, KB_1, user_id=UTENTE)
        for indice in range(8)
    ])
    assert {kb for kb, _ in esiti} == {KB_1}
    assert sum(1 for _, creata in esiti if creata) == 1, "una sola deve crearla"

    # Una connessione sola in cache, e una riga workspace sola.
    db = await kb_scope.apri_db(KB_1, base)
    cur = await db.execute("SELECT count(*), max(id) FROM workspace")
    conteggio, identita = await cur.fetchone()
    assert (conteggio, identita) == (1, KB_1)
    for _ in range(3):
        assert await kb_scope.apri_db(KB_1, base) is db


async def test_aperture_in_parallelo_tornano_la_stessa_connessione(base):
    await kb_scope.crea_kb("Prima", base, KB_1, user_id=UTENTE)
    kb_scope.scorda(base)       # finge un processo appena riavviato
    kb_scope.registra_legacy(base, kb_scope.KB_LEGACY)
    connessioni = await asyncio.gather(*[
        kb_scope.apri_db(KB_1, base) for _ in range(8)
    ])
    assert len({id(c) for c in connessioni}) == 1


# ═══════════════════ la chiave della cache ═══════════════════

async def test_due_radici_base_con_lo_stesso_id_non_condividono_il_database(tmp_path):
    """La chiave della cache DEVE contenere la radice base: due basi (il
    processo vero e un test, o due montaggi) hanno KB con gli stessi id, e una
    cache con la sola KB nella chiave servirebbe il database dell'una
    all'altra."""
    prima, seconda = tmp_path / "prima", tmp_path / "seconda"
    try:
        await kb_scope.crea_kb("Prima", prima, KB_1, user_id=UTENTE)
        await kb_scope.crea_kb("Seconda", seconda, KB_1, user_id=UTENTE)

        db_prima = await kb_scope.apri_db(KB_1, prima)
        db_seconda = await kb_scope.apri_db(KB_1, seconda)
        assert db_prima is not db_seconda

        await db_prima.execute(
            "INSERT INTO documents (id, user_id, filename, title, path, "
            "relative_path, source_kind, file_type, status, content) VALUES "
            "(?, ?, 'solo-prima.md', 'T', '/', 'solo-prima.md', 'source', 'md', "
            "'ready', 'x')",
            ("33333333-3333-4333-8333-333333333333", UTENTE),
        )
        await db_prima.commit()

        cur = await db_seconda.execute("SELECT count(*) FROM documents")
        assert (await cur.fetchone())[0] == 0, "la seconda base non deve vederlo"
    finally:
        await kb_scope.chiudi_tutte(prima)
        await kb_scope.chiudi_tutte(seconda)
        kb_scope.scorda(prima)
        kb_scope.scorda(seconda)


# ═══════════════════ identita' importata ═══════════════════

async def test_una_cartella_col_database_di_unaltra_wiki_si_rifiuta(base):
    """Il caso dell'importazione: si sposta una cartella sotto `kb/<X>` ma
    dentro c'e' il database di `<Y>`. Ogni controllo di percorso passa, e l'API
    servirebbe i documenti di Y sotto l'id X. Nessun errore: dati di un altro."""
    import shutil

    await kb_scope.crea_kb("Vera", base, KB_1, user_id=UTENTE)
    await kb_scope.chiudi_tutte(base)
    kb_scope.scorda(base)
    kb_scope.registra_legacy(base, kb_scope.KB_LEGACY)

    shutil.copytree(base / "kb" / KB_1, base / "kb" / KB_2)
    assert kb_scope.kb_pronta(KB_2, base) is True, "marcatore e database ci sono"
    with pytest.raises(HTTPException) as errore:
        await kb_scope.apri_db(KB_2, base)
    assert errore.value.status_code == 500
    assert KB_1 in errore.value.detail


# ═══════════════════ cancellazione ═══════════════════

async def test_cancellare_porta_via_tutto_ed_e_idempotente(base):
    await kb_scope.crea_kb("Prima", base, KB_1, user_id=UTENTE)
    (base / "kb" / KB_1 / "wiki").mkdir(parents=True, exist_ok=True)
    (base / "kb" / KB_1 / "wiki" / "pagina.md").write_text("testo")

    assert await kb_scope.elimina_kb(KB_1, base) is True
    assert not (base / "kb" / KB_1).exists()
    assert kb_scope.kb_registrate(base) == [kb_scope.KB_LEGACY]
    # Seconda volta: non c'era, non e' un errore.
    assert await kb_scope.elimina_kb(KB_1, base) is False


async def test_il_legacy_non_si_cancella_senza_consenso(base):
    with pytest.raises(HTTPException) as errore:
        await kb_scope.elimina_kb(kb_scope.KB_LEGACY, base)
    assert errore.value.status_code == 400
    assert (base / ".llmwiki" / "index.db").is_file()


def test_la_cancellazione_dice_cosa_non_ha_cancellato(tmp_path):
    """`ignore_errors=True` tace: una cartella rimasta per un permesso o un
    handle aperto tornerebbe «cancellata», e il giro dopo la KB ricomparirebbe
    senza che niente lo abbia detto."""
    rimasti = kb_scope._rmtree_severo(tmp_path / "non-esiste")
    assert rimasti, "un percorso che non si cancella deve essere RIFERITO"


# ═══════════════════ percorsi riservati ═══════════════════

def test_kb_e_llmwiki_sono_riservati_anche_dietro_un_link(tmp_path):
    radice = tmp_path / "r"
    (radice / "kb" / KB_1).mkdir(parents=True)
    (radice / ".llmwiki").mkdir()
    (radice / "wiki").mkdir()

    assert kb_scope.e_riservato(radice / "kb" / KB_1 / "x.md", radice) is True
    assert kb_scope.e_riservato(radice / ".llmwiki" / "index.db", radice) is True
    assert kb_scope.e_riservato(radice / "wiki" / "x.md", radice) is False
    assert kb_scope.e_riservato(radice.parent / "fuori.md", radice) is True

    # Il nome del link non dice niente: conta il BERSAGLIO, e il controllo si
    # fa sul percorso risolto.
    (radice / "scorciatoia").symlink_to(radice / "kb" / KB_1)
    assert kb_scope.e_riservato(radice / "scorciatoia" / "x.md", radice) is True


def test_il_watcher_della_base_non_entra_nel_sottoalbero_kb(tmp_path):
    from domain import watcher

    radice = tmp_path / "r"
    (radice / "kb" / KB_1 / "wiki").mkdir(parents=True)
    (radice / "wiki").mkdir()

    assert watcher._should_ignore(
        radice / "kb" / KB_1 / "wiki" / "pagina.md", radice, []) is True
    assert watcher._should_ignore(radice / "wiki" / "pagina.md", radice, []) is False
    assert watcher._should_ignore(
        radice / ".llmwiki" / "index.db", radice, []) is True

    # Un link dal legacy che punta dentro un altra KB: il percorso RIFERITO
    # ha come primo segmento «scorciatoia», non «kb». Guardando la stringa
    # chiesta il watcher indicizzerebbe i file di un altra wiki nel proprio
    # indice, e nessun errore lo direbbe.
    (radice / "scorciatoia").symlink_to(radice / "kb" / KB_1)
    assert watcher._should_ignore(
        radice / "scorciatoia" / "pagina.md", radice, []) is True


# ═══════════════════ chiusura ═══════════════════

async def test_in_arresto_si_chiudono_tutte_le_connessioni(tmp_path):
    radice = tmp_path / "r"
    try:
        await kb_scope.crea_kb("Prima", radice, KB_1, user_id=UTENTE)
        await kb_scope.crea_kb("Seconda", radice, KB_2, user_id=UTENTE)
        aperte = [await kb_scope.apri_db(KB_1, radice),
                  await kb_scope.apri_db(KB_2, radice)]

        await kb_scope.chiudi_tutte(radice)

        for conn in aperte:
            with pytest.raises(Exception):
                await conn.execute("SELECT 1")
    finally:
        kb_scope.scorda(radice)
