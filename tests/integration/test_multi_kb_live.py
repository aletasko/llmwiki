"""Multi-KB visto da fuori: un processo vero, richieste vere, file veri.

PERCHE' UN PROCESSO E NON UN `TestClient`
Quello che questa parte deve dimostrare e' che DUE richieste in volo insieme,
su due wiki diverse, non si leggono a vicenda. Un test che chiama le funzioni
dei servizi passandogli l'ambito giusto non puo' dimostrarlo: l'ambito
sbagliato e' esattamente il guasto che cerchiamo. Qui l'API gira come gira in
produzione — uvicorn, i middleware, il gettone, la risoluzione della KB dalla
testata — e i file si guardano sul disco.

Si salta se uvicorn o httpx non ci sono (l'immagine dell'API li ha entrambi).
"""

from __future__ import annotations

import asyncio
import os
import socket
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

GETTONE = "gettone-di-prova-abbastanza-lungo"
KB_A = "aaaaaaaa-0000-4000-8000-000000000001"
KB_B = "bbbbbbbb-0000-4000-8000-000000000002"
KB_IGNOTA = "cccccccc-0000-4000-8000-000000000003"
API_DIR = Path(__file__).resolve().parents[2] / "api"


def _porta_libera() -> int:
    with socket.socket() as presa:
        presa.bind(("127.0.0.1", 0))
        return presa.getsockname()[1]


def _db_kb(base: Path, kb_id: str | None = None) -> Path:
    radice = base if kb_id is None else base / "kb" / kb_id
    return radice / ".llmwiki" / "index.db"


def _righe(percorso: Path, sql: str, parametri: tuple = ()) -> list[tuple]:
    # Connessione di sola lettura e di passaggio: il processo dell'API tiene la
    # sua, e aprirne una seconda in scrittura si contenderebbe il lock.
    conn = sqlite3.connect(f"file:{percorso}?mode=ro", uri=True)
    try:
        return list(conn.execute(sql, parametri))
    finally:
        conn.close()


class Api:
    """L'API accesa, piu' un client che sa presentare il gettone."""

    def __init__(self, processo, base: Path, porta: int):
        self.processo = processo
        self.base = base
        self.url = f"http://127.0.0.1:{porta}"

    def client(self, *, gettone: str | None = GETTONE) -> httpx.AsyncClient:
        testate = {} if gettone is None else {"X-Wiki-Service-Token": gettone}
        return httpx.AsyncClient(base_url=self.url, headers=testate, timeout=30.0)


@pytest.fixture
async def api(tmp_path):
    base = tmp_path / "wiki-base"
    base.mkdir()
    porta = _porta_libera()
    ambiente = {
        **os.environ,
        "MODE": "local",
        "WORKSPACE_PATH": str(base),
        "LLMWIKI_SERVICE_TOKEN": GETTONE,
        "APP_URL": "http://localhost:3000",
        "API_URL": f"http://127.0.0.1:{porta}",
        "DATABASE_URL": "",
        "SUPABASE_URL": "",
        "AWS_ACCESS_KEY_ID": "",
        "S3_BUCKET": "",
        "LOGFIRE_TOKEN": "",
        "SENTRY_DSN": "",
    }
    processo = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", str(porta), "--log-level", "warning"],
        cwd=str(API_DIR),
        env=ambiente,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    acceso = Api(processo, base, porta)
    try:
        await _aspetta_salute(acceso)
        yield acceso
    finally:
        processo.terminate()
        try:
            processo.wait(timeout=20)
        except subprocess.TimeoutExpired:
            processo.kill()
            processo.wait(timeout=10)


async def _aspetta_salute(acceso: Api, secondi: float = 30.0) -> None:
    scaduto = asyncio.get_running_loop().time() + secondi
    async with httpx.AsyncClient(base_url=acceso.url, timeout=2.0) as client:
        while asyncio.get_running_loop().time() < scaduto:
            if acceso.processo.poll() is not None:
                uscita = acceso.processo.stdout.read().decode("utf-8", "replace")
                raise RuntimeError(f"l'API e' morta all'avvio:\n{uscita}")
            try:
                risposta = await client.get("/health")
            except httpx.HTTPError:
                await asyncio.sleep(0.2)
                continue
            if risposta.status_code == 200:
                return
            await asyncio.sleep(0.2)
    raise RuntimeError("l'API non ha risposto a /health")


async def _crea_kb(client, kb_id: str, nome: str, descrizione: str = "") -> dict:
    risposta = await client.post(
        "/v1/knowledge-bases",
        json={"id": kb_id, "name": nome, "description": descrizione},
    )
    assert risposta.status_code == 201, risposta.text
    return risposta.json()


async def _crea_nota(client, kb_id: str, filename: str, contenuto: str, path: str = "/"):
    return await client.post(
        f"/v1/knowledge-bases/{kb_id}/documents/note",
        json={"filename": filename, "path": path, "content": contenuto},
    )


# ════════════════════════ il gettone ════════════════════════

async def test_salute_anonima_ma_i_dati_pretendono_il_gettone(api):
    async with httpx.AsyncClient(base_url=api.url, timeout=10.0) as nudo:
        assert (await nudo.get("/health")).status_code == 200
        # Nessun gettone: la porta chiude PRIMA delle rotte, quindi vale anche
        # per i file statici, che sono l'altra via per cui i dati escono.
        assert (await nudo.get("/v1/knowledge-bases")).status_code == 401
        assert (await nudo.get("/v1/files/wiki/overview.md")).status_code == 401
        assert (await nudo.get("/v1/me")).status_code == 401
        sbagliato = await nudo.get(
            "/v1/knowledge-bases",
            headers={"X-Wiki-Service-Token": GETTONE + "x"},
        )
        assert sbagliato.status_code == 401

    async with api.client() as client:
        assert (await client.get("/v1/knowledge-bases")).status_code == 200


# ════════════════════════ creazione e idempotenza ════════════════════════

async def test_due_kb_vere_e_un_secondo_post_che_non_duplica(api):
    async with api.client() as client:
        iniziali = (await client.get("/v1/knowledge-bases")).json()
        assert len(iniziali) == 1, "all'avvio c'e' solo il legacy"
        legacy = iniziali[0]["id"]

        prima = await _crea_kb(client, KB_A, "Wiki A", "la prima")
        assert prima["id"] == KB_A
        await _crea_kb(client, KB_B, "Wiki B")

        # Lo stesso POST una seconda volta: il ponte lo rimanda a ogni
        # riconciliazione. Non deve nascere una terza wiki, e il nome nuovo
        # deve arrivare.
        di_nuovo = await _crea_kb(client, KB_A, "Wiki A rinominata")
        assert di_nuovo["id"] == KB_A
        assert di_nuovo["name"] == "Wiki A rinominata"

        elenco = (await client.get("/v1/knowledge-bases")).json()
        assert sorted(v["id"] for v in elenco) == sorted([legacy, KB_A, KB_B])

        # Ognuna ha la sua radice e il suo database.
        assert _db_kb(api.base, KB_A).is_file()
        assert _db_kb(api.base, KB_B).is_file()
        assert (api.base / "kb" / KB_A / ".llmwiki" / "pronta").is_file()


async def test_una_kb_a_meta_non_si_elenca_e_non_si_apre(api):
    """Una cartella con il database ma senza marcatore e' una creazione
    interrotta: offrirla farebbe fallire la prima lettura."""
    meta = api.base / "kb" / KB_IGNOTA / ".llmwiki"
    meta.mkdir(parents=True)
    (meta / "index.db").write_bytes(b"")
    async with api.client() as client:
        elenco = (await client.get("/v1/knowledge-bases")).json()
        assert KB_IGNOTA not in [v["id"] for v in elenco]
        assert (await client.get(f"/v1/knowledge-bases/{KB_IGNOTA}")).status_code == 404
        risposta = await _crea_nota(client, KB_IGNOTA, "x.md", "niente")
        assert risposta.status_code == 404


async def test_kb_ignota_non_ripiega_sul_legacy(api):
    async with api.client() as client:
        ignota = f"dddddddd-0000-4000-8000-{'0' * 11}9"
        assert (await client.get(f"/v1/knowledge-bases/{ignota}")).status_code == 404
        documenti = await client.get(f"/v1/knowledge-bases/{ignota}/documents")
        assert documenti.status_code == 404
        # id malformato: 400, non 404 e non «la prima»
        assert (await client.get("/v1/knowledge-bases/non-un-uuid")).status_code == 422


async def test_percorso_e_testata_in_disaccordo_si_rifiutano(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")
        risposta = await client.get(
            f"/v1/knowledge-bases/{KB_A}/documents",
            headers={"X-Wiki-KB": KB_B},
        )
        assert risposta.status_code == 400
        assert "mismatch" in risposta.text.lower()
        # D'accordo: passa.
        uguale = await client.get(
            f"/v1/knowledge-bases/{KB_A}/documents",
            headers={"X-Wiki-KB": KB_A},
        )
        assert uguale.status_code == 200


# ════════════════ isolamento sotto concorrenza ════════════════

async def test_stesso_nome_contenuto_diverso_nello_stesso_istante(api):
    """Il caso che il guasto di partenza non superava: due wiki, lo STESSO
    nome di file, contenuti diversi, nello stesso momento."""
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")

        testo_a = "# Pagina di A\n\nA ha scritto questo e solo questo.\n"
        testo_b = "# Pagina di B\n\nB ha scritto altro, nello stesso istante.\n"
        risposta_a, risposta_b = await asyncio.gather(
            _crea_nota(client, KB_A, "condivisa.md", testo_a),
            _crea_nota(client, KB_B, "condivisa.md", testo_b),
        )
        assert risposta_a.status_code == 201, risposta_a.text
        assert risposta_b.status_code == 201, risposta_b.text
        doc_a, doc_b = risposta_a.json(), risposta_b.json()
        assert doc_a["id"] != doc_b["id"]

        # I FILE: due alberi diversi, e niente nella radice del legacy.
        file_a = api.base / "kb" / KB_A / "condivisa.md"
        file_b = api.base / "kb" / KB_B / "condivisa.md"
        assert file_a.read_text(encoding="utf-8") == testo_a
        assert file_b.read_text(encoding="utf-8") == testo_b
        assert not (api.base / "condivisa.md").exists()

        # LE RIGHE: ogni indice vede un documento solo.
        for kb_id, atteso in ((KB_A, testo_a), (KB_B, testo_b)):
            righe = _righe(
                _db_kb(api.base, kb_id),
                "SELECT relative_path, content FROM documents WHERE filename = ?",
                ("condivisa.md",),
            )
            assert righe == [("condivisa.md", atteso)]

        # L'INDICE DEL LEGACY non ha preso niente: il suo watcher guarda la
        # radice base, che CONTIENE kb/, e senza l'esclusione avrebbe
        # indicizzato i file delle altre due.
        await asyncio.sleep(2.0)
        legacy = _righe(
            _db_kb(api.base),
            "SELECT count(*) FROM documents WHERE filename = ?",
            ("condivisa.md",),
        )
        assert legacy == [(0,)]

        # LETTURA incrociata: per id di documento la KB arriva dalla testata.
        letto_a = await client.get(
            f"/v1/documents/{doc_a['id']}/content",
            headers={"X-Wiki-KB": KB_A},
        )
        assert letto_a.json()["content"] == testo_a
        # lo stesso id chiesto all'altra wiki non esiste: 404, non il suo omonimo
        incrociato = await client.get(
            f"/v1/documents/{doc_a['id']}/content",
            headers={"X-Wiki-KB": KB_B},
        )
        assert incrociato.status_code == 404


async def test_elenco_lettura_aggiornamento_e_cancellazione_restano_separati(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")
        # Testi lunghi a sufficienza: il chunker non produce pezzi da tre
        # parole, e un test che misura zero pezzi su un testo cortissimo
        # misura il chunker, non l'isolamento.
        testo_a = "Pagina di A. " + ("A racconta la sua storia per intero. " * 40)
        testo_b = "Pagina di B. " + ("B racconta un'altra storia diversa. " * 40)
        a = (await _crea_nota(client, KB_A, "solo-a.md", testo_a)).json()
        b = (await _crea_nota(client, KB_B, "solo-b.md", testo_b)).json()

        elenco_a = (await client.get(f"/v1/knowledge-bases/{KB_A}/documents")).json()
        elenco_b = (await client.get(f"/v1/knowledge-bases/{KB_B}/documents")).json()
        assert [d["filename"] for d in elenco_a] == ["solo-a.md"]
        assert [d["filename"] for d in elenco_b] == ["solo-b.md"]

        # i pezzi per la ricerca stanno nel database della propria KB
        for kb_id, doc in ((KB_A, a), (KB_B, b)):
            pezzi = _righe(
                _db_kb(api.base, kb_id),
                "SELECT count(*) FROM document_chunks WHERE document_id = ?",
                (doc["id"],),
            )
            assert pezzi[0][0] >= 1, f"KB {kb_id[:8]} dovrebbe avere i suoi pezzi"
            altrove = _righe(
                _db_kb(api.base, KB_B if kb_id == KB_A else KB_A),
                "SELECT count(*) FROM document_chunks WHERE document_id = ?",
                (doc["id"],),
            )
            assert altrove == [(0,)], "i pezzi non devono finire nell'altra wiki"

        # aggiornamento: tocca solo la sua
        nuovo = "contenuto di A, versione due"
        aggiornato = await client.put(
            f"/v1/documents/{a['id']}/content",
            json={"content": nuovo},
            headers={"X-Wiki-KB": KB_A},
        )
        assert aggiornato.status_code == 200, aggiornato.text
        assert (api.base / "kb" / KB_A / "solo-a.md").read_text() == nuovo
        assert (api.base / "kb" / KB_B / "solo-b.md").read_text() == testo_b

        # cancellazione: tocca solo la sua
        cancellato = await client.delete(
            f"/v1/documents/{a['id']}", headers={"X-Wiki-KB": KB_A},
        )
        assert cancellato.status_code == 204
        assert not (api.base / "kb" / KB_A / "solo-a.md").exists()
        assert (api.base / "kb" / KB_B / "solo-b.md").is_file()
        residuo_b = (await client.get(f"/v1/knowledge-bases/{KB_B}/documents")).json()
        assert [d["filename"] for d in residuo_b] == ["solo-b.md"]


# ════════════════ il legacy ════════════════

async def test_senza_indicazione_e_sempre_il_legacy_e_resta_nella_sua_radice(api):
    async with api.client() as client:
        legacy = (await client.get("/v1/knowledge-bases")).json()[0]["id"]
        await _crea_kb(client, KB_A, "Wiki A")

        # Nessun percorso, nessuna testata: il legacy. Mai «la prima», che
        # dopo una creazione sarebbe un'altra.
        nota = await _crea_nota(client, legacy, "nota-legacy.md", "sta nella base")
        assert nota.status_code == 201
        assert (api.base / "nota-legacy.md").read_text() == "sta nella base"
        assert not (api.base / "kb" / KB_A / "nota-legacy.md").exists()

        senza_testata = await client.get("/v1/usage")
        assert senza_testata.status_code == 200
        assert senza_testata.json()["document_count"] >= 1

        # Il legacy non si cancella: dentro c'e' il corpus di prima.
        rifiutata = await client.delete(f"/v1/knowledge-bases/{legacy}")
        assert rifiutata.status_code == 400
        assert _db_kb(api.base).is_file()


async def test_il_legacy_non_puo_scrivere_nellalbero_di_unaltra_wiki(api):
    """La radice del legacy CONTIENE kb/<altra>/: un controllo che chiede solo
    «sta sotto la mia radice?» dice si' a un percorso che e' di un'altra."""
    async with api.client() as client:
        legacy = (await client.get("/v1/knowledge-bases")).json()[0]["id"]
        await _crea_kb(client, KB_A, "Wiki A")

        dentro_altra = await _crea_nota(
            client, legacy, "rubata.md", "non deve arrivare",
            path=f"/kb/{KB_A}/",
        )
        assert dentro_altra.status_code == 400, dentro_altra.text
        assert not (api.base / "kb" / KB_A / "rubata.md").exists()

        dentro_interna = await _crea_nota(
            client, legacy, "index.db", "sovrascrivo l'indice", path="/.llmwiki/",
        )
        assert dentro_interna.status_code == 400
        assert _righe(_db_kb(api.base), "SELECT count(*) FROM workspace") == [(1,)]

        fuori = await _crea_nota(client, legacy, "fuga.md", "x", path="/../")
        assert fuori.status_code == 400
        assert not (api.base.parent / "fuga.md").exists()


async def test_un_link_simbolico_verso_unaltra_wiki_non_serve_i_suoi_file(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        segreto = "contenuto che solo A deve servire"
        await _crea_nota(client, KB_A, "privata.md", segreto)

        # Un link dal legacy che punta dentro la cartella di A. Il nome non
        # dice niente: il controllo si fa sul percorso RISOLTO.
        (api.base / "scorciatoia").symlink_to(api.base / "kb" / KB_A)
        risposta = await client.get("/v1/files/scorciatoia/privata.md")
        assert risposta.status_code == 404, risposta.text


# ════════════════ gli URL dei file ════════════════

async def test_lurl_di_un_documento_porta_la_kb_e_serve_il_file_giusto(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")
        testo_a = "il file di A"
        testo_b = "il file di B"
        a = (await _crea_nota(client, KB_A, "omonimo.md", testo_a)).json()
        await _crea_nota(client, KB_B, "omonimo.md", testo_b)

        url = (await client.get(
            f"/v1/documents/{a['id']}/url", headers={"X-Wiki-KB": KB_A},
        )).json()["url"]
        assert f"/v1/knowledge-bases/{KB_A}/files/" in url, url

        # Seguito NUDO, senza rimettere la testata: deve servire il file di A.
        # Prima tornava `/v1/files/omonimo.md`, che senza testata serviva il
        # legacy — con il nome giusto e nessun errore.
        percorso = url.split(api.url, 1)[1]
        servito = await client.get(percorso)
        assert servito.status_code == 200
        assert servito.text == testo_a

        gemello = percorso.replace(KB_A, KB_B)
        assert (await client.get(gemello)).text == testo_b


# ════════════════ cancellazione di una KB ════════════════

async def test_cancellare_una_kb_porta_via_radice_e_indice(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")
        await _crea_nota(client, KB_A, "da-cancellare.md", "testo")

        cancellata = await client.delete(f"/v1/knowledge-bases/{KB_A}")
        assert cancellata.status_code == 204
        assert not (api.base / "kb" / KB_A).exists()

        elenco = (await client.get("/v1/knowledge-bases")).json()
        assert KB_A not in [v["id"] for v in elenco]
        assert KB_B in [v["id"] for v in elenco]
        assert (await client.get(f"/v1/knowledge-bases/{KB_A}")).status_code == 404

        # Idempotente: una seconda cancellazione non e' un errore interno.
        di_nuovo = await client.delete(f"/v1/knowledge-bases/{KB_A}")
        assert di_nuovo.status_code == 404

        # E si puo' ricreare con lo stesso id senza portarsi dietro i documenti.
        await _crea_kb(client, KB_A, "Wiki A di nuovo")
        documenti = (await client.get(f"/v1/knowledge-bases/{KB_A}/documents")).json()
        assert documenti == []


async def test_una_kb_con_lidentita_sbagliata_non_si_serve(api):
    """Una cartella `kb/<X>` con dentro il database di un'altra wiki passerebbe
    ogni controllo di percorso e servirebbe i suoi documenti sotto l'id
    sbagliato. Non e' un errore: sono dati di qualcun altro."""
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_nota(client, KB_A, "di-a.md", "roba di A")

    # L'API non ha ancora mai aperto KB_B: si importa a mano la cartella di A
    # sotto il nome di B, marcatore compreso.
    import shutil

    shutil.copytree(api.base / "kb" / KB_A, api.base / "kb" / KB_B)

    async with api.client() as client:
        risposta = await client.get(f"/v1/knowledge-bases/{KB_B}/documents")
        assert risposta.status_code == 500, risposta.text
        assert "workspace id" in risposta.text or risposta.status_code == 500
        # E non compare nemmeno nell'elenco.
        elenco = (await client.get("/v1/knowledge-bases")).json()
        assert KB_B not in [v["id"] for v in elenco]


async def test_gli_eventi_di_una_wiki_non_sono_quelli_dellaltra(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")
        await _crea_nota(client, KB_A, "evento-a.md", "A")

        eventi_a = (await client.get(f"/v1/knowledge-bases/{KB_A}/events")).json()
        eventi_b = (await client.get(f"/v1/knowledge-bases/{KB_B}/events")).json()
        percorsi_a = [e["subject_path"] or "" for e in eventi_a["items"]]
        percorsi_b = [e["subject_path"] or "" for e in eventi_b["items"]]
        assert any("evento-a.md" in p for p in percorsi_a), percorsi_a
        assert not any("evento-a.md" in p for p in percorsi_b), percorsi_b


async def test_il_conteggio_duso_segue_la_kb_chiesta(api):
    async with api.client() as client:
        await _crea_kb(client, KB_A, "Wiki A")
        await _crea_kb(client, KB_B, "Wiki B")
        await _crea_nota(client, KB_A, "uno.md", "x")
        await _crea_nota(client, KB_A, "due.md", "y")
        await _crea_nota(client, KB_B, "solo-uno.md", "z")

        uso_a = (await client.get("/v1/usage", headers={"X-Wiki-KB": KB_A})).json()
        uso_b = (await client.get("/v1/usage", headers={"X-Wiki-KB": KB_B})).json()
        assert uso_a["document_count"] == 2
        assert uso_b["document_count"] == 1

        elenco = (await client.get("/v1/knowledge-bases")).json()
        per_id = {v["id"]: v for v in elenco}
        assert per_id[KB_A]["source_count"] == 2
        assert per_id[KB_B]["source_count"] == 1


async def test_creazioni_identiche_in_parallelo_fanno_una_sola_kb(api):
    """Due POST con lo stesso id nello stesso istante: il lucchetto per KB
    tiene per TUTTA la creazione, quindi non nascono due database ne' due
    connessioni in cache sullo stesso file."""
    async with api.client() as client:
        risposte = await asyncio.gather(*[
            client.post(
                "/v1/knowledge-bases",
                json={"id": KB_A, "name": f"Tentativo {indice}"},
            )
            for indice in range(6)
        ])
        assert {r.status_code for r in risposte} == {201}, [r.text for r in risposte]
        assert {r.json()["id"] for r in risposte} == {KB_A}

        elenco = (await client.get("/v1/knowledge-bases")).json()
        assert [v["id"] for v in elenco].count(KB_A) == 1
        cartelle = sorted(p.name for p in (api.base / "kb").iterdir())
        assert cartelle == [KB_A]
        # Un database solo, e con una riga workspace sola.
        assert _righe(_db_kb(api.base, KB_A), "SELECT count(*) FROM workspace") == [(1,)]
