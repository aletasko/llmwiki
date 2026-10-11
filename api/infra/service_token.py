"""La porta d'ingresso del processo nativo: una testata, su TUTTE le rotte dati.

PERCHE' SERVE
La modalita' locale nasceva per girare su un portatile, in ascolto solo sul
loopback: nessuna autenticazione, e la sicurezza era «solo chi e' seduto qui
puo' parlarci». Qui l'API serve PIU' wiki a un'applicazione che sta su un
altro host: il confine non e' piu' la macchina, e un controllo di Host e
Origin dice da dove arriva una richiesta, non CHI la manda.

COSA COPRE, E PERCHE' NON SOLO LE ROTTE DEI DOCUMENTI
Tutto tranne la salute. I file statici (`/v1/files/...`) e il WebSocket sono
le due vie per cui i dati escono senza passare dai servizi: proteggere solo
le rotte «dati» e lasciare aperta quella che SERVE I FILE e' il modo in cui un
controllo sembra fatto e non lo e'.

`/health` resta anonimo perche' la sonda del container non ha segreti, e una
sonda che fallisce per autenticazione fa riavviare un servizio sano.

OPTIONS passa: il preflight CORS del browser NON porta le testate
applicative, e rifiutarlo renderebbe ogni richiesta del browser impossibile
prima che possa presentare il suo gettone.

SENZA GETTONE CONFIGURATO SI CHIUDE, NON SI APRE
Se `LLMWIKI_SERVICE_TOKEN` e' vuoto ogni rotta dati risponde 503. Il ripiego
opposto — nessun gettone, nessun controllo — e' il guasto che non da' errore:
l'API sembra protetta e serve tutto a tutti.
"""

from __future__ import annotations

import hmac
import os

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

NOME_TESTATA = b"x-wiki-service-token"
NOME_TESTATA_LEGGIBILE = "X-Wiki-Service-Token"
VARIABILE = "LLMWIKI_SERVICE_TOKEN"

# Rotte senza gettone. Un insieme ESATTO, non un prefisso: `/health` e
# `/healthz-mio-endpoint-che-legge-tutto` non si distinguono per prefisso.
PERCORSI_ANONIMI = frozenset({"/health"})


def gettone_configurato() -> str:
    return (os.environ.get(VARIABILE) or "").strip()


class ServiceTokenMiddleware:
    """Pretende `X-Wiki-Service-Token` su tutto tranne `/health`."""

    def __init__(self, app: ASGIApp, token: str | None = None) -> None:
        self.app = app
        # Il valore si legge all'avvio: una lettura per richiesta renderebbe
        # possibile cambiare il segreto senza riavviare, cioe' una finestra in
        # cui meta' delle richieste usa il vecchio e meta' il nuovo.
        self._token = (token if token is not None else gettone_configurato()).strip()

    def _testata(self, scope: Scope) -> str | None:
        for nome, valore in scope.get("headers", []):
            if nome.lower() == NOME_TESTATA:
                return valore.decode("latin-1").strip()
        return None

    def _verdetto(self, scope: Scope) -> tuple[int, str] | None:
        """None se la richiesta puo' passare, altrimenti (codice, motivo)."""
        percorso = scope.get("path", "")
        if percorso in PERCORSI_ANONIMI:
            return None
        if scope["type"] == "http" and scope.get("method", "GET").upper() == "OPTIONS":
            return None
        if not self._token:
            return 503, (
                f"{VARIABILE} is not configured; the API refuses data requests "
                "until a service token is set"
            )
        presentato = self._testata(scope)
        if not presentato or not hmac.compare_digest(presentato, self._token):
            return 401, f"Missing or invalid {NOME_TESTATA_LEGGIBILE}"
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        verdetto = self._verdetto(scope)
        if verdetto is None:
            await self.app(scope, receive, send)
            return

        codice, motivo = verdetto
        if scope["type"] == "websocket":
            # Un WebSocket non ha un corpo di risposta: si rifiuta la stretta
            # di mano. 1008 = violazione di policy.
            await send({"type": "websocket.close", "code": 1008, "reason": motivo[:120]})
            return
        risposta = PlainTextResponse(motivo, status_code=codice)
        await risposta(scope, receive, send)
