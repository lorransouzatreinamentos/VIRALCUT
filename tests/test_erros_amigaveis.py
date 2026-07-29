"""
Bug reportado (Windows): "Montar falas" exibia HTTP 500 mudo. Causa: o rate
limit da OpenAI (429) apos os retries virava httpx.HTTPStatusError, que o
endpoint nao capturava (so RuntimeError) -> 500 sem explicacao na tela.

Estes testes travam as duas camadas do conserto:
1. O retry converte falha final em RuntimeError com mensagem em portugues.
2. O endpoint converte QUALQUER excecao em 502 com detail legivel (nunca 500).
"""
import asyncio

import httpx
import pytest
from fastapi import HTTPException

import core.main as m
import core.objectives as obj
from core.model import Segment, TranscriptState, Word


class _FakeResp:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class _FakeClient:
    """Substitui httpx.AsyncClient: devolve sempre a mesma resposta fake."""

    def __init__(self, resp):
        self._resp = resp

    def __call__(self, *a, **k):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        return self._resp


@pytest.fixture(autouse=True)
def _sleep_instantaneo(monkeypatch):
    """O backoff real espera 20s/40s (janela TPM de 1 min) -- nos testes, zero."""
    async def _noop(_s):
        return None
    monkeypatch.setattr(obj.asyncio, "sleep", _noop)


def test_429_esgotado_vira_mensagem_amigavel(monkeypatch):
    monkeypatch.setattr(obj.httpx, "AsyncClient", _FakeClient(_FakeResp(429)))

    with pytest.raises(RuntimeError, match="Limite de uso da conta OpenAI"):
        asyncio.run(obj._post_openai_with_retry({"model": "x"}))


def test_erro_400_vira_mensagem_com_detalhe(monkeypatch):
    resp = _FakeResp(400, {"error": {"message": "Request too large for gpt-4o"}})
    monkeypatch.setattr(obj.httpx, "AsyncClient", _FakeClient(resp))

    with pytest.raises(RuntimeError, match="Request too large"):
        asyncio.run(obj._post_openai_with_retry({"model": "x"}))


def test_sucesso_apos_429_retenta_e_retorna(monkeypatch):
    """1a tentativa 429, 2a tentativa 200 -> retorna o JSON, sem excecao."""
    respostas = [_FakeResp(429), _FakeResp(200, {"ok": True})]

    class _SeqClient(_FakeClient):
        def __init__(self):
            pass

        async def post(self, *a, **k):
            return respostas.pop(0)

    monkeypatch.setattr(obj.httpx, "AsyncClient", _SeqClient())

    out = asyncio.run(obj._post_openai_with_retry({"model": "x"}))
    assert out == {"ok": True}


def test_endpoint_frankenbite_nunca_vaza_500(monkeypatch):
    """Qualquer excecao (ex: HTTPStatusError do httpx) vira 502 com mensagem --
    era exatamente isso que aparecia como 'HTTP 500' na tela do Windows."""
    m._dv["transcript"] = TranscriptState(
        words=[Word(id=0, text="oi", start=0.0, end=0.5)],
        segments=[Segment(id=0, start=0.0, end=0.5, text="oi", word_ids=[0])],
    )

    async def explode(*_a, **_k):
        raise httpx.HTTPStatusError("429 Too Many Requests", request=None, response=None)

    monkeypatch.setattr(m, "extract_montages", explode)

    with pytest.raises(HTTPException) as e:
        asyncio.run(m.dv_frankenbite(None))

    assert e.value.status_code == 502, "nao pode vazar 500 generico"
    assert "429" in str(e.value.detail)
