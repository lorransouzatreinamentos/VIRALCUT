"""
Feedback do usuario (Windows): "ta um tempao aqui e eu nao sei se ta rodando ou
nao" -- faltava porcentagem na barra. E: sem chave da IA ou sem faster-whisper,
o app deve AVISAR na abertura e nao funcionar, em vez de falhar no meio.

Testes: (1) o encanamento de progresso -- a engine emite "PROGRESS N" no stderr
e o callback recebe; (2) o /preflight reporta chave/recursos faltando.
"""
import textwrap

import pytest

from core import transcribe


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """Substitui o local_transcribe.py por um script que emite PROGRESS no stderr
    e um JSON valido no stdout -- testa o Popen/pump sem precisar do faster-whisper."""
    script = tmp_path / "fake_engine.py"
    script.write_text(textwrap.dedent("""
        import json, sys
        for p in (10, 50, 99):
            print(f"PROGRESS {p}", file=sys.stderr, flush=True)
        print(json.dumps({
            "words": [{"id": 0, "text": "oi", "start": 0.0, "end": 0.5}],
            "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "oi", "word_ids": [0]}],
        }))
    """), encoding="utf-8")
    monkeypatch.setattr(transcribe, "_LOCAL_SCRIPT", script)
    return script


def test_progresso_chega_no_callback(fake_engine, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    recebidos = []

    state = transcribe._local_transcribe_sync(str(audio), "pt", on_progress=recebidos.append)

    assert recebidos == [10.0, 50.0, 99.0], "as linhas PROGRESS do stderr nao viraram callbacks"
    assert state.segments[0].text == "oi", "o JSON do stdout nao foi parseado junto"


def test_callback_quebrado_nao_derruba_a_transcricao(fake_engine, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    def cb_ruim(_pct):
        raise RuntimeError("callback bugado")

    state = transcribe._local_transcribe_sync(str(audio), "pt", on_progress=cb_ruim)
    assert state.segments, "erro no callback de progresso nao pode derrubar a engine"


def test_sem_callback_continua_funcionando(fake_engine, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    state = transcribe._local_transcribe_sync(str(audio), "pt")
    assert state.segments


# --- /preflight --------------------------------------------------------------

def test_preflight_reporta_chave_faltando(monkeypatch):
    import core.main as m
    monkeypatch.setattr(m.settings, "openai_api_key", "")
    monkeypatch.setattr(m, "_fw_available", True)  # isola: so a chave falta

    r = m.preflight()

    assert r["ok"] is False
    assert any(p["id"] == "openai_key" for p in r["problems"])
    # a mensagem orienta o conserto, nao e um erro tecnico
    msg = next(p["msg"] for p in r["problems"] if p["id"] == "openai_key")
    assert "instalador" in msg


def test_preflight_reporta_whisper_faltando(monkeypatch):
    import core.main as m
    monkeypatch.setattr(m.settings, "openai_api_key", "sk-ok")
    monkeypatch.setattr(m, "_fw_available", False)

    r = m.preflight()

    assert r["ok"] is False
    assert any(p["id"] == "faster_whisper" for p in r["problems"])


def test_preflight_ok_quando_tudo_configurado(monkeypatch):
    import core.main as m
    monkeypatch.setattr(m.settings, "openai_api_key", "sk-ok")
    monkeypatch.setattr(m, "_fw_available", True)

    r = m.preflight()

    assert r == {"ok": True, "problems": []}
