import os
from pathlib import Path

from realdoc_bench.cli import _env


def test_env_is_noop_without_optin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RDB_ALLOW_DOTENV", raising=False)
    monkeypatch.delenv("SNEAKY_KEY", raising=False)
    Path(".env").write_text("SNEAKY_KEY=leaked\n")
    _env()
    assert "SNEAKY_KEY" not in os.environ


def test_env_loads_with_explicit_optin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RDB_ALLOW_DOTENV", "1")
    monkeypatch.delenv("OPTED_KEY", raising=False)
    Path(".env").write_text("OPTED_KEY=ok\n")
    _env()
    assert os.environ.get("OPTED_KEY") == "ok"
