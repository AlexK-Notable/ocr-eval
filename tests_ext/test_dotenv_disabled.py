import os
from pathlib import Path

import pytest

from realdoc_bench.cli import _env


@pytest.fixture(autouse=True)
def _restore_environ():
    """L-a: `_env()` -> `load_dotenv` mutates `os.environ` directly (not via `monkeypatch`), so
    `monkeypatch.setenv`/`delenv`'s own teardown never sees — and never undoes — that mutation.
    Without this, `test_env_loads_with_explicit_optin` leaks a real `OPTED_KEY=ok` into
    `os.environ` for the rest of the test session. Snapshot/restore the whole environment around
    every test in this module regardless of how a var got set."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


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
