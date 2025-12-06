# conftest.py
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.resolve()


def _cc22():
    try:
        import torch_musa
        major, minor = torch_musa.get_device_capability()
        return 10 * major + minor == 22
    except Exception:
        return False


def pytest_ignore_collect(collection_path, config):
    if _cc22():
        return False
    p = Path(str(collection_path)).resolve()
    return p == ROOT or ROOT in p.parents


@pytest.fixture(autouse=True)
def _wmma_env_fn(monkeypatch):
    if _cc22():
        monkeypatch.setenv("ENABLE_MUSA_MMA", "1")
