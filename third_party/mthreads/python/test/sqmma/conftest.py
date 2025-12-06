# conftest.py
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.resolve()


def _cc31():
    try:
        import torch_musa
        major, minor = torch_musa.get_device_capability()
        return 10 * major + minor == 31
    except Exception:
        return False


def pytest_ignore_collect(collection_path, config):
    if _cc31():
        return False
    p = Path(str(collection_path)).resolve()
    return p == ROOT or ROOT in p.parents


@pytest.fixture(autouse=True)
def _sqmma_env_fn(monkeypatch):
    if _cc31():
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")
