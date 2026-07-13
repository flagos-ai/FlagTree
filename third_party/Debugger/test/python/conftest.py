# SPDX-License-Identifier: MIT
import os
import tempfile

import pytest

_MARKERS = {
    "module_a": "FlagTree debugger Module A coverage",
    "module_a_a1": "Module A frontend marker coverage",
    "module_a_a2": "Module A metadata contract coverage",
    "module_a_a3": "Module A launch ABI coverage",
    "module_a_ctt1": "Module A scope validation coverage",
    "module_a_ctt3": "Module A hidden-argument contract coverage",
    "module_a_smoke": "Module A compile-and-run smoke coverage",
}


def pytest_configure(config):
    for name, description in _MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


@pytest.fixture
def fresh_triton_cache():
    previous = os.environ.get("TRITON_CACHE_DIR")
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["TRITON_CACHE_DIR"] = tmpdir
        try:
            yield tmpdir
        finally:
            if previous is None:
                os.environ.pop("TRITON_CACHE_DIR", None)
            else:
                os.environ["TRITON_CACHE_DIR"] = previous
