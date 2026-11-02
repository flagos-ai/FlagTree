# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Template counter slots are default-off and reach only the calls of the
families a registry names (and the functions that lead to them)."""

import json

import pytest

from triton.flagmega.codegen.triton import tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

pytest.importorskip("torch")


def _gemv():
    from python.test.flagmega.codegen.triton.kernels.block_fp8 import test_amd_native_gguf_gemv
    return test_amd_native_gguf_gemv


def _artifact(tmp_path, name, build):
    gemv = _gemv()
    module = gemv._glu("IQ4_XS", 64, 5120) if build == "glu" else gemv._projection("IQ4_XS", 64, 5120)
    path = gemv._artifact(tmp_path, name, module, "amd-gfx1201")
    return (path / "generated_kernels.py").read_text(), json.loads((path / "artifact.json").read_text())


def _counting(slots, families):
    class Counting(TritonTemplateRegistry):

        def counter_slots(self, platform, architecture):
            return slots

        def counter_families(self, platform, architecture):
            return frozenset(families)

    return Counting


@pytest.mark.parametrize("build", ("glu", "projection"))
def test_default_registry_threads_and_records_no_counters(tmp_path, build):
    source, artifact = _artifact(tmp_path, "default", build)
    assert "_flagmega_counters" not in source
    assert "counter_slots" not in json.dumps(artifact)


def test_counters_reach_only_the_requested_family(tmp_path, monkeypatch):
    monkeypatch.setattr(tir_package, "TritonTemplateRegistry", _counting(2, {"matmul_glu"}))
    source, artifact = _artifact(tmp_path, "glu", "glu")
    text = json.dumps(artifact)
    assert '"counter_slots": 2' in text and '"counter_slot_argument": "_flagmega_counters"' in text
    glu_def = next(line for line in source.splitlines() if line.startswith("def ") and "_glu(" in line)
    assert glu_def.endswith(", _flagmega_counters):")
    other_defs = [line for line in source.splitlines()
                  if line.startswith("def _flagmega_") and "_glu(" not in line and "_flagmega_counters" in line]
    assert other_defs == []


def test_a_module_without_counter_users_allocates_none(tmp_path, monkeypatch):
    monkeypatch.setattr(tir_package, "TritonTemplateRegistry", _counting(1, {"matmul_glu"}))
    source, artifact = _artifact(tmp_path, "projection", "projection")
    assert "_flagmega_counters" not in source and "counter_slots" not in json.dumps(artifact)


def test_negative_counter_request_is_rejected(tmp_path, monkeypatch):
    from triton.flagmega.errors import CodegenError
    monkeypatch.setattr(tir_package, "TritonTemplateRegistry", _counting(-1, {"matmul_glu"}))
    with pytest.raises(CodegenError, match="negative number of counters"):
        _artifact(tmp_path, "negative", "glu")
