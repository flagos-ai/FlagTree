# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import cli
from triton.flagmega.errors import IRSchemaError
from triton.flagmega.importer import VLLM_INDUCTOR_LEVEL3


def test_import_cli_dispatches_the_explicit_profile(monkeypatch, tmp_path):
    seen = []

    def imported(path, **kwargs):
        seen.append((path, kwargs))
        raise RuntimeError("import boundary reached")

    monkeypatch.setattr(cli, "import_model", imported)
    args = cli._parser().parse_args(["import", "--model", "checkpoint", "--full-model",
                                    "--numerical-profile", VLLM_INDUCTOR_LEVEL3,
                                    "--output", str(tmp_path / "imported.py")])
    with pytest.raises(RuntimeError, match="import boundary reached"):
        cli._run(args)
    assert seen == [("checkpoint", {"revision": None, "numerical_profile": VLLM_INDUCTOR_LEVEL3,
                                   "mode": "decode-1", "num_tokens": 1})]


def test_compile_cannot_silently_override_a_saved_contract(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "load_module", lambda _: SimpleNamespace(metadata={"numerical_contract": VLLM_INDUCTOR_LEVEL3}))
    args = cli._parser().parse_args(["compile", "--input", "edited.py", "--numerical-profile", "nncase",
                                    "--output", str(tmp_path / "artifact")])
    with pytest.raises(IRSchemaError, match="does not match"):
        cli._run(args)
