# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest

from triton.flagmega import cli
from triton.flagmega.errors import IRSchemaError


def test_import_cli_passes_prefill_mode_and_static_token_extent(monkeypatch):

    def imported(path, **kwargs):
        assert path == "checkpoint"
        assert kwargs["mode"] == "prefill" and kwargs["num_tokens"] == 65
        raise RuntimeError("import reached")

    monkeypatch.setattr(cli, "import_model", imported)
    args = cli._parser().parse_args([
        "import", "--model", "checkpoint", "--full-model", "--mode", "prefill", "--num-tokens", "65", "--output",
        "out.py"
    ])
    with pytest.raises(RuntimeError, match="import reached"):
        cli._run(args)


@pytest.mark.parametrize("options", (("--mode", "decode-1"), ("--num-tokens", "1")))
def test_compile_input_cannot_override_saved_execution_contract(monkeypatch, options):
    module = SimpleNamespace(metadata={"mode": "prefill", "tokens_per_call": 65})
    monkeypatch.setattr(cli, "load_module", lambda _: module)
    args = cli._parser().parse_args(["compile", "--input", "prefill.py", "--output", "artifact", *options])
    with pytest.raises(IRSchemaError, match="does not match the saved IR"):
        cli._run(args)
