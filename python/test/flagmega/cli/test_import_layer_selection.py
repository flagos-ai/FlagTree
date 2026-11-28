# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega.cli import _parser


@pytest.mark.parametrize("command", ["import", "compile"])
def test_cli_passes_nonzero_layer_to_architecture_importer(command):
    options = _parser().parse_args([command, "--model", "checkpoint", "--layer", "3", "--output", "out"])
    assert options.layer == 3
