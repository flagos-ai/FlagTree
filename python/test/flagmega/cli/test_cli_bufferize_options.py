# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

from triton.flagmega import cli


@pytest.mark.parametrize("command", ["compile", "resume", "replay", "stage"])
@pytest.mark.parametrize("level", ["fast", "optimized"])
def test_allocator_flag_reaches_compiler_options(command, level):
    arguments = [command, "--input", "before.py", "--output", "trial", "--bufferize-opt-level", level]
    if command == "stage":
        arguments.insert(1, "bufferize")
    args = cli._parser().parse_args(arguments)
    assert cli._options(args).bufferize_opt_level == level


@pytest.mark.parametrize("command", ["compile", "resume", "replay", "stage"])
def test_omitting_allocator_flag_preserves_target_or_saved_level(command):
    arguments = [command, "--input", "before.py", "--output", "trial"]
    if command == "stage":
        arguments.insert(1, "bufferize")
    args = cli._parser().parse_args(arguments)
    assert cli._options(args).bufferize_opt_level is None
