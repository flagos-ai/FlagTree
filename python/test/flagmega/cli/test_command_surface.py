# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

import pytest

import triton.flagmega.cli as cli


def test_cli_does_not_advertise_unimplemented_measure_command():
    parser = cli._parser()
    subparsers = next(action for action in parser._actions if hasattr(action, "choices") and action.choices)
    assert "measure" not in subparsers.choices


def test_cli_does_not_swallow_process_control_exceptions(monkeypatch):
    monkeypatch.setattr(cli, "_run", lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        cli.main(["verify", "unused.py"])
