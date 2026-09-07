# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from triton.flagmega import cli


class _Checkpoint:
    pass


def _args(*extra):
    return cli._parser().parse_args((
        "resume",
        "--input",
        "checkpoint.py",
        "--output",
        "build/artifacts/result",
        "--checkpoint",
        "model",
        *extra,
    ))


def test_cli_rdata_cache_defaults_to_artifact_parent_filesystem():
    assert cli._rdata_cache_dir(_args(), _Checkpoint()) == Path(
        "build/artifacts/.flagmega-rdata-cache"
    )


def test_cli_rdata_cache_can_be_relocated_or_disabled():
    assert cli._rdata_cache_dir(
        _args("--rdata-cache-dir", "build/cache"), _Checkpoint()
    ) == Path("build/cache")
    assert cli._rdata_cache_dir(
        _args("--no-rdata-cache"), _Checkpoint()
    ) is None


def test_cli_does_not_create_an_unused_cache_without_checkpoint():
    assert cli._rdata_cache_dir(_args(), None) is None
