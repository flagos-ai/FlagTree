# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.kernel_call_renderers import (
    _call_runtime_arguments,
)


def _parameter(storage: str, runtime_argument: str):
    return {
        "formal": storage,
        "buffers": ({
            "runtime_argument": runtime_argument,
            "runtime_value_kind": "pointer",
            "abi": {"storage": storage, "pooled": True},
        },),
    }


def test_renderer_uses_resolved_pool_argument_instead_of_storage_spelling():
    raw = {
        "inputs": (
            _parameter("workspace", "workspace_2"),
            _parameter("rdata", "rdata_2"),
            _parameter("block_data", "block_data"),
        ),
        "outputs": (),
        "workspaces": (),
    }

    assert _call_runtime_arguments(raw) == [
        "workspace_2",
        "rdata_2",
        "block_data",
    ]
