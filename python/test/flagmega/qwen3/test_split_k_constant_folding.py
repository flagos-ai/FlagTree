# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.compiler import Compiler
from triton.flagmega.importer import import_qwen3_model

from .helpers import full_metadata_checkpoint


def test_removed_target_pack_does_not_leave_qkv_constant_artifacts():
    module = Compiler().compile(
        import_qwen3_model(full_metadata_checkpoint(num_hidden_layers=2)))
    module = module.module

    assert not any(
        "qkv_pack" in node.id and node.op == "tir.kernel"
        for node in module.nodes
    )
    packed_assets = [
        node for node in module.nodes
        if node.op == "tir.buffer"
        and node.metadata.get("constant_output", "").endswith("qkv_pack.physical")
    ]
    assert packed_assets == []
    assert all(
        "target_asset_abi" not in node.metadata for node in module.nodes
    )
