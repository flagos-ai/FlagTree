# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT

from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
from triton.flagmega.codegen.triton.tir_package import _mesh_context


def test_entry_inlines_address_helper_for_a_replicated_block_pool():
    source = TritonTemplateRegistry().render(
        "module.py.jinja",
        {
            **_mesh_context({
                "hierarchy": (2, 4),
                "hierarchy_levels": "bb",
                "name": "yx",
            }),
            "distributed_entry": True,
            "renderer_version": "test",
            "entry_template": "entrypoints/call_graph.py.jinja",
            "use_tle": True,
            "grid_mesh": False,
            "grid_barrier_axis_groups": (),
            "shared_silu": False,
            "kernel_templates": (),
            "replicated_block_runtime_pool": True,
            "symbol": "flagmega_main",
            "signature": "workspace, block_data",
            "pipeline_schedule": None,
            "entry_events": ({
                "kind": "tir.kernel_call",
                "call": "nested",
                "family": "elementwise",
                "variant": "silu",
                "execution_kind": "local_shard",
                "symbol": "worker",
                "arguments": "_flagmega_block_scope_base(block_data, 64)",
                "barrier_before": False,
            },),
        },
    )

    entry = source.index("def flagmega_main(")
    definition = source.index("def _flagmega_block_scope_base(")
    use = source.index(
        "_flagmega_block_scope_base(block_data, 64)", entry
    )
    assert definition < use
    assert "@triton.jit\ndef _flagmega_block_scope_base" in source
    assert "tl.program_id(0)" not in source[entry:use]
