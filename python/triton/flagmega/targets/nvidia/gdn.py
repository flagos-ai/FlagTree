# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Masked owner-row state prefetch profiles for recurrent GDN."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.ir import T, tensor_type


R004_REFERENCE_ID = "tir.gdn_recurrent.r004_reference_v4"


def r004_reference_implementations():
    """Frozen C comparator; applicability requires the versioned standalone plan."""
    yield TritonImplementation(
        R004_REFERENCE_ID, "gdn_recurrent", "r004_reference_v4",
        {"tile_state": (128, 16), "projection_tile": 128,
         "phase_schema": "r004.gdn.column-register/v4"},
        {"standalone_module": "r004_C_full_p5_v4",
         "standalone_schema": "r004.gdn.standalone/v4"},
        facts={"internal_grid_barriers": 0},
    )

    yield TritonImplementation(
        "tir.gdn_recurrent.r004_column_v4", "gdn_recurrent", "r004_column_v4",
        {"tile_state": (128, 16), "projection_tile": 128, "phase_schema": "r004.gdn.column-register/v4"},
        {"standalone_module": "r004_T_full_p5_v4", "standalone_schema": "r004.gdn.standalone/v4"},
        facts={"internal_grid_barriers": 0},
    )

    yield TritonImplementation(
        "tir.gdn_recurrent.r004_column_v6", "gdn_recurrent", "r004_column_v6",
        {"tile_state": (128, 16), "projection_tile": 128, "phase_schema": "r004.gdn.column-register/v6"},
        {"standalone_module": "r004_T_full_p5_v6", "standalone_schema": "r004.gdn.standalone/v4"},
        facts={"internal_grid_barriers": 0},
    )

    yield TritonImplementation('tir.gdn_recurrent.r017_w1', 'gdn_recurrent', 'r017_w1',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':1, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w1', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_w2', 'gdn_recurrent', 'r017_w2',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':2, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w2', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_w4', 'gdn_recurrent', 'r017_w4',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':4, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w4', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_w8', 'gdn_recurrent', 'r017_w8',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':8, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w8', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_w16', 'gdn_recurrent', 'r017_w16',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':16, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w16', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_w32', 'gdn_recurrent', 'r017_w32',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':32, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w32', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_w64', 'gdn_recurrent', 'r017_w64',
        {'tile_state':(128,16), 'projection_tile':128, 'projection_window':64, 'phase_schema':'r017.gdn.ordered-window/v1'},
        {'standalone_module':'r017_T_full_p5_w64', 'standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})

    yield TritonImplementation('tir.gdn_recurrent.r017_split','gdn_recurrent','r017_split',
        {'tile_state':(128,16),'projection_tile':128,'phase_schema':'r017.gdn.split-ordered/v1'},
        {'standalone_module':'r017_T_full_p5_split','standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})

    yield TritonImplementation('tir.gdn_recurrent.r017_split4','gdn_recurrent','r017_split4',
        {'tile_state':(128,16),'projection_tile':128,'phase_schema':'r017.gdn.split-ordered/v1'},
        {'standalone_module':'r017_T_full_p5_split4','standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})
    yield TritonImplementation('tir.gdn_recurrent.r017_split8','gdn_recurrent','r017_split8',
        {'tile_state':(128,16),'projection_tile':128,'phase_schema':'r017.gdn.split-ordered/v1'},
        {'standalone_module':'r017_T_full_p5_split8','standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})

    yield TritonImplementation('tir.gdn_recurrent.r017_split4lut','gdn_recurrent','r017_split4lut',
        {'tile_state':(128,16),'projection_tile':128,'phase_schema':'r017.gdn.split-ordered/v1'},
        {'standalone_module':'r017_T_full_p5_split4lut','standalone_schema':'r004.gdn.standalone/v4'},
        facts={'internal_grid_barriers':0})

def r004_reference_applicable(node, context, implementation):
    """Fail closed: no default model, unvalidated plan, or other implementation."""
    from collections.abc import Mapping
    from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import (
        standalone_request, validate_phase_plan,
    )

    request = standalone_request(context.module)
    plan = node.attrs.get("phase_plan")
    if not (
        implementation.id == R004_REFERENCE_ID
        and node.op == "nn.gdn_recurrent_core"
        and isinstance(request, Mapping)
        and request.get("schema") == "r004.gdn.standalone/v4"
        and request.get("module") == "r004_C_full_p5_v4"
        and type(request.get("readonly_layer")) is int
        and 0 <= request["readonly_layer"] < 48
        and isinstance(plan, Mapping)
    ):
        return False
    validate_phase_plan(plan)
    return True


def gdn_state_pipeline_implementations():
    for columns in (32, 64, 128, 256, 512):
        for rows in (4, 16, 32):
            partition = T.inplace_transfer_partition(("recurrent",), 3, 0, 1, rows)
            yield TritonImplementation(
                f"tir.gdn_recurrent.state_smem_pipeline_k{columns}_v{rows}",
                "gdn_recurrent", "state_smem_pipeline",
                {"tile_state": (columns, rows), "projection_tile": 128, "num_stages": 2,
                 "consumer_warps": 8, "producer_warps": 1, "producer_registers": 24},
                {"owner_row_state_snapshot": True, "required_activation_dtype": "bfloat16"},
                ("cooperative_grid", "grid_sync", "async_copy", "warp_specialize"),
                {"internal_grid_barriers": 1},
                (T.shared_workspace_descriptor("state_stage", tensor_type("float32", (2, rows, columns)), 16),),
                T.transfer_pipeline_contract((T.transfer_pipeline_channel(
                    "state", (0,), (0,), 16, inplace_partition=partition),), capacity=2),
            )


def gdn_large_key_implementations():
    for columns in (256, 512):
        yield TritonImplementation(
            f"tir.gdn_recurrent.persistent_k{columns}", "gdn_recurrent", "persistent",
            {"tile_state": (columns, 4), "projection_tile": 128},
            requires=("cooperative_grid", "grid_sync"), facts={"internal_grid_barriers": 1},
        )


def gdn_scalar_head_gate_implementations():
    """Optional CPP scalar-ordered gate publication; never an ordinary default."""
    yield TritonImplementation(
        "tir.gdn_recurrent.persistent_scalar_head_gates_v1", "gdn_recurrent", "persistent",
        {"tile_state": (128, 4), "projection_tile": 128, "head_gate_cache": True,
         "head_cache_slots": 2, "projection_order": "ascending_scalar_C1",
         "cache_contents": ("beta", "decay"), "cache_epoch": "per_GDN_invocation",
         "num_warps": 4},
        {"scalar_head_gates": True, "required_codegen_backend": "cpp"},
        ("cooperative_grid", "grid_sync"),
        {"internal_grid_barriers": 1, "internal_cta_barriers": 1,
         "required_codegen_backend": "cpp"},
        (T.shared_workspace_descriptor("head_gates", tensor_type("float32", (2, 2)), 16),),
    )


__all__ = ["gdn_state_pipeline_implementations", "gdn_large_key_implementations",
           "gdn_scalar_head_gate_implementations", "r004_reference_implementations",
           "r004_reference_applicable", "R004_REFERENCE_ID"]
