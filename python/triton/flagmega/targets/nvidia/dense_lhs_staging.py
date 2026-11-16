# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit consumer-owned LHS transport for the ordinary BF16 GEMV."""

from dataclasses import replace


def dense_lhs_staging_implementations(plain_gemv, lhs_workspace):
    """Reuse the ordinary arithmetic and the existing typed row workspace."""
    return (replace(
        plain_gemv,
        id=plain_gemv.id + "_lhs8192_async",
        parameters={**plain_gemv.parameters, "lhs_stage_extent": 8192,
                    "lhs_copy_tile": 1024, "lhs_copy_kind": "async"},
        contract={**plain_gemv.contract, "required_input_dtype": "bfloat16",
                  "required_output_dtype": "bfloat16"},
        requires=(*plain_gemv.requires, "async_copy"),
        facts={**plain_gemv.facts, "stages_local_lhs": True},
        shared_workspaces=(*plain_gemv.shared_workspaces, lhs_workspace),
        transfer_pipeline=replace(
            plain_gemv.transfer_pipeline, consumer_shared_workspace_indices=(1,)),
    ),)
