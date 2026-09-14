# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Masked owner-row state prefetch profiles for recurrent GDN."""

from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.ir import T, tensor_type


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


__all__ = ["gdn_state_pipeline_implementations", "gdn_large_key_implementations"]
