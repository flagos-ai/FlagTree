"""Explicit row8 candidate and typed-workspace contract for dense4x24/block128.

Starting point: sibling pc_catalog._candidate typed per-kernel Shared inventory.
Adaptation: no producer role or transfer_pipeline contract, uniform cp.async.
All numerical settings match current local dual_glu (single block, dual GLU).
"""
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.ir import T, tensor_type

ROWS = 8
CHUNK = 1024
STAGES = 2
ROW_BYTES = CHUNK // 256 * 136
PREFIX = "tir.local_iq4_async"

def candidate(family, k, consumer_warps):
    if family not in ("block_fp8", "matmul_glu"):
        raise ValueError("Only native block_fp8 and matmul_glu are in the first scope")
    if k not in (5120, 6144, 17408) or consumer_warps != 4:
        raise ValueError("Unsupported K or uniform consumer warp count")
    if family == "matmul_glu" and k != 5120:
        raise ValueError("The initial GLU bucket is K5120")
    fields = 2 if family == "matmul_glu" else 1
    components = 2 if family == "matmul_glu" else 1
    return TritonImplementation(
        f"{PREFIX}.{family}.r8.k{k}.w{consumer_warps}.v1", family, "simt",
        parameters={"tile_n": ROWS, "compute_num_warps": consumer_warps,
                    "consumer_warps": consumer_warps, "iq4_chunk": CHUNK,
                    "num_stages": STAGES, "activation_components": components,
                    "required_k": k, "required_row_bytes": k // 256 * 136,
                    "projection_order": "lane_group_then32_ascending",
                    "copy_fallback": "synchronous_bytes_if_base_unaligned",
                    "shared_epoch": "per_call_per_row_tile_per_K_chunk"},
        contract={"weight_encoding": "IQ4_XS",
                  "activation_quantization": "existing_local_q8_residual2" if components == 2 else "existing_local_q8_single",
                  "required_codegen_backend": "cpp"},
        requires=("async_copy", "cooperative_grid"),
        facts={"required_codegen_backend": "cpp", "native_quantized_weights": True,
               "transport": "uniform_cp_async_16B", "warp_specialize": False,
               "declared_alias_scope": "synchronous_sequential_model_call",
               "barrier_formula": "2 + ceil(local_scalar_rows/8)*(1+K/1024)",
               "drains_async_copies_before_return": True},
        shared_workspaces=(
            # Q8Block ABI: 4B F32 d followed by 8 packed I32 words. An opaque
            # typed-byte tensor preserves this existing mixed-scalar layout.
            T.shared_workspace_descriptor("activation_q8",
                tensor_type("uint8", (components, k // 32, 36)), 16),
            T.shared_workspace_descriptor("native_weight_stages",
                tensor_type("uint8", (STAGES, fields, ROWS, ROW_BYTES)), 16),
        ),
    )

def catalog(consumer_warps):
    return tuple(candidate("block_fp8", k, consumer_warps) for k in (5120,6144,17408)) + (
        candidate("matmul_glu", 5120, consumer_warps),)

