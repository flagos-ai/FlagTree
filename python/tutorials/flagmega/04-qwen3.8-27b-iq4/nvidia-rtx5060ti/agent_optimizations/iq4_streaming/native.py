"""Versioned native catalogs: accepted r003 Q8 and historical r001 E32.

The graph's r003 implementation model calls q8_implementations. Historical
r001 entry points remain available for its existing reproduction suites.
"""
from triton.flagmega.codegen.triton.implementation import TritonImplementation
from triton.flagmega.codegen.triton.iq4_streaming_renderer import (
    SCHEMA, PLAN_SHA256, NATIVE_INTERFACE, RESIDENCY_ENCODING, FAMILIES,
    validate_raw, projection, glu, interleaved, register,
)
from triton.flagmega.errors import CodegenError
from triton.flagmega.codegen.triton import iq4_q8_renderer as q8

Q8_FAMILIES = q8.FAMILIES
Q8_SCHEMA = q8.SCHEMA


def q8_implementations(*, row_tile=1, num_warps=4):
    """Four complete-result ordinary families using the accepted JIT source."""
    if type(row_tile) is not int or row_tile != 1 or type(num_warps) is not int or num_warps != 4:
        raise CodegenError("r003 Q8 requires row tile 1 and four warps")
    return tuple(TritonImplementation(
        f"tir.{family}.q8_r003_v2", family, q8.VARIANT,
        parameters=q8.implementation_parameters(family),
        facts={"internal_grid_barriers": 0, "required_codegen_backend": "triton"},
    ) for family in q8.FAMILIES)


q8_register = q8.register
q8_validate_raw = q8.validate_raw
q8_materialize = q8.materialize

def implementations(*, row_tile=4, kv_row_tile=4):
    tiles = sorted({row_tile, kv_row_tile})
    if tiles != [4]:
        raise CodegenError("r001 requires R4; no other row tile is authorized")
    return tuple(TritonImplementation(
        f"tir.{family}.warp_local_r001_v21", family, "stream",
        parameters={"tile_n": r, "num_warps": 4, "native_interface": NATIVE_INTERFACE,
                    "plan_sha256": PLAN_SHA256, "plan_hash": PLAN_SHA256,
                    "residency_encoding": RESIDENCY_ENCODING},
        facts={"internal_grid_barriers": 0, "required_codegen_backend": "triton"},
    ) for family in FAMILIES for r in tiles)
