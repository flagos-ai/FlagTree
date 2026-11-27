# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Static R031 t3 recipe, with one output column and four physical warps.

The signed source has a two-column L2 issue window, not register prefetch.
Inline issue removes the otherwise padded producer warp from this one-column
CTA. No payload or inter-CTA completion promise accompanies a prefetch.
"""
from dataclasses import replace

from . import iq4_q8_renderer as accepted
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir.launch_recipe import TransportRowLaunchRecipe

VARIANT = "r041_glu_transport_t3_compact"
SCHEMA = "flagmega.iq4-q8-static-transport/r041-t3-compact"
ABI = "H003ConsumerPartial-f32-3x32/bar3-count128/v26-global-pf2"
N, K = 17408, 5120
GRID, NUM_WARPS, MAXNREG = 17408, 4, 40
CAPACITY, PRODUCER_REGISTERS = 2, 32
NO_SHARED_STAGING = True
TRANSPORT_KIND = "global_idle"
SEGMENTS = (176,)
PREFETCH_WINDOW = 2
# Actual signed t3 register-prefetch depth; the window above is two.
PREFETCH_DEPTH = 0
PACKET_MODE = "cached_tables"
PTX_OPTIONS = "--minnctapersm=9 --opt-level=2"
RECIPE = TransportRowLaunchRecipe(N, 1, "persistent", GRID, 484,
                                 capacity=CAPACITY, whole_row=True)


def recipe(rows=1):
    if type(rows) is not int or not 1 <= rows <= 5:
        raise CodegenError("static t3 requires 1..5 checked token rows")
    return RECIPE


def token_rows(raw):
    return accepted._shape(accepted._bindings(raw, "inputs", "q8")[0]["abi"], "logical_shape")[0]


def parameters(family, rows=1):
    if family != "iq4_q8_glu":
        raise CodegenError("static t3 qualifies only N17408/K5120 GLU")
    recipe(rows)
    return {**accepted.implementation_parameters(family),
            "maxnreg": MAXNREG, "ptx_options": PTX_OPTIONS,
            "consumer_barrier_abi": ABI, "transfer_schema": SCHEMA,
            "no_shared_staging": NO_SHARED_STAGING,
            "transport_kind": TRANSPORT_KIND, "transport_segments": list(SEGMENTS),
            "transport_whole_row": True, "prefetch_window": PREFETCH_WINDOW,
            "prefetch_depth": PREFETCH_DEPTH, "packet_mode": PACKET_MODE,
            "producer_registers": PRODUCER_REGISTERS, "num_stages": 3,
            "transport_capacity": CAPACITY, "token_rows": rows,
            "persistent_row_protocol": recipe(rows).to_data()}


def implementations(rows=1):
    from .iq4_q8_pipeline_renderer import implementations as base_implementations
    base = next(i for i in base_implementations() if i.family == "iq4_q8_glu")
    return (replace(base, id="tir.iq4_q8_glu." + VARIANT, variant=VARIANT,
                    parameters=parameters(base.family, rows), requires=(),
                    facts={**base.facts, "transfer_pipeline": False, "host_tensor_descriptor": False},
                    shared_workspaces=(), transfer_pipeline=None,
                    contract={**base.contract, "mapping_identity": VARIANT,
                              "row_epoch_recipe": recipe(rows).to_data()}),)


def validate_raw(raw):
    if raw["variant"] != VARIANT or raw["family"] != "iq4_q8_glu":
        raise CodegenError("wrong static t3 subject")
    if (raw["semantic_attrs"]["logical_n"], raw["semantic_attrs"]["logical_k"]) != (N, K):
        raise CodegenError("wrong static t3 shape")
    rows = token_rows(raw)
    for key, value in parameters(raw["family"], rows).items():
        if accepted._canonical(raw["parameters"].get(key)) != accepted._canonical(value):
            raise CodegenError("static t3 parameter mismatch " + key)
    if accepted._canonical(raw["parameters"].get("task_mapping")) != accepted._canonical(recipe(rows).to_data()):
        raise CodegenError("static t3 task mapping mismatch")
    accepted.validate_raw({**raw, "variant": accepted.VARIANT,
                           "parameters": accepted.implementation_parameters(raw["family"])})
    for name in ("gate_weight", "up_weight"):
        accepted._abi(accepted._bindings(raw, "inputs", name)[0], dtype="uint8", shape=(N, 2720), alignment=16)
    if raw.get("transfer_pipeline") is not None or raw.get("shared_workspaces"):
        raise CodegenError("static four-warp t3 has no producer/shared transport")


def materialize(raw):
    validate_raw(raw)
    rows = token_rows(raw)
    result = accepted.materialize({**raw, "variant": accepted.VARIANT,
                                   "parameters": accepted.implementation_parameters(raw["family"])})
    return {**result, "schema": SCHEMA, "grid": [GRID, 1, 1], "program_kind": "persistent",
            "jit_module": "triton.flagmega.codegen.triton.iq4_q8_static_t3_compact", "jit_entry": "transport_t3",
            "constexprs": {"N": N, "K": K, "ROWS": rows},
            "compile_options": {**result["compile_options"], "num_warps": NUM_WARPS,
                                "maxnreg": MAXNREG, "ptx_options": PTX_OPTIONS,
                                "consumer_barrier_abi": ABI},
            "no_shared_staging": True, "transport_kind": TRANSPORT_KIND,
            "persistent_row_recipe": recipe(rows).to_data(),
            "logical_static_shared_bytes": 768 if rows == 1 else min(rows, 4) * 128 * 8}
