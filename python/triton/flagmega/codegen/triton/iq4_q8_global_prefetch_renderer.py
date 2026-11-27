# Copyright 2026 FlagOS Contributors; SPDX-License-Identifier: MIT
"""Static per-family Q8 producer/consumer lowering. No global configuration.

Pinned from R031 bootstrap and generated R027 GLU: the two-column bounded
L2 issue window is distinct from t3's optional register-prefetch depth (zero).
The role channels describe producer read permissions; no weight payload crosses
shared memory and no completion/residency promise is made by an L2 prefetch.
"""

from dataclasses import replace

from . import iq4_q8_renderer as accepted
from . import iq4_q8_pipeline_renderer as pipeline
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import tensor_type
from triton.flagmega.ir.launch_recipe import StaticGlobalPrefetchLaunchRecipe, STATIC_PREFETCH_QUALIFICATIONS
from triton.flagmega.ir.tir import TIRSharedWorkspaceDescriptor

VARIANT = "r038_glu_global_prefetch_v1"
SCHEMA = "flagmega.iq4-q8-static-global-prefetch/r038-v1"
QUALIFIED_T3_ABI = "H003ConsumerPartial-f32-3x32/bar3-count128/v26-global-pf2"
ABI = "H003ConsumerPartial-f32-ROWSx4x32/count128/v2-static-global-pf2"
N, K = 17408, 5120
NO_SHARED_STAGING = True
TRANSPORT_KIND = "global_idle"
QUALIFIED_T3_MAXNREG = 40
MAXNREG = 56
PRODUCER_REGISTERS = 32
PREFETCH_WINDOW = 2
PREFETCH_DEPTH = 0
PACKET_MODE = "cached_tables"
PTX_OPTIONS = "--minnctapersm=9 --opt-level=2"
QUALIFIED_T3_GRID = 17408
GRID = 2176
SEGMENTS = (176,)
RECIPE = StaticGlobalPrefetchLaunchRecipe(N, 1, "persistent", GRID, 61, 5)


def recipe(rows=5, qualification="glu"):
    if type(rows) is not int or not 2 <= rows <= 5:
        raise CodegenError("static global-prefetch requires 2..5 checked token rows; ROWS=1 keeps the accepted path")
    q = STATIC_PREFETCH_QUALIFICATIONS[qualification]
    return StaticGlobalPrefetchLaunchRecipe(q[1], 1, "persistent", q[4], q[5], rows, qualification)


def token_rows(raw):
    return accepted._shape(accepted._bindings(raw, "inputs", "q8")[0]["abi"], "logical_shape")[0]


def qualify(family, n, k):
    for name, q in STATIC_PREFETCH_QUALIFICATIONS.items():
        if q[:3] == (family, n, k):
            return name
    return None


def parameters(family, rows=5, qualification="glu"):
    q = STATIC_PREFETCH_QUALIFICATIONS[qualification]
    if q[0] != family:
        raise CodegenError("static global-prefetch family qualification mismatch")
    recipe(rows, qualification)
    return {
        **pipeline.parameters(family),
        "maxnreg": MAXNREG,
        "producer_registers": PRODUCER_REGISTERS,
        "consumer_barrier_abi": ABI,
        "ptx_options": PTX_OPTIONS,
        "transfer_schema": SCHEMA,
        "transfer_block_shape": [1],
        "transfer_k256_blocks": 0,
        "no_shared_staging": NO_SHARED_STAGING,
        "transport_kind": TRANSPORT_KIND,
        "bounded_prefetch": True,
        "prefetch_window": PREFETCH_WINDOW,
        "prefetch_depth": PREFETCH_DEPTH,
        "packet_mode": PACKET_MODE,
        "transport_segments": list(SEGMENTS),
        "token_rows": rows,
        "persistent_row_protocol": recipe(rows, qualification).to_data(),
    }


def implementation(rows, qualification):
    family = STATIC_PREFETCH_QUALIFICATIONS[qualification][0]
    base = next(i for i in pipeline.implementations() if i.family == family)
    return replace(
            base,
            id="tir." + family + "." + VARIANT + ("" if qualification == "glu" else "." + qualification),
            variant=VARIANT,
            parameters=parameters(base.family, rows, qualification),
            requires=("warp_specialize",),
            facts={**base.facts, "host_tensor_descriptor": False},
            shared_workspaces=tuple(
                TIRSharedWorkspaceDescriptor(bank + "_role", tensor_type("int32", (2, 1)), 128, matrix_compatible=False)
                for bank in (("gate", "up") if family == "iq4_q8_glu" else ("weight",))
            ),
            contract={**base.contract, "mapping_identity": VARIANT, "row_epoch_recipe": recipe(rows, qualification).to_data()},
        )


def implementations(rows=5):
    return tuple(implementation(rows, q) for q in STATIC_PREFETCH_QUALIFICATIONS)


def validate_raw(raw):
    if raw["variant"] != VARIANT:
        raise CodegenError("wrong static global-prefetch subject")
    qualification = qualify(raw["family"], raw["semantic_attrs"]["logical_n"], raw["semantic_attrs"]["logical_k"])
    if qualification is None:
        raise CodegenError("wrong static global-prefetch shape")
    rows = token_rows(raw)
    for key, value in parameters(raw["family"], rows, qualification).items():
        if accepted._canonical(raw["parameters"].get(key)) != accepted._canonical(value):
            raise CodegenError("static global-prefetch parameter mismatch " + key)
    if accepted._canonical(raw["parameters"].get("task_mapping")) != accepted._canonical(recipe(rows, qualification).to_data()):
        raise CodegenError("static global-prefetch task mapping mismatch")
    accepted.validate_raw(
        {**raw, "variant": accepted.VARIANT, "parameters": accepted.implementation_parameters(raw["family"])}
    )
    n, k = STATIC_PREFETCH_QUALIFICATIONS[qualification][1:3]
    if n * (k // 256) * 34 >= 2147483648:
        raise CodegenError("static prefetch weight word offsets must fit signed int32")
    for name in (("gate_weight", "up_weight") if raw["family"] == "iq4_q8_glu" else ("weight",)):
        accepted._abi(accepted._bindings(raw, "inputs", name)[0], dtype="uint8", shape=(n, 136 * (k // 256)), alignment=16)
    pipeline.validate_transfer_contract(raw, implementation(rows, qualification))


def materialize(raw):
    validate_raw(raw)
    rows = token_rows(raw)
    qualification = qualify(raw["family"], raw["semantic_attrs"]["logical_n"], raw["semantic_attrs"]["logical_k"])
    _, n, k, mode, grid, _ = STATIC_PREFETCH_QUALIFICATIONS[qualification]
    result = accepted.materialize(
        {**raw, "variant": accepted.VARIANT, "parameters": accepted.implementation_parameters(raw["family"])}
    )
    result["constexprs"].pop("COLS", None)
    if mode != 1:
        result["constexprs"]["COLS"] = 1
    return {
        **result,
        "schema": SCHEMA,
        "grid": [grid, 1, 1],
        "program_kind": "persistent",
        "jit_module": "",
        "jit_entry": "",
        "compile_options": {
            **result["compile_options"],
            "maxnreg": MAXNREG,
            "ptx_options": PTX_OPTIONS,
            "consumer_barrier_abi": ABI,
        },
        "pipeline_contract": dict(raw["transfer_pipeline"]),
        "shared_workspaces": tuple(raw["shared_workspaces"]),
        "num_stages": 2,
        "producer_warps": 1,
        "producer_registers": PRODUCER_REGISTERS,
        "host_tensor_descriptor_requests": (),
        "transport_descriptors": (),
        "transport_n": n,
        "transport_k": k,
        "transport_mode": mode,
        "persistent_row_recipe": recipe(rows, qualification).to_data(),
        "transport_grid_cap": grid,
        "no_shared_staging": NO_SHARED_STAGING,
        "transport_kind": TRANSPORT_KIND,
        "bounded_prefetch": True,
        "prefetch_window": PREFETCH_WINDOW,
        "prefetch_depth": PREFETCH_DEPTH,
        "packet_mode": PACKET_MODE,
        "pipeline_commits_per_invocation": 0,
        "pipeline_epoch_contract": {
            "schema": SCHEMA,
            "token_rows": rows,
            "payload": "none; independent producer L2 prefetch issue",
            "cache_residency_guaranteed": False,
            "prefetch_issue_window": PREFETCH_WINDOW,
            "progress": "column arithmetic complete before reduction",
            "numerical_k_steps_per_output_column": (k // 256 + 15) // 16,
            "partial_recycle": "count128 barrier after each token partial read",
        },
    }
