# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Model caller contract, distinct from the single-row standalone constructor.

The explicit version selects column-register/v4 or corrected v6. This contract describes its
embedding in a closed model: readonly rows are caller parameters, state is an
input-owned layer view, and the arena is a planned function workspace.
"""
from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from .r004_column_phase_bindings import column_phase_plan, PLAN_BODY_SHA256, IMPLEMENTATION
from .r004_gdn_phase_bindings import _same_contract, standalone_request
from .r004_model_policy import model_request, SUBJECT_SHA256

SCHEMA = "r004.gdn.native-model/v1"
REQUEST_KEY = "r004_native_model_request"


def native_model_contract(version=4):
    if version == 6:
        from .r004_column_phase_bindings_v6 import column_phase_plan as v6_plan
        from .r004_model_policy_v6 import SUBJECT_SHA256 as v6_subject
        plan = v6_plan()
        contract = native_model_contract()
        return {**contract, "schema":"r004.gdn.native-model/v2",
                "implementation":plan["implementation"],
                "plan_body_sha256":plan["plan_body_sha256"],
                "numerical_subject_sha256":v6_subject,
                "norm_primitive":plan["norm_primitive"]}
    if version != 4:
        raise IRVerificationError("Unsupported native model version.")
    plan = column_phase_plan()
    return dict(schema=SCHEMA, implementation=IMPLEMENTATION,
                plan_body_sha256=PLAN_BODY_SHA256, numerical_subject_sha256=SUBJECT_SHA256,
                text_layers=64, recurrent_calls=48, readonly_rows=48,
                parameter_binding="typed-caller-fields-per-invocation",
                state_binding="selected-layer-view-once",
                arena_binding="planned-chip-function-workspace",
                arena={k:v for k,v in plan["pool"].items() if k != 'root'},
                arena_formal='phase_pool',
                native_state_shape=[48,128,128], native_state_nbytes=3145728,
                native_layer=0, entry_order=[e["phase"] for e in plan["entries"]],
                completion="serial-caller-stream-all-ctas",
                constructor="ordinary-model-runtime-binding")


def native_model_request(module):
    request = module.metadata.get(REQUEST_KEY)
    if request is None:
        return None
    if (standalone_request(module) is not None or model_request(module) is None
            or not _same_contract(request, native_model_contract(
                6 if model_request(module)["schema"] == "flagmega.r004-model-policy/v2" else 4))):
        raise IRVerificationError("Invalid native model GDN contract; standalone is a separate ABI.",
                                  stage=module.stage)
    if (module.metadata.get("num_hidden_layers") != 64
            or tuple(module.metadata.get("imported_layer_indices", ())) != tuple(range(64))):
        raise IRVerificationError("Native model GDN requires all 64 text layers.", stage=module.stage)
    return request


def prepare_native_model(module):
    if module.stage not in ("unused_functions_removed", "distribution_constants_frozen"):
        raise IRVerificationError("Native model contract must be installed before distribution.")
    if model_request(module) is None or standalone_request(module) is not None:
        raise IRVerificationError("Native model GDN requires the owning model policy.")
    if native_model_request(module) is not None:
        return module
    return replace(module, metadata={**module.metadata, REQUEST_KEY: native_model_contract(
        6 if model_request(module)["schema"] == "flagmega.r004-model-policy/v2" else 4)})


def model_phase_symbol(phase, contract=None):
    if phase not in ("prep", "core", "y"):
        raise IRVerificationError("Unsupported native model GDN phase.")
    if contract is None or _same_contract(contract, native_model_contract()):
        return "r004_model_T_" + phase
    if _same_contract(contract, native_model_contract(6)):
        return "r004_model_T_" + phase + "_v6"
    raise IRVerificationError("Invalid native model phase contract.")
