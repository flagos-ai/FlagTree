# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A56 representation guards; allocation and execution remain owner supplied.

These checks consume actual TensorTypes or normal typed raw bindings. They do
not create buffers, certify liveness, select a provider or prove native math.
"""
from collections.abc import Mapping

from triton.flagmega.ir import TupleType
from triton.flagmega.ir.distributed_inference import tensor_of

from .u4_q8_abi import Q8Layout, exact, require, validate_q8_tensor_type
from .u4_views import TensorLayout

CONTRACT_ID = "a56-norm-c1-c2-vector/v1"
OP = "u4.norm_apply_q8_c1_c2"
INPUT_ROLES = ("hidden", "stats", "scale", "bias")
RESULT_ROLES = ("result_0", "result_1", "result_2")
INPUT_LAYOUTS = (
    TensorLayout("bfloat16", (1, 640), (8,)),
    TensorLayout("float32", (1, 1, 1)),
    TensorLayout("float32", (640,), (8,)),
    TensorLayout("float32", (640,), (8,)),
)
RESULT_LAYOUTS = (
    INPUT_LAYOUTS[0],
    TensorLayout("uint8", (1, 160, 36)),
    TensorLayout("uint8", (2, 160, 36)),
)


def validate_norm_types(input_types, result_type):
    """Validate real scalar/vector dtype, every lane, rank and fixed extent."""
    require(len(input_types) == 4, "A56 requires exactly four NormApply inputs")
    require(isinstance(result_type, TupleType) and len(result_type.fields) == 3,
            "A56 requires three independent semantic result fields")
    for actual, expected in zip(input_types, INPUT_LAYOUTS):
        expected.validate_tensor_type(tensor_of(actual))
    for actual, expected in zip(result_type.fields, RESULT_LAYOUTS):
        expected.validate_tensor_type(tensor_of(actual))
    return dict(contract=CONTRACT_ID, inputs=[x.to_data() for x in INPUT_LAYOUTS],
                results=[x.to_data() for x in RESULT_LAYOUTS])


def field_q8_contract(producer, field):
    """Derive C1/C2 from the actual registered field, never top-level attrs."""
    from .u4_ops import q8_tuple_contract
    require(producer.op == OP and type(field) is int and field in (1, 2),
            "A56 Q8 ABI requires actual field1 or field2 of its registered producer")
    require(isinstance(producer.type, TupleType) and len(producer.type.fields) == 3,
            "A56 producer has no complete three-field result")
    attrs = q8_tuple_contract(producer, field)
    layout = Q8Layout(5120, field)
    require(attrs["components"] == field and exact(attrs["q8_schema"], layout.to_data()),
            "A56 actual field Q8 contract differs from its physical shape")
    validate_q8_tensor_type(tensor_of(producer.type.fields[field]), layout)
    return attrs


def validate_gdn_q8_type(q8_type):
    """The new input9 is a complete scalar U8 C2 tensor, not a vector packet."""
    layout = Q8Layout(5120, 2)
    validate_q8_tensor_type(tensor_of(q8_type), layout)
    return layout.to_data()


def _chip(binding, layout, *, q8=False):
    from .u4_encoders import _typed
    abi = _typed(binding, layout, alignment=16 if q8 else layout.itemsize)
    require(abi.get("memory_sharing_scope") == "chip" and not abi.get("pool_scope_local")
            and abi.get("storage_kind") in (None, "canonical_global"),
            "A56 input/result must expose complete canonical chip storage")
    if q8:
        require(abi.get("pool_byte_offset", 0) % 16 == 0,
                "A56 Q8 root offset must retain its declared 16-byte alignment")
    return abi


def validate_norm_raw_representation(raw):
    """Representation subset of the new Kernel encoder's actual raw checks.

    The Kernel owner must also validate selected implementation, arithmetic,
    launch/workspaces, and emitted accesses. Distribution must check actual
    all64 caller resolution and inclusive liveness through every reader.
    """
    from .u4_encoders import _binding, _no_alias
    require(isinstance(raw, Mapping) and raw.get("semantic_op") == OP,
            "A56 requires its actual registered semantic operation")
    require(len(raw.get("inputs", ())) == 4 and len(raw.get("outputs", ())) == 3,
            "A56 raw tuple ABI requires four inputs and exactly three outputs")
    reads = tuple(_binding(raw, "inputs", x) for x in INPUT_ROLES)
    writes = tuple(_binding(raw, "outputs", x) for x in RESULT_ROLES)
    for binding, layout in zip(reads, INPUT_LAYOUTS):
        _chip(binding, layout)
    for field, (binding, layout) in enumerate(zip(writes, RESULT_LAYOUTS)):
        _chip(binding, layout, q8=field in (1, 2))
    _no_alias(reads, writes)
    return dict(contract=CONTRACT_ID, roles=[*INPUT_ROLES, *RESULT_ROLES],
                layouts=[x.to_data() for x in (*INPUT_LAYOUTS, *RESULT_LAYOUTS)],
                roots=[b["abi"]["physical_buffer"] for b in writes],
                field_components={"1": 1, "2": 2},
                actual_lifetime_checked=False, native_access_verified=False)


def validate_gdn_q8_binding(binding):
    """Validate only the new external C2 buffer; keep the old nine ABI checks."""
    return _chip(binding, RESULT_LAYOUTS[2], q8=True)
