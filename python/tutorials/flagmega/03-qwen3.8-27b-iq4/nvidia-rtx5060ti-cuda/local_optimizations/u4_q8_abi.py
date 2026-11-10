# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Closed, serialized U4-L Q8 producer/consumer ABI; no allocator or device code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from triton.flagmega.errors import CodegenError

FORMAT_ID = "flagmega.q8-f32-residual/v1"
NUMERICAL_ID = "u4l-g128-w0123-comp-separate-sd16-v1"
SCHEMA_ID = "flagmega.u4-q8-abi/v1"
PRIMITIVE_SCHEMA = "flagmega.u4-primitive-abi/v1"
SUPPORTED_K = (5120, 6144, 17408)


def require(condition, message):
    if not condition:
        raise CodegenError("U4 ABI: " + message)


def exact(actual, expected):
    """JSON structural equality without bool/int or float/int coercion."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            exact(actual[k], v) for k, v in expected.items())
    if type(expected) in (tuple, list):
        return type(actual) in (tuple, list) and len(actual) == len(expected) and all(
            exact(a, e) for a, e in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def integer(value, minimum=0):
    return type(value) is int and minimum <= value < 2**63


def identity(value):
    return type(value) is str and bool(value.strip())


@dataclass(frozen=True)
class Q8Layout:
    k: int
    components: int

    def __post_init__(self):
        require(type(self.k) is int and self.k in SUPPORTED_K, "unsupported complete native K")
        require(type(self.components) is int and self.components in (1, 2), "components must be C1 or C2")

    @property
    def groups(self):
        return self.k // 32

    @property
    def shape(self):
        return (self.components, self.groups, 36)

    @property
    def strides(self):
        return (self.groups * 36, 36, 1)

    @property
    def nbytes(self):
        return self.components * self.groups * 36

    @property
    def native_row_bytes(self):
        return self.k // 256 * 136

    def record_offset(self, component, group):
        require(integer(component) and component < self.components and integer(group) and group < self.groups,
                "Q8 record outside allocation")
        return (component * self.groups + group) * 36

    def quant_offset(self, component, group, lane):
        require(integer(lane) and lane < 32, "quant lane outside record")
        return self.record_offset(component, group) + 4 + lane

    def native_word_offsets(self, group):
        require(integer(group) and group < self.groups, "native group outside row")
        base = 136 * (group // 8)
        return (base, base + 4, *(base + 8 + 16 * (group % 8) + 4 * j for j in range(4)))

    def to_data(self):
        return q8_schema(self.k, self.components)


def q8_schema(k, components):
    layout = Q8Layout(k, components)
    return {
        "schema": SCHEMA_ID, "format_id": FORMAT_ID, "numerical_id": NUMERICAL_ID,
        "k": k, "components": components, "groups": layout.groups,
        "tensor_dtype": "uint8", "scalar_lane_shape": [], "shape": list(layout.shape),
        "scalar_storage_strides": list(layout.strides), "nbytes": layout.nbytes,
        "record_bytes": 36, "global_base_alignment_bytes": 16, "record_alignment_bytes": 4,
        "memory_space": "workspace", "memory_sharing_scope": "chip",
        "storage_kind": "canonical_global", "endianness": "little",
        "fields": [
            {"name": "d", "dtype": "float32", "byte_offset": 0, "nbytes": 4},
            {"name": "qs", "dtype": "int32", "shape": [8], "byte_offset": 4,
             "word_stride_bytes": 4, "nbytes": 32, "packed_element_dtype": "int8",
             "packed_lanes_per_word": 4, "quant_lane_byte_offset": "4+lane"}],
        "source": {"dtype": "bfloat16", "scalar_lane_shape": [], "shape": [1, k],
                   "scalar_storage_strides": [k, 1], "alignment_bytes": 2,
                   "value_boundary": "complete rounded BF16"},
    }


def validate_q8_schema(data):
    require(isinstance(data, Mapping), "missing Q8 schema")
    layout = Q8Layout(data.get("k"), data.get("components"))
    require(exact(data, layout.to_data()), "malformed or unknown Q8 schema/fields/types/layout")
    return layout


def validate_q8_tensor_type(tensor, layout):
    """Inspect the actual TensorType, including scalar dtype rather than lane count alone."""
    from triton.flagmega.ir import TensorType, DType
    require(isinstance(tensor, TensorType) and tensor.dtype == DType.UINT8,
            "Q8 must have scalar U8 TensorType")
    require(all(d.is_fixed for d in tensor.shape)
            and tuple(d.fixed_value for d in tensor.shape) == layout.shape, "wrong Q8 TensorType shape")


def primitive_fields(*, kind, k, components, source_ssa, source_epoch, output_layouts=()):
    """Create mandatory facts for an encoder or encoded consumer's selected primitive.

    These facts accompany (never replace) actual typed operands/PhysicalBuffers.
    Output layouts are TensorLayout.to_data() records, in result tuple order.
    """
    require(kind in ("encode", "matmul", "glu", "interleaved"), "unknown primitive kind")
    require(identity(source_ssa) and identity(source_epoch), "missing source SSA/epoch identity")
    layout = Q8Layout(k, components)
    from .u4_views import TensorLayout
    require(type(output_layouts) in (list, tuple), "output layouts must be an ordered tuple/list")
    outputs = [v.to_data() if isinstance(v, TensorLayout) else v for v in output_layouts]
    if kind == "encode":
        require(not outputs, "encode result is defined by the Q8 schema")
    else:
        require(len(outputs) == (2 if kind == "interleaved" else 1), "wrong result tuple arity")
        outputs = [TensorLayout.from_data(output).to_data() for output in outputs]
        if kind == "matmul":
            allowed = {
                5120: (TensorLayout("bfloat16", (1, 10240)), TensorLayout("bfloat16", (1, 6144)),
                       TensorLayout("bfloat16", (1, 128), (8,)), TensorLayout("float32", (1, 248320))),
                6144: (TensorLayout("bfloat16", (1, 5120)),),
                17408: (TensorLayout("bfloat16", (1, 5120)),),
            }
            require(components == 1 and outputs[0] in [v.to_data() for v in allowed[k]],
                    "encoded matrix is outside the selected full12 geometry")
        if kind == "glu":
            require(components == 2 and outputs == [TensorLayout("bfloat16", (1, 17408)).to_data()]
                    and k == 5120, "GLU requires C2/K5120 scalar BF16 Y17408")
        if kind == "interleaved":
            require(components == 2 and k == 5120 and outputs == [
                TensorLayout("bfloat16", (1, 24, 32), (8,)).to_data()] * 2,
                "interleaved requires C2/K5120 packed Q/gate tuple")
    roles = {"encode": (["value"], ["q8"]), "matmul": (["q8", "weight", "weight_scale"], ["result"]),
             "glu": (["q8", "gate_weight", "up_weight", "gate_scale", "up_scale"], ["result"]),
             "interleaved": (["q8", "weight", "weight_scale"], ["q", "gate"])}
    inputs, results = roles[kind]
    return {"schema": PRIMITIVE_SCHEMA, "kind": kind, "q8": layout.to_data(),
            "source_ssa": source_ssa, "source_epoch": source_epoch,
            "input_roles": inputs, "result_roles": results,
            "canonical_global_arguments": inputs,
            "canonical_global_results": results, "output_layouts": outputs,
            "native_encoding": None if kind == "encode" else "IQ4_XS",
            "native_row_bytes": None if kind == "encode" else layout.native_row_bytes,
            "native_alignment_bytes": None if kind == "encode" else 1,
            "native_word_access": None if kind == "encode" else "global-aligned-u32-or-little-endian-bytes",
            "input_bf16_boundary": "complete rounded BF16", "index_bits": 64,
            "native_rows": None if kind == "encode" else sum(
                TensorLayout.from_data(v).nbytes // TensorLayout.from_data(v).itemsize for v in outputs),
            "output_row_map": {"encode": "group=4*cta+warp;scale=lane0;quant=lane",
                "matmul": "field=0;scalar=row;outer=row//lanes;lane=row%lanes",
                "glu": "field=0;scalar=row",
                "interleaved": "field=(row%512)//256;scalar=256*(row//512)+row%256"}[kind]}


def validate_primitive_fields(data):
    require(isinstance(data, Mapping), "missing primitive fields")
    layout = validate_q8_schema(data.get("q8"))
    expected = primitive_fields(kind=data.get("kind"), k=layout.k, components=layout.components,
        source_ssa=data.get("source_ssa"), source_epoch=data.get("source_epoch"),
        output_layouts=data.get("output_layouts", ()))
    require(exact(data, expected), "primitive roles/identity/native pitch/canonical facts differ")
    return layout


def validate_producer_consumer(producer, consumer):
    a, b = validate_primitive_fields(producer), validate_primitive_fields(consumer)
    require(producer["kind"] == "encode" and consumer["kind"] != "encode", "wrong Q8 edge roles")
    require(a == b and all(producer[k] == consumer[k] for k in ("source_ssa", "source_epoch")),
            "Q8 source/epoch/format differs; component views and cross-epoch reuse are unsupported")


def validate_native_access(*, k, rows, row, group, word, base_address, allocation_nbytes,
                           memory_space="global"):
    """Return (relative byte offset, load mode) for a bounded native 32-bit load."""
    layout = Q8Layout(k, 1)
    require(integer(rows, 1) and integer(row) and row < rows, "native row must be masked before addressing")
    require(integer(word) and word < 6 and integer(base_address), "invalid native pointer/word")
    require(integer(allocation_nbytes) and allocation_nbytes == rows * layout.native_row_bytes,
            "native span must cover exactly the declared rows")
    offset = row * layout.native_row_bytes + layout.native_word_offsets(group)[word]
    require(base_address + offset + 4 < 2**63 and offset + 4 <= allocation_nbytes, "native address overflow")
    # Word loads are legal only for global space AND aligned addresses.
    return offset, "u32" if memory_space == "global" and (base_address + offset) % 4 == 0 else "bytes"


def validate_primitive_bindings(data, *, q8, source=None, outputs=(), weights=()):
    """Bind local facts to concrete views and native core BufferDescriptors.

    Caller supplies named semantic buffers already bound by the normal planner;
    this checks that none has been replaced with a shape-compatible foreign root.
    """
    from triton.flagmega.ir.bufferization import BufferDescriptor
    from .u4_views import TensorLayout, PhysicalView, validate_q8_view, validate_live_views
    layout = validate_primitive_fields(data)
    require(isinstance(q8, PhysicalView), "Q8 binding must be a concrete physical view")
    validate_q8_view(q8, data["q8"])
    require(q8.epoch == data["source_epoch"], "Q8 physical epoch differs from selected primitive")
    if data["kind"] == "encode":
        require(isinstance(source, PhysicalView) and not outputs and not weights,
                "encoder requires exactly its source and Q8 result")
        require(source.name == data["source_ssa"] and source.epoch == q8.epoch
                and source.layout == TensorLayout("bfloat16", (1, layout.k))
                and source.coordinate_space == "canonical_global" and q8.writable,
                "encoder source identity/rounded BF16 boundary/canonical output differs")
        validate_live_views((source, q8))
        return
    require(source is None and len(outputs) == len(data["output_layouts"])
            and len(weights) == (2 if data["kind"] == "glu" else 1), "encoded consumer binding arity differs")
    require(not q8.writable, "encoded consumer cannot mutate immutable Q8 input")
    for output, expected in zip(outputs, data["output_layouts"]):
        require(isinstance(output, PhysicalView) and output.layout == TensorLayout.from_data(expected)
                and output.coordinate_space == "canonical_global" and output.writable
                and output.epoch == q8.epoch and output.alias_of is None,
                "encoded output root/epoch/type/canonical requirement differs")
    validate_live_views((q8, *outputs))
    for weight in weights:
        require(isinstance(weight, BufferDescriptor) and weight.dtype.value == "uint8"
                and weight.shape == (data["native_rows"], layout.native_row_bytes)
                and weight.strides == (layout.native_row_bytes, 1)
                and weight.mem_span.nbytes == data["native_rows"] * layout.native_row_bytes
                and weight.mem_span.buffer.memory_space in ("rdata", "weight", "weights", "readonly")
                and weight.alignment >= 1 and weight.physical_id not in {v.root_id for v in (q8,*outputs)},
                "native weight must retain exact external U8 rows/pitch/span/readonly root")


def validate_q8_edge(producer, consumer, *, produced, consumed):
    """Verify the actual immutable allocation shared by two separate programs."""
    from .u4_views import validate_q8_view
    validate_producer_consumer(producer, consumer)
    validate_q8_view(produced, producer["q8"])
    validate_q8_view(consumed, consumer["q8"])
    require(produced.writable and not consumed.writable and produced.span == consumed.span
            and produced.arena == consumed.arena and produced.frame_offset == consumed.frame_offset
            and produced.epoch == consumed.epoch == producer["source_epoch"]
            and produced.live_start <= consumed.live_start <= consumed.live_end <= produced.live_end,
            "Q8 consumer does not retain producer root/span/frame/completion lifetime")
