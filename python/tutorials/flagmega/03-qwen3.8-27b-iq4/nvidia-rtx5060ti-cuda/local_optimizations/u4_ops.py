"""Frozen U4-L graph semantics; imported explicitly before constructing/resuming IR.

The CPU evaluators are finite-input diagnostics, not native-instruction acceptance.
Native FMA/FTZ/SiLU and final model NUM remain device review obligations.
"""
from math import prod
from collections.abc import Mapping

from triton.flagmega.errors import EvaluationError, IRSchemaError, IRVerificationError, CodegenError
from triton.flagmega.ir import Node, DType, VectorType, TupleType, Placement, SBP, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of, all_broadcast
from triton.flagmega.ir.model import DistributedType, canonical_attributes
from triton.flagmega.ir.ops.core import (
    OpDefinition, OpCost, input_parameter, attribute_parameter, op_definition,
    tensor_nbytes, PythonCall, NodeRef, context_independent_verifier,
)
from triton.flagmega.ir.type_pattern import is_tensor, has_rank

FORMAT_ID = "flagmega.q8-f32-residual/v1"
NUMERICAL_ID = "u4l-g128-w0123-comp-separate-sd16-v1"
ENCODING_ID = "f32-amax127-roundf-residual-rn/v1"
SUPPORTED_K = (5120, 6144, 17408)


def _common(attrs):
    if type(attrs["k"]) is not int or attrs["k"] not in SUPPORTED_K:
        raise IRSchemaError("U4 requires K in 5120/6144/17408.")
    if type(attrs["components"]) is not int or attrs["components"] not in (1, 2):
        raise IRSchemaError("U4 components must be 1 or 2.")
    for key, value in (("format_id", FORMAT_ID), ("numerical_id", NUMERICAL_ID),
                       ("encoding_id", ENCODING_ID)):
        if attrs[key] != value:
            raise IRSchemaError(f"U4 {key} differs from the frozen contract.")
    if not isinstance(attrs["source_epoch"], str) or not attrs["source_epoch"].strip():
        raise IRSchemaError("U4 source_epoch must identify the function invocation/source SSA.")
    # V0 owns the closed heterogeneous byte-field schema.
    from .u4_q8_abi import validate_q8_schema, q8_schema
    try:
        layout = validate_q8_schema(attrs["q8_schema"])
        if (layout.k, layout.components) != (attrs["k"], attrs["components"]):
            raise ValueError("C/K mismatch")
    except (ValueError, TypeError, KeyError, CodegenError) as error:
        raise IRSchemaError(f"U4 invalid q8_schema: {error}") from error
    attrs["q8_schema"] = canonical_attributes(attrs["q8_schema"])
    return attrs


def encoding_attrs(*, k, components, source_epoch):
    from .u4_q8_abi import q8_schema
    return dict(k=k, components=components, source_epoch=source_epoch,
                format_id=FORMAT_ID, numerical_id=NUMERICAL_ID,
                encoding_id=ENCODING_ID, q8_schema=q8_schema(k, components))


def _complete(inputs):
    distributed = [n.type for n in inputs if isinstance(n.type, DistributedType)]
    if distributed and (len(distributed) != len(inputs)
                        or len({t.placement for t in distributed}) != 1
                        or not all(all_broadcast(t) for t in distributed)):
        raise IRSchemaError("U4 ordinary operations require complete broadcast operands on one placement.")


Q8_CONTRACT_KEYS = ("k", "components", "source_epoch", "format_id", "numerical_id", "encoding_id", "q8_schema")
_Q8_TUPLE_PRODUCERS = {}
_Q8_ALLOWED_FIELDS = {("u4.norm_apply_q8_c1", 1): 1,
                      ("u4.norm_apply_q8_c1_c2", 1): 1,
                      ("u4.norm_apply_q8_c1_c2", 2): 2}


def register_q8_tuple_producer(op, field):
    """Register one closed semantic field; unrelated producers remain rejected."""
    if type(field) is not int or (op, field) not in _Q8_ALLOWED_FIELDS:
        raise IRSchemaError("Unknown Q8 semantic producer field.")
    _Q8_TUPLE_PRODUCERS[(op, field)] = _Q8_ALLOWED_FIELDS[(op, field)]


def q8_tuple_contract(producer, field):
    components = _Q8_TUPLE_PRODUCERS.get((producer.op, field)) if type(field) is int else None
    if components is None:
        raise IRSchemaError("Unregistered opaque Q8 tuple producer field.")
    # Validate the actual producer's closed contract before deriving field C.
    actual = _common(dict(producer.attrs))
    expected_components = 1 if producer.op == "u4.norm_apply_q8_c1" else 2
    if actual["k"] != 5120 or actual["components"] != expected_components:
        raise IRSchemaError("Q8 tuple actual producer geometry mismatch.")
    return _common(encoding_attrs(k=actual["k"], components=components,
                          source_epoch=actual["source_epoch"]))


def q8_projection_metadata(producer, *, field=1):
    return {"u4_q8_contract": q8_tuple_contract(producer, field),
            "u4_q8_producer": {"op": producer.op, "node": producer.id, "field": field},
            "u4_opaque": True, "canonical_global": True}


def _q8(q8, attrs):
    expected = tensor_type(DType.UINT8, (attrs["components"], attrs["k"] // 32, 36))
    if tensor_of(q8.type) != expected:
        raise IRSchemaError("U4 encoded C/K/type mismatch; opaque Q8 cannot be packed.")
    if q8.op == "u4.q8_encode":
        provenance = q8.attrs
    elif q8.op == "builtin.get_item":
        producer = q8.metadata.get("u4_q8_producer", {})
        if (type(producer.get("field")) is not int
                or (producer.get("op"), producer.get("field")) not in _Q8_TUPLE_PRODUCERS
                or q8.attrs.get("index") != producer["field"]
                or q8.inputs != (producer.get("node"),)):
            raise IRSchemaError("U4 Q8 projection lacks exact registered tuple provenance.")
        provenance = q8.metadata.get("u4_q8_contract", {})
    elif q8.op.startswith("distributed."):
        # Existing distributed wrappers must retain the complete schema. The
        # complete graph verifier follows their inputs to the registered root.
        provenance = q8.metadata.get("u4_q8_contract", {})
    else:
        raise IRSchemaError("U4 Q8 requires an encoded producer, not generic same-shape U8.")
    for key in Q8_CONTRACT_KEYS:
        if provenance.get(key) != attrs[key]:
            raise IRSchemaError(f"U4 encoded producer/consumer {key} mismatch.")


def verify_q8_provenance(module, q8, attrs):
    """Resolve module edges so copied metadata cannot forge a producer."""
    _q8(q8, attrs)
    current, seen = q8, set()
    while current.op.startswith("distributed."):
        if current.id in seen or len(current.inputs) != 1:
            raise IRSchemaError("U4 Q8 boundary lacks one acyclic source.")
        seen.add(current.id)
        current = module.node_map[current.inputs[0]]
        _q8(current, attrs)
    if current.op == "builtin.get_item":
        producer = module.node_map[current.inputs[0]]
        if ((producer.op, current.attrs["index"]) not in _Q8_TUPLE_PRODUCERS
                or current.metadata["u4_q8_producer"]["op"] != producer.op):
            raise IRSchemaError("U4 Q8 projection references an unregistered actual producer.")
        from triton.flagmega.ir.ops.core import get_definition
        get_definition(producer.op).verify(producer, module)
        actual = q8_tuple_contract(producer, current.attrs["index"])
        for key in Q8_CONTRACT_KEYS:
            if actual.get(key) != attrs[key]:
                raise IRSchemaError(f"U4 tuple actual producer {key} mismatch.")
    elif current.op != "u4.q8_encode":
        raise IRSchemaError("U4 Q8 boundary does not reach a registered producer.")


def _weight(weight, scale, attrs):
    w, s = tensor_of(weight.type), tensor_of(scale.type)
    if w.dtype != DType.UINT8 or w.rank != 2 or not all(d.is_fixed for d in w.shape):
        raise IRSchemaError("U4 IQ4 weight requires scalar U8 fixed native rows.")
    n, row_bytes = (d.fixed_value for d in w.shape)
    if n <= 0 or row_bytes != attrs["k"] // 256 * 136:
        raise IRSchemaError("U4 native IQ4 row pitch/K mismatch.")
    bn, bk = attrs["weight_block_n"], attrs["weight_block_k"]
    if s != tensor_type(DType.FLOAT32, ((n + bn - 1) // bn, (attrs["k"] + bk - 1) // bk)):
        raise IRSchemaError("U4 native placeholder scale shape/dtype mismatch.")
    return n


def _matrix_attrs(attrs):
    attrs = _common(attrs)
    for key in ("weight_block_n", "weight_block_k"):
        if type(attrs[key]) is not int or attrs[key] <= 0:
            raise IRSchemaError(f"U4 {key} must be positive.")
    if attrs["weight_encoding"] != "IQ4_XS":
        raise IRSchemaError("U4 supports native IQ4_XS only.")
    return attrs


def _lanes(value):
    lanes = tuple(value)
    if lanes not in ((), (8,)):
        raise IRSchemaError("U4 frozen output_lanes requires scalar or BF16<8>.")
    return lanes


class _U4Base(OpDefinition):
    const_evaluable = False
    inplace_input_parameters = ()
    inplace_output_parameters = ()

    @classmethod
    def verify(cls, node, module):
        super().verify(node, module)
        if node.op in ("u4.iq4_matmul", "u4.iq4_glu", "u4.iq4_interleaved"):
            verify_q8_provenance(module, module.node_map[node.inputs[0]], node.attrs)

    @classmethod
    def python_call(cls, node):
        # PythonCall is an expression hook; a local import here also makes a
        # standalone emitted checkpoint register extensions in a fresh process.
        expression = ("__import__('local_optimizations.u4_ops', "
                      f"fromlist=['{cls.__name__}']).{cls.__name__}.construct")
        return PythonCall(expression, tuple(NodeRef(x) for x in node.inputs),
                          {**node.attrs, "name": node.id, "metadata": node.metadata})

    @classmethod
    def cost(cls, node):
        fields = node.type.fields if isinstance(node.type, TupleType) else (node.type,)
        return OpCost(bytes_written=sum(tensor_nbytes(tensor_of(t)) for t in fields),
                      notes=("U4-L diagnostic estimate; native cost unmeasured",))


@op_definition("u4.q8_encode")
class Q8Encode(_U4Base):
    value = input_parameter(is_tensor() & has_rank(2))
    k = attribute_parameter()
    components = attribute_parameter()
    source_epoch = attribute_parameter()
    format_id = attribute_parameter()
    numerical_id = attribute_parameter()
    encoding_id = attribute_parameter()
    q8_schema = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attrs):
        return _common(super().normalize_attrs(attrs))

    @classmethod
    def infer_type(cls, inputs, attrs):
        _complete(inputs)
        if tensor_of(cls.value.type_of(inputs)) != tensor_type(DType.BFLOAT16, (1, attrs["k"])):
            raise IRSchemaError("U4 Q8 input must be complete rounded scalar BF16[1,K].")
        return tensor_type(DType.UINT8, (attrs["components"], attrs["k"] // 32, 36))

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        if value.device.type != "cpu":
            raise EvaluationError("U4 diagnostic evaluator is CPU only.")
        import torch
        return torch.from_numpy(encode_q8_reference(value.float().numpy(), node.attrs["components"]))


@op_definition("u4.iq4_matmul")
class IQ4Matmul(_U4Base):
    q8 = input_parameter(is_tensor() & has_rank(3))
    weight = input_parameter(is_tensor() & has_rank(2))
    weight_scale = input_parameter(is_tensor() & has_rank(2))
    k = attribute_parameter()
    components = attribute_parameter()
    source_epoch = attribute_parameter()
    format_id = attribute_parameter()
    numerical_id = attribute_parameter()
    encoding_id = attribute_parameter()
    q8_schema = attribute_parameter()
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    weight_encoding = attribute_parameter(default="IQ4_XS")
    output_dtype = attribute_parameter(default="bfloat16")
    output_lanes = attribute_parameter(default=())

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = _matrix_attrs(super().normalize_attrs(attrs))
        if attrs["components"] != 1:
            raise IRSchemaError("U4 full12 ordinary projection selects C=1.")
        if attrs["output_dtype"] not in ("bfloat16", "float32"):
            raise IRSchemaError("U4 output_dtype must be BF16 or full-logits F32.")
        attrs["output_lanes"] = _lanes(attrs["output_lanes"])
        if attrs["output_dtype"] == "float32" and attrs["output_lanes"]:
            raise IRSchemaError("U4 logits use scalar F32.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        _complete(inputs)
        _q8(cls.q8.read(inputs), attrs)
        n = _weight(cls.weight.read(inputs), cls.weight_scale.read(inputs), attrs)
        lanes = attrs["output_lanes"]
        if n % prod(lanes):
            raise IRSchemaError("U4 N is not divisible by output_lanes.")
        dtype = DType(attrs["output_dtype"])
        return tensor_type(VectorType(dtype, lanes) if lanes else dtype, (1, n // prod(lanes)))

    @classmethod
    def evaluate(cls, node, arguments, context):
        import torch
        q8, weight, _ = arguments
        result = torch.from_numpy(iq4_reference(q8, weight, node.attrs))
        result = result.to(getattr(torch, node.attrs["output_dtype"]))
        lanes = node.attrs["output_lanes"]
        return result.reshape(1, result.numel() // prod(lanes), *lanes)


@op_definition("u4.iq4_glu")
class IQ4Glu(_U4Base):
    q8 = input_parameter(is_tensor() & has_rank(3))
    gate_weight = input_parameter(is_tensor() & has_rank(2))
    up_weight = input_parameter(is_tensor() & has_rank(2))
    gate_scale = input_parameter(is_tensor() & has_rank(2))
    up_scale = input_parameter(is_tensor() & has_rank(2))
    k = attribute_parameter()
    components = attribute_parameter()
    source_epoch = attribute_parameter()
    format_id = attribute_parameter()
    numerical_id = attribute_parameter()
    encoding_id = attribute_parameter()
    q8_schema = attribute_parameter()
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    weight_encoding = attribute_parameter(default="IQ4_XS")
    activation = attribute_parameter(default="silu")

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = _matrix_attrs(super().normalize_attrs(attrs))
        if attrs["components"] != 2 or attrs["k"] != 5120 or attrs["activation"] != "silu":
            raise IRSchemaError("U4 fused GLU requires C=2/K=5120 and complete-row SiLU.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        _complete(inputs)
        _q8(cls.q8.read(inputs), attrs)
        n = _weight(cls.gate_weight.read(inputs), cls.gate_scale.read(inputs), attrs)
        if _weight(cls.up_weight.read(inputs), cls.up_scale.read(inputs), attrs) != n:
            raise IRSchemaError("U4 GLU gate/up row mismatch.")
        return tensor_type(DType.BFLOAT16, (1, n))

    @classmethod
    def evaluate(cls, node, arguments, context):
        import torch
        q8, gate, up, _, _ = arguments
        g = torch.from_numpy(iq4_reference(q8, gate, node.attrs))
        u = torch.from_numpy(iq4_reference(q8, up, node.attrs))
        return (torch.nn.functional.silu(g) * u).to(torch.bfloat16).reshape(1, -1)


@op_definition("u4.iq4_interleaved")
class IQ4Interleaved(_U4Base):
    q8 = input_parameter(is_tensor() & has_rank(3))
    weight = input_parameter(is_tensor() & has_rank(2))
    weight_scale = input_parameter(is_tensor() & has_rank(2))
    k = attribute_parameter()
    components = attribute_parameter()
    source_epoch = attribute_parameter()
    format_id = attribute_parameter()
    numerical_id = attribute_parameter()
    encoding_id = attribute_parameter()
    q8_schema = attribute_parameter()
    weight_block_n = attribute_parameter()
    weight_block_k = attribute_parameter()
    weight_encoding = attribute_parameter(default="IQ4_XS")
    feature_shape = attribute_parameter()
    output_lanes = attribute_parameter()

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = _matrix_attrs(super().normalize_attrs(attrs))
        if attrs["components"] != 2 or attrs["k"] != 5120 or tuple(attrs["feature_shape"]) != (24, 256):
            raise IRSchemaError("U4 interleaved requires C=2 and feature_shape=(24,256).")
        attrs["feature_shape"] = (24, 256)
        attrs["output_lanes"] = tuple(_lanes(x) for x in attrs["output_lanes"])
        if attrs["output_lanes"] != ((8,), (8,)):
            raise IRSchemaError("U4 interleaved requires two distinct BF16<8> fields.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        _complete(inputs)
        _q8(cls.q8.read(inputs), attrs)
        if _weight(cls.weight.read(inputs), cls.weight_scale.read(inputs), attrs) != 12288:
            raise IRSchemaError("U4 interleaved row count mismatch.")
        return TupleType((tensor_type(VectorType(DType.BFLOAT16, (8,)), (1, 24, 32)),) * 2)

    @classmethod
    def evaluate(cls, node, arguments, context):
        import torch
        q8, weight, _ = arguments
        result = torch.from_numpy(iq4_reference(q8, weight, node.attrs)).to(torch.bfloat16)
        result = result.reshape(1, 24, 2, 256)
        return tuple(result[:, :, f, :].contiguous().reshape(1, 24, 32, 8) for f in range(2))


@op_definition("u4.gdn_z_gather")
class GDNZGather(_U4Base):
    """Explicit data movement into the unchanged fixed96 GDN private Z frame."""
    # Invalid complete/broadcast placements must not be retried as logical
    # identity and lifted back to their original (incorrect) owners.
    supports_broadcast_lifting = False
    value = input_parameter(is_tensor() & has_rank(2))
    source_epoch = attribute_parameter()
    mapping_id = attribute_parameter(default="gdn-z-c64-private/v1")

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = super().normalize_attrs(attrs)
        if not isinstance(attrs["source_epoch"], str) or not attrs["source_epoch"]:
            raise IRSchemaError("U4 Z gather requires source_epoch.")
        if attrs["mapping_id"] not in ("gdn-z-c64-private/v1", "gdn-z-c128-private/v1"):
            raise IRSchemaError("U4 Z gather mapping differs from the frozen contract.")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        value = cls.value.type_of(inputs)
        if tensor_of(value) != tensor_type(DType.BFLOAT16, (1, 6144)):
            raise IRSchemaError("U4 Z gather requires BF16[1,6144].")
        if isinstance(value, DistributedType):
            if not all_broadcast(value) or value.placement != Placement((1, 1), "yx", "bb"):
                raise IRSchemaError("U4 Z gather requires complete unit(1,1)/yx/bb broadcast input.")
            fullhead48 = attrs["mapping_id"] == "gdn-z-c128-private/v1"
            return DistributedType(
                value.tensor, (SBP.broadcast(), SBP.split_contiguous((0, 1), 128 if fullhead48 else 64)),
                Placement((4, 12) if fullhead48 else (4, 24), "yx", "bb"),
            )
        return value

    @classmethod
    @context_independent_verifier
    def verify(cls, node, module):
        super().verify(node, module)
        # Generic verification also has a logical fallback for distributed
        # nodes without candidate metadata. This ownership-changing op must
        # retain its exact source/destination relation on that path as well.
        inputs = tuple(module.node_map[value] for value in node.inputs)
        try:
            inferred = cls.infer_call_type(inputs, cls.normalize_attrs(node.attrs))
        except IRSchemaError as error:
            raise IRVerificationError(str(error), stage=module.stage, node_id=node.id) from error
        if inferred != node.type:
            raise IRVerificationError(
                "U4 Z gather result must retain the inferred fixed96 split64 owners.",
                stage=module.stage, node_id=node.id,
            )

    @classmethod
    def evaluate(cls, node, arguments, context):
        return cls.value.read(arguments).clone()


def q8_encode(value, *, k, components, source_epoch, **kwargs):
    return Q8Encode.construct(value, **encoding_attrs(k=k, components=components, source_epoch=source_epoch), **kwargs)


def iq4_matmul(q8, weight, weight_scale, **kwargs):
    return IQ4Matmul.construct(q8, weight, weight_scale, **kwargs)


def iq4_glu(q8, gate_weight, up_weight, gate_scale, up_scale, **kwargs):
    return IQ4Glu.construct(q8, gate_weight, up_weight, gate_scale, up_scale, **kwargs)


def iq4_interleaved(q8, weight, weight_scale, **kwargs):
    return IQ4Interleaved.construct(q8, weight, weight_scale, **kwargs)


def encode_q8_reference(value, components):
    """CPU byte oracle: no dequantize/requantize shortcut and no half-scale Q8."""
    import numpy as np
    x = np.asarray(value, dtype=np.float32).reshape(-1, 32)
    if x.size not in SUPPORTED_K or components not in (1, 2) or not np.isfinite(x).all():
        raise EvaluationError("U4 reference requires supported K, C and finite BF16 values.")
    records = np.zeros((components, len(x), 36), dtype=np.uint8)
    for c in range(components):
        d = np.max(np.abs(x), axis=1).astype(np.float32) / np.float32(127)
        scaled = np.divide(x, d[:, None], out=np.zeros_like(x), where=d[:, None] != 0)
        # floor(abs+0.5) is evaluated in F64 so the addition cannot move a
        # just-below-half F32 value across a tie.
        q = (np.sign(scaled) * np.floor(np.abs(scaled.astype(np.float64)) + 0.5)).astype(np.int8)
        records[c, :, :4] = d.astype("<f4").view(np.uint8).reshape(-1, 4)
        records[c, :, 4:] = q.view(np.uint8)
        x = np.subtract(x, np.multiply(q.astype(np.float32), d[:, None], dtype=np.float32), dtype=np.float32)
    return records


def reduce_u4_partials(partials):
    """Apply the frozen interwarp/component/lane tree to F32[4,C,32]."""
    import numpy as np
    p = np.asarray(partials, dtype=np.float32)
    if p.ndim != 3 or p.shape[0] != 4 or p.shape[1] not in (1, 2) or p.shape[2] != 32:
        raise ValueError("Expected F32[4,C,32] partials.")
    lane = np.add(np.add(np.add(p[0], p[1], dtype=np.float32), p[2], dtype=np.float32),
                  p[3], dtype=np.float32)
    value = lane[0].copy()
    if p.shape[1] == 2:
        value = np.add(value, lane[1], dtype=np.float32)
    for offset in (16, 8, 4, 2, 1):
        prior = value.copy()
        value[:32-offset] = np.add(prior[:32-offset], prior[offset:], dtype=np.float32)
    return value[0]


def iq4_reference(q8, weight, attrs):
    """Native-byte integer-dot oracle with the frozen U4 F32 accumulation tree.

    The scale product is rounded to F32, and the dot update uses FMA rounding
    (the production compiler's contraction still requires native inspection).
    """
    import numpy as np
    if any(x.device.type != "cpu" for x in (q8, weight)):
        raise EvaluationError("U4 diagnostic evaluator is CPU only.")
    records = q8.numpy()
    rows = weight.numpy()
    c, groups, _ = records.shape
    if (c, groups, 36) != (attrs["components"], attrs["k"] // 32, 36):
        raise EvaluationError("U4 runtime encoded payload geometry mismatch.")
    d8 = records[:, :, :4].copy().view("<f4").reshape(c, groups)
    qs = records[:, :, 4:].copy().view(np.int8).astype(np.int32)
    lut = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.int32)
    result = np.zeros(len(rows), np.float32)
    for r, row in enumerate(rows):
        partials = np.zeros((4, c, 32), np.float32)
        for g in range(groups):
            block, inner = 136 * (g // 8), g % 8
            d = np.frombuffer(row[block:block+2].tobytes(), dtype="<f2")[0].astype(np.float32)
            high = int(row[block+2]) | (int(row[block+3]) << 8)
            ls = ((int(row[block+4+inner//2]) >> (4 * (inner % 2))) & 15) | (((high >> (2*inner)) & 3) << 4)
            packed = row[block+8+16*inner:block+24+16*inner]
            decoded = np.concatenate((lut[packed & 15], lut[packed >> 4]))
            dot = (qs[:, g] * decoded).sum(axis=1, dtype=np.int32) * (ls - 32)
            w, lane = (g % 128) // 32, g % 32
            scale = np.multiply(d, d8[:, g], dtype=np.float32)
            partials[w, :, lane] = (scale.astype(np.float64) * dot.astype(np.float64)
                                    + partials[w, :, lane].astype(np.float64)).astype(np.float32)
        result[r] = reduce_u4_partials(partials)
    return result

