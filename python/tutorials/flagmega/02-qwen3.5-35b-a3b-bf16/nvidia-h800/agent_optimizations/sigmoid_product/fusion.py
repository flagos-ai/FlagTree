"""Fuse a proven FP32 gate chain in the BF16 value producer's local domain."""

from dataclasses import replace
from triton.flagmega import ir as fm
from triton.flagmega.errors import IRSchemaError, StageError
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.ops.distributed.sharded_view import ShardedView
from triton.flagmega.ir.ops.tensors.pack import Pack
from triton.flagmega.ir.ops.tensors.bitcast import Bitcast
from triton.flagmega.passes.tir.fuse_gather_reduce_norm_apply import _users
from agent_optimizations.sigmoid_product.op import SigmoidProduct


def _element(value):
    dtype = tensor_of(value.type).dtype
    return dtype.elem_type if isinstance(dtype, fm.VectorType) else dtype


def _last_axis(attrs, rank, field):
    axes = attrs.get(field, (attrs.get("axis", -1), ))
    return bool(axes) and all(axis in (-1, rank - 1) for axis in axes)


def _same_element_bitcast(node, nodes):
    return (node.op == "tensors.bitcast" and node.effect.is_pure
            and _element(node) == _element(nodes[node.inputs[0]]))


def _is_gate_lane_view(node, nodes):
    if not node.effect.is_pure:
        return False
    if node.op == "tensors.pack":
        return _last_axis(node.attrs, tensor_of(node.type).rank, "axes")
    if node.op != "tensors.bitcast":
        return False
    source = nodes[node.inputs[0]]
    dtype = tensor_of(node.type).dtype
    if (tensor_of(source.type).dtype != fm.DType.FLOAT32 or not isinstance(dtype, fm.VectorType)
            or dtype.elem_type != fm.DType.FLOAT32):
        return False
    try:
        return Pack.infer_type((source,), {"lanes": dtype.lanes, "axis": -1}) == node.type
    except IRSchemaError:
        return False


def fuse_sigmoid_product(module):
    if module.stage not in {"frozen_constants", "tuple_boxing_lowered"}:
        raise StageError("SigmoidProduct fusion requires the pre-TIR proposal boundary.", stage=module.stage)
    fm.verify_module(module)
    nodes, users = module.node_map, _users(module)
    inserted, replacements, removable = {}, {}, set()
    known_ids = set(nodes)

    def view_source(node, matched):
        while node.op == "distributed.sharded_view" and node.effect.is_pure:
            matched.add(node.id)
            node = nodes[node.inputs[0]]
        return node

    def cast_source(node, matched):
        node = view_source(node, matched)
        # Bitcast changes only the final-axis packet grouping here. It is not
        # a numeric cast, transpose, or cross-axis Pack. Keep the original
        # narrow producer as the fused kernel's compute domain.
        while _same_element_bitcast(node, nodes):
            matched.add(node.id)
            node = view_source(nodes[node.inputs[0]], matched)
        if node.op not in {"ntt.vectorized_cast", "tensors.cast"} or not node.effect.is_pure:
            return None
        source = nodes[node.inputs[0]]
        if _element(node) != fm.DType.FLOAT32 or _element(source) != fm.DType.BFLOAT16:
            return None
        if node.op == "ntt.vectorized_cast" and not _last_axis(node.attrs, tensor_of(node.type).rank, "vectorize_axes"):
            return None
        matched.add(node.id)
        return view_source(source, matched)

    for root in module.nodes:
        if (root.op != "ntt.vectorized_cast" or not root.effect.is_pure or _element(root) != fm.DType.BFLOAT16
                or not _last_axis(root.attrs,
                                  tensor_of(root.type).rank, "vectorize_axes")):
            continue
        seed = set()
        product = view_source(nodes[root.inputs[0]], seed)
        if (product.op != "math.vectorized_binary" or product.attrs["binary_op"] != "mul" or not product.effect.is_pure
                or _element(product) != fm.DType.FLOAT32):
            continue
        for gate_index in (1, 0):
            matched = {*seed, product.id}
            gate = view_source(nodes[product.inputs[gate_index]], matched)
            if _is_gate_lane_view(gate, nodes):
                matched.add(gate.id)
                gate = view_source(nodes[gate.inputs[0]], matched)
            is_sigmoid = (gate.op == "math.sigmoid" or
                          gate.op == "math.vectorized_unary" and gate.attrs["unary_op"] == "sigmoid")
            if not is_sigmoid or not gate.effect.is_pure or _element(gate) != fm.DType.FLOAT32:
                continue
            matched.add(gate.id)
            gate_input = cast_source(nodes[gate.inputs[0]], matched)
            value = cast_source(nodes[product.inputs[1 - gate_index]], matched)
            if value is None or gate_input is None:
                continue
            tensor = tensor_of(value.type)
            if (not isinstance(tensor.dtype, fm.VectorType)
                    or len(tensor.dtype.lanes) != 1 or _element(gate_input) != fm.DType.BFLOAT16):
                continue
            try:
                restored_type = Bitcast.prepare((value,), {"dtype": tensor_of(root.type).dtype}).result_type
            except IRSchemaError:
                continue
            if tensor_of(restored_type) != tensor_of(root.type):
                continue
            additions = []
            prefix = root.id + ".sigmoid_product"

            def make(definition, arguments, attrs, identity):
                if identity in known_ids:
                    raise ValueError(f"SigmoidProduct node name collides: {identity}")
                prepared = definition.prepare(tuple(arguments), attrs)
                node = fm.Node(identity, definition.op_name, tuple(n.id for n in arguments), prepared.result_type,
                               attrs=prepared.attrs, effect=prepared.effect)
                additions.append(node)
                return node

            try:
                packed = gate_input
                if not isinstance(tensor_of(packed.type).dtype, fm.VectorType):
                    packed = make(Pack, (gate_input, ), {"axes": (-1, ), "lanes": tensor.dtype.lanes},
                                  prefix + ".gate_pack")
                if tensor_of(packed.type) != tensor:
                    continue
                if packed.type != value.type:
                    packed = make(ShardedView, (packed, ), {"new_type": value.type}, prefix + ".gate_view")
                fused = make(SigmoidProduct, (value, packed), {}, prefix)
                output = fused
                if tensor_of(fused.type) != tensor_of(root.type):
                    output = make(Bitcast, (fused,), {"dtype": tensor_of(root.type).dtype}, prefix + ".output_view")
                if output.type == root.type:
                    restored = replace(root, op=output.op, inputs=output.inputs, attrs=output.attrs,
                                       effect=output.effect)
                    additions.pop()
                else:
                    prepared = ShardedView.prepare((output, ), {"new_type": root.type})
                    restored = replace(root, op=ShardedView.op_name, inputs=(output.id, ), attrs=prepared.attrs,
                                       effect=prepared.effect)
            except IRSchemaError:
                continue
            inserted[root.id] = tuple(additions)
            known_ids.update(n.id for n in additions)
            replacements[root.id] = restored
            removable.update(matched)
            break
    while True:
        retained = {
            name
            for name in removable
            if any(user not in removable and user not in replacements for user in users[name])
        }
        if not retained:
            break
        removable.difference_update(retained)
    changed = removable | replacements.keys()
    obsolete = {p.id for p in module.selection_points if p.owner in changed}
    result = replace(
        module, nodes=tuple(n for original in module.nodes if original.id not in removable
                            for n in (*inserted.get(original.id, ()), replacements.get(original.id, original))),
        selection_points=tuple(p for p in module.selection_points if p.id not in obsolete),
        selections=tuple(s for s in module.selections if s.point_id not in obsolete))
    return fm.verify_module(result)
