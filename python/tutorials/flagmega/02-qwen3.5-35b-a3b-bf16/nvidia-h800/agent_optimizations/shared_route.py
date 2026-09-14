"""Expose an always-active shared expert as an additional sparse route.

This explicitly adopts the sparse expert's rounding schedule for the shared
branch and final expert sum. It is an opt-in numerical optimization, validated
by the requested independent token prefix, not a bitwise graph equivalence.
"""

from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.errors import StageError
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.passes.functions import remove_unused_functions
from triton.flagmega.rules.neutral._utility import make_node


def fuse_shared_route(module, *, fuse_router=False):
    if module.stage != "imported":
        raise StageError("Shared-route fusion requires imported IR", stage=module.stage)
    nodes = module.node_map
    replacements, insertions = {}, {}
    scale_names = ("gate_input_scale", "gate_proj_scale", "down_input_scale", "down_proj_scale",
                   "up_input_scale", "up_proj_scale")
    for root in module.nodes:
        if root.op != "math.add" or not root.effect.is_pure:
            continue
        routed, scaled = (nodes[i] for i in root.inputs)
        if routed.op != SparseExperts.op_name:
            continue
        operands = {p.name: nodes[p.read(routed.inputs)] for p in SparseExperts.input_parameters}
        if scaled.op == SparseExperts.op_name:
            shared_inputs = {p.name: nodes[p.read(scaled.inputs)] for p in SparseExperts.input_parameters}
            shared_ids = shared_inputs["router_expert_ids"]
            if (shared_ids.op != "builtin.splat_const" or shared_ids.attrs["value"] != 0
                    or shared_ids.type.shape[1].fixed_value != 1
                    or shared_inputs["gate_weight"].type.shape[0].fixed_value != 1
                    or any(shared_inputs[name].op != "builtin.splat_const" or shared_inputs[name].attrs["value"] != 1
                           for name in scale_names)):
                continue
            gate = shared_inputs["router_expert_weights"]
            shared_q = shared_inputs["q"]
            shared = {name: shared_inputs[name] for name in ("gate_weight", "up_weight", "down_weight")}
        elif scaled.op == "math.mul":
            down, broadcast = (nodes[i] for i in scaled.inputs)
            if down.op != "math.matmul" or broadcast.op != "tensors.broadcast_to":
                continue
            glu, gate = nodes[down.inputs[0]], nodes[broadcast.inputs[0]]
            if (glu.op != "nn.dense_matmul_glu" or glu.attrs.get("activation") != "silu"
                    or not down.attrs.get("transpose_b") or down.attrs.get("transpose_a", False)):
                continue
            shared_q = nodes[glu.inputs[0]]
            shared = {"gate_weight": nodes[glu.inputs[1]], "up_weight": nodes[glu.inputs[2]],
                      "down_weight": nodes[down.inputs[1]]}
        else:
            continue
        if gate.op != "math.sigmoid" or operands["q"].id != shared_q.id or operands["q"].type.dtype != fm.DType.BFLOAT16:
            continue
        if any(nodes[p.read(routed.inputs)].op != "builtin.splat_const"
               or nodes[p.read(routed.inputs)].attrs["value"] != 1
               for p in SparseExperts.input_parameters if p.name in scale_names):
            continue
        if any(operands[name].type.shape[1:] != (value.type.shape[1:] if value.type.rank == 3 else value.type.shape)
               or operands[name].type.dtype != value.type.dtype for name, value in shared.items()):
            continue
        expert_dim = operands["gate_weight"].type.shape[0]
        token_dim = operands["router_expert_ids"].type.shape[0]
        if not expert_dim.is_fixed or not token_dim.is_fixed or gate.type.shape != (token_dim, fm.dim(1)):
            continue
        experts, tokens = expert_dim.fixed_value, token_dim.fixed_value
        if fuse_router:
            ids_source = operands["router_expert_ids"]
            if ids_source.op != "builtin.get_item":
                continue
            topk = nodes[ids_source.inputs[0]]
            softmax = nodes[topk.inputs[0]] if topk.op == "tensors.top_k" else None
            router_projection = nodes[softmax.inputs[0]] if softmax is not None and softmax.op == "nn.softmax" else None
            gate_projection = nodes[gate.inputs[0]]
            if (router_projection is None or router_projection.op != "math.matmul" or gate_projection.op != "math.matmul"
                    or any(n.inputs[0] != operands["q"].id or not n.attrs.get("transpose_b")
                           or n.attrs.get("transpose_a", False) for n in (router_projection, gate_projection))):
                continue
        prefix = root.id + ".shared_route"
        added = []

        def emit(op, suffix, inputs=(), attrs=None):
            name = prefix + "." + suffix
            if name in nodes:
                raise ValueError(f"Shared-route name collision: {name}")
            node = make_node(op, name, tuple(inputs), attrs or {}, {"formed_by": "SharedExpertRoute"})
            added.append(node)
            return node

        for name, value in shared.items():
            shape = tuple(d.fixed_value for d in value.type.shape)
            expanded = value if value.type.rank == 3 else emit("tensors.reshape", name + ".expanded", (value,), {"shape": (1, *shape)})
            operands[name] = emit("tensors.concat", name + ".bank", (operands[name], expanded), {"axis": 0})
        one = emit("builtin.splat_const", "scales", attrs={"result_type": fm.tensor_type("float32", (experts + 1, 1)), "value": 1.0})
        for name in scale_names:
            operands[name] = one
        if fuse_router:
            from .shared_router import SharedRouter
            weights = emit("tensors.concat", "router_bank",
                           (nodes[router_projection.inputs[1]], nodes[gate_projection.inputs[1]]), {"axis": 0})
            capacity = 1 << experts.bit_length()
            if capacity != experts + 1:
                weights = emit("tensors.pad", "router_bank_padded", (weights,),
                               {"pad_end": (capacity - experts - 1, 0), "pad_value": 0.0})
            projected = emit("math.matmul", "router_projection", (operands["q"], weights),
                          {**router_projection.attrs, "output_data_type": "float32"})
            logits = projected if capacity == experts + 1 else emit(
                "tensors.slice_to_shape", "router_logits", (projected,), {"shape": (tokens, experts + 1)})
            selection = emit(SharedRouter.op_name, "routing", (logits,), {"experts": experts, "k": topk.attrs["k"]})
            operands["router_expert_weights"] = emit("builtin.get_item", "weights", (selection,), {"index": 0})
            operands["router_expert_ids"] = emit("builtin.get_item", "indices", (selection,), {"index": 1})
        else:
            ids = operands["router_expert_ids"]
            shared_id = emit("builtin.splat_const", "index", attrs={"result_type": fm.tensor_type(ids.type.dtype, (tokens, 1)), "value": experts})
            operands["router_expert_ids"] = emit("tensors.concat", "indices", (ids, shared_id), {"axis": 1})
            scores = operands["router_expert_weights"]
            coefficient = gate if gate.type.dtype == scores.type.dtype else emit(
                "tensors.cast", "coefficient", (gate,), {"dtype": scores.type.dtype.value})
            operands["router_expert_weights"] = emit("tensors.concat", "weights", (scores, coefficient), {"axis": 1})
        arguments = tuple(operands[p.name] for p in SparseExperts.input_parameters)
        call = SparseExperts.prepare(arguments, routed.attrs)
        if call.result_type != root.type:
            raise ValueError("Shared-route output changes the public tensor contract")
        replacements[root.id] = replace(root, op=SparseExperts.op_name, inputs=tuple(n.id for n in arguments),
                                         attrs=call.attrs, effect=call.effect,
                                         metadata={**root.metadata, "shared_expert_index": experts,
                                                   "rounding_policy": "sparse-expert-schedule"})
        insertions[root.id] = tuple(added)
    if not replacements:
        return module
    result = replace(module, nodes=tuple(n for original in module.nodes
                                        for n in (*insertions.get(original.id, ()), replacements.get(original.id, original))),
                     metadata={**module.metadata, "shared_expert_routes": tuple(replacements),
                               "shared_route_numerics": "shared branch follows sparse-expert rounding",
                               "shared_router_fused": fuse_router})
    return fm.verify_module(remove_unused_functions(result))
