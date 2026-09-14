"""Explicit compact-owner logits reduction followed by shared-expert routing."""

from dataclasses import replace
from math import prod

from .shared_router import SharedRouter, router_work
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton import kernel_call_renderers as render
from triton.flagmega.codegen.triton.physical_access import (
    emit_local_scalar_offset, emit_logical_coordinate, emit_storage_pointer,
)
from triton.flagmega.codegen.triton.reduction_domain import local_reduction_domain
from triton.flagmega.errors import CodegenError, IRSchemaError
from triton.flagmega.ir.distributed_type import local_tensor_type
from triton.flagmega.ir.ops.core import (
    OpDefinition, OpCostFactors, PythonCall, NodeRef,
    input_parameter, attribute_parameter, op_definition,
)
from triton.flagmega.ir.type_pattern import is_tensor
from triton.flagmega.pattern_match import is_call, is_op, wildcard
from triton.flagmega.rules.core import RewriteResult, RewriteRule


@op_definition("local.gather_reduce_shared_router", display_name="Local.GatherReduceSharedRouter")
class GatherReduceSharedRouter(OpDefinition):
    value = input_parameter(is_tensor(), memory_effect=fm.MemoryEffect.CHIP_READ.across_partial_owners())
    materialized_type = attribute_parameter()
    experts = attribute_parameter()
    k = attribute_parameter()
    supports_broadcast_lifting = False

    @classmethod
    def normalize_attrs(cls, attrs):
        attrs = super().normalize_attrs(attrs)
        SharedRouter.normalize_attrs({key: attrs[key] for key in ("experts", "k")})
        if not isinstance(attrs["materialized_type"], fm.DistributedType):
            raise IRSchemaError("GatherReduceSharedRouter requires a distributed materialized_type")
        return attrs

    @classmethod
    def infer_type(cls, inputs, attrs):
        source = cls.value.type_of(inputs)
        materialized = attrs["materialized_type"]
        if (not isinstance(source, fm.DistributedType)
                or source.tensor != materialized.tensor or source.placement != materialized.placement
                or source.partial is None or source.partial.reduce_op != fm.ReduceOp.SUM
                or not source.partial.axes or materialized.partial is not None
                or source.exclusive or materialized.exclusive):
            raise IRSchemaError("GatherReduceSharedRouter requires a Sum-partial source on the materialized placement")
        tensor = source.tensor
        scalar = tensor.dtype.elem_type if isinstance(tensor.dtype, fm.VectorType) else tensor.dtype
        lanes = getattr(tensor.dtype, "lane_count", 1)
        if (tensor.rank != 2 or scalar != fm.DType.FLOAT32
                or not tensor.shape[1].is_fixed or tensor.shape[1].fixed_value * lanes < attrs["experts"] + 1
                or materialized.axis_policies[1] != fm.SBP.broadcast()):
            raise IRSchemaError("GatherReduceSharedRouter needs complete FP32 expert and gate columns")
        logical = replace(materialized, tensor=fm.tensor_type("float32", (tensor.shape[0], attrs["experts"] + 1)))
        return SharedRouter.infer_type((fm.Node("<logits>", "builtin.var", (), logical),), attrs)

    @classmethod
    def evaluate(cls, node, arguments, context):
        value = cls.value.read(arguments)
        # The evaluator holds the logical sum; only device lowering reads owners.
        logical = value.reshape(value.shape[0], -1)[:, :node.attrs["experts"] + 1]
        return SharedRouter.evaluate(node, (logical,), context)

    @classmethod
    def cost(cls, node):
        metric = SharedRouter.cost(node)
        return replace(metric, bytes_read=None, communication_bytes=None, synchronizations=1,
                       notes=("compact-owner-sum-stable-topk-no-logit-materialization",))

    @classmethod
    def cost_factors(cls, inputs, attrs, return_type):
        source = cls.value.type_of(inputs)
        rows = local_tensor_type(return_type.fields[0]).shape[0]
        if not rows.is_fixed:
            return None
        rows = rows.fixed_value
        fan_in = prod(source.placement.hierarchy[axis] for axis in source.partial.axes)
        return OpCostFactors(
            elementwise_operations=router_work(attrs, rows) + rows * (attrs["experts"] + 1) * (fan_in - 1),
            chip_global_memory_load_bytes=rows * (attrs["experts"] + 1) * fan_in * 4,
            block_local_memory_store_bytes=rows * (attrs["k"] + 1) * 12,
            grid_synchronizations=1,
        )

    @classmethod
    def python_call(cls, node):
        return PythonCall("__import__('agent_optimizations.shared_router_collective', fromlist=['gather_reduce_shared_router']).gather_reduce_shared_router",
                          (NodeRef(node.inputs[0]),), {**node.attrs, "name": node.id, "metadata": dict(node.metadata)})


def gather_reduce_shared_router(value, *, materialized_type, experts, k, name=None, metadata=None):
    return GatherReduceSharedRouter.construct(value, materialized_type=materialized_type, experts=experts, k=k,
                                              name=name, metadata=metadata)


def fuse_shared_router_rule():
    return RewriteRule("fuse_gather_reduce_shared_router",
                       is_call(is_op(SharedRouter.op_name), wildcard("value"), name="router"), _rewrite)


def _rewrite(match, module):
    router = match["router"]
    if not isinstance(match["value"].type, fm.DistributedType):
        return router
    users = {node.id: [] for node in module.nodes}
    for node in module.nodes:
        for value in node.inputs:
            users[value].append(node.id)
    for function in module.functions:
        for value in function.outputs:
            users[value].append(f"@{function.name}:return")
    node = match["value"]
    removed = []
    if node.op == "tensors.slice_to_shape":
        if len(users[node.id]) != 1:
            return router
        parent = module.node_map[node.inputs[0]]
        if parent.type.tensor.shape[0] != node.type.tensor.shape[0]:
            return router
        removed.append(node.id)
        node = parent
    if node.op == "tensors.bitcast":
        if len(users[node.id]) != 1 or node.type.tensor.dtype != fm.DType.FLOAT32:
            return router
        removed.append(node.id)
        node = module.node_map[node.inputs[0]]
    if node.op != "distributed.boxing" or len(users[node.id]) != 1:
        return router
    source = module.node_map[node.inputs[0]]
    attrs = {**router.attrs, "materialized_type": node.type}
    try:
        result_type = GatherReduceSharedRouter.infer_call_type((source,), attrs)
        effect = GatherReduceSharedRouter.infer_effect((source,), attrs)
    except (IRSchemaError, TypeError, ValueError):
        return router
    if result_type != router.type:
        return router
    return RewriteResult(replace(router, op=GatherReduceSharedRouter.op_name, inputs=(source.id,),
                                 type=result_type, effect=effect, attrs=attrs,
                                 metadata={**router.metadata, "fused_logits_materialization": node.id}),
                         removed_ids=(*removed, node.id))


def encode_collective(raw):
    source, weights, ids = render._buffer(raw, "inputs", "value"), render._buffer(raw, "outputs", "result_0"), render._buffer(raw, "outputs", "result_1")
    abi = source["abi"]
    if abi.get("storage_kind") != "compact_per_owner":
        raise CodegenError("GatherReduceSharedRouter needs compact per-owner input storage")
    stride = int(abi.get("component_stride_scalar_elements", 0))
    if stride <= 0:
        raise CodegenError("GatherReduceSharedRouter requires an owner component stride")
    attrs = raw["semantic_attrs"]
    tile = 1 << (attrs["experts"] - 1).bit_length()
    domain = local_reduction_domain(weights["abi"], (1,), tile)
    row = emit_logical_coordinate(weights["abi"], 0, domain["coordinates"])
    distribution = abi["distributed_type"]
    axes = tuple(distribution["partial"]["axes"])
    hierarchy = tuple(distribution["placement"]["hierarchy"])
    lanes = int(abi["scalar_lane_count"])
    partial_owner = render._group_owner_expression_for_axes(
        abi, axes, "_sr_partial_member", preserved_coordinates=tuple("0" for _ in hierarchy))

    def source_offset(column):
        coordinates, owner = render._compact_source_coordinates(abi, (row, f"(({column}) // {lanes})"))
        offset = emit_local_scalar_offset(abi, coordinates,
                                         lane_coordinate=None if lanes == 1 else f"(({column}) % {lanes})")
        return f"(({partial_owner}) + ({owner})) * {stride} + ({offset})"

    def output_offset(output, column):
        return emit_local_scalar_offset(output["abi"], (domain["coordinates"][0], column))

    return {**domain, "tile": tile, "experts": attrs["experts"], "k": attrs["k"],
            "selected_tile": 1 << (attrs["k"] - 1).bit_length(), "fan_in": prod(hierarchy[axis] for axis in axes),
            "source": emit_storage_pointer(abi, source["runtime_argument"]),
            "source_offset": source_offset("_fm_offsets"), "gate_offset": source_offset(str(attrs["experts"])),
            "weights": render._pointer(weights), "ids": render._pointer(ids),
            "weights_offset": output_offset(weights, "_sr_selected_offsets"),
            "ids_offset": output_offset(ids, "_sr_selected_offsets"),
            "gate_weight_offset": output_offset(weights, str(attrs["k"])),
            "gate_id_offset": output_offset(ids, str(attrs["k"])),
            "weights_active": render._canonical_writer_active(weights["abi"]),
            "ids_active": render._canonical_writer_active(ids["abi"])}
