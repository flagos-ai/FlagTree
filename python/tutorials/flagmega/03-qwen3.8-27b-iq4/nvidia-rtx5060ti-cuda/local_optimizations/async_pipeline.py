"""Normal CPP uniform IQ4 transport: guarded selection and typed Shared ABI.

Private a006 initial scope is four warps and the actual dense 4x8 decode map.
The separate experimental block256 GDN interface is not used by this package.
"""
import ast
from dataclasses import replace
from math import prod
import os

from triton.flagmega.codegen.triton.candidates import (
    BlockFp8CandidateProvider, MatmulGluCandidateProvider, TritonCandidateProposal,
    TritonCandidateProviderRegistry,
)
from triton.flagmega.codegen.triton.gdn_head_gates import _reject_alias
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DType, DistributedType, VectorType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import local_shape
from triton.flagmega.targets import get_target
from .async_catalog import PREFIX, ROWS, CHUNK, ROW_BYTES, catalog

WARPS = 4
MESH = (4, 8)


def _shape(tensor):
    return tuple(d.fixed_value for d in tensor.shape) if all(d.is_fixed for d in tensor.shape) else None


class AsyncCandidateProvider:
    def __init__(self, original, family):
        self.original, self.family, self.op_names = original, family, original.op_names

    def propose(self, node, context):
        # Keep extra native candidates outside every other semantic domain.
        ordinary = replace(context.implementation_model,
                           implementations=tuple(i for i in context.implementation_model.implementations
                                                 if not i.id.startswith(PREFIX + ".")))
        base_context = replace(context, implementation_model=ordinary)
        attrs = node.attrs
        encoding = attrs.get("weight_encoding") if self.family == "block_fp8" else attrs.get("gate_weight_encoding")
        if encoding != "IQ4_XS":
            return self.original.propose(node, base_context)
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        weight = tensor_of(context.module.node_map[node.inputs[1]].type)
        out = node.type
        shape = _shape(value)
        if (context.codegen_backend != "cpp" or not context.cooperative_grid
                or getattr(context.target, "codegen_architecture", None) != "sm120"
                or not isinstance(out, DistributedType) or out.partial is not None or out.exclusive is not None
                or out.placement.hierarchy != MESH or out.placement.hierarchy_levels != "bb"
                or shape is None or shape[0] != 1 or len(shape) != 2 or value.dtype != DType.BFLOAT16
                or _shape(weight) is None or len(weight.shape) != 2 or weight.dtype != DType.UINT8
                or (self.family == "matmul_glu" and attrs.get("up_weight_encoding") != "IQ4_XS")):
            raise CodegenError("Uniform IQ4 pipeline requires native single-token BF16 decode on CPP SM120/dense4x8")
        k = shape[-1]
        matches = [i for i in context.implementations(self.family)
                   if i.id.startswith(PREFIX + ".") and i.parameters["required_k"] == k]
        if len(matches) != 1 or _shape(weight)[-1] != k // 256 * 136:
            raise CodegenError("Uniform IQ4 pipeline has no exact native K/row-byte candidate")
        output_shape = local_shape(out)
        if len(output_shape) != 2 or not all(d.is_fixed for d in output_shape) or output_shape[0].fixed_value != 1:
            raise CodegenError("Uniform IQ4 pipeline requires static one-row output geometry")
        lanes = prod(out.tensor.dtype.lanes) if isinstance(out.tensor.dtype, VectorType) else 1
        rows = output_shape[-1].fixed_value * lanes
        if rows <= 0:
            raise CodegenError("Uniform IQ4 pipeline requires a nonempty CTA row interval")
        barriers = 2 + ((rows + ROWS - 1) // ROWS) * (1 + k // CHUNK)
        candidate = context.configure_implementation(matches[0],
            facts={"internal_cta_barriers": barriers},
            semantic_parameters={"local_scalar_rows": rows})
        return TritonCandidateProposal((candidate,), candidate.id)


def enable():
    if "FLAGMEGA_CPP_BLOCK_THREADS" in os.environ:
        raise CodegenError("Uniform IQ4 pipeline owns preplanning warp count; no block-thread override")
    target = get_target("nvidia-sm120")
    model = target.triton_implementation_model
    if any(i.id.startswith(PREFIX + ".") for i in model.implementations):
        raise CodegenError("Uniform IQ4 pipeline was already installed")
    policy = target.tir_selection_policy
    registry = TritonCandidateProviderRegistry()
    counts = {"block_fp8": 0, "matmul_glu": 0}
    for provider in policy.registry.providers:
        family = "block_fp8" if isinstance(provider, BlockFp8CandidateProvider) else "matmul_glu" if isinstance(provider, MatmulGluCandidateProvider) else None
        if family:
            provider = AsyncCandidateProvider(provider, family)
            counts[family] += 1
        registry.add(provider)
    if counts != {"block_fp8": 1, "matmul_glu": 1}:
        raise CodegenError("Uniform IQ4 pipeline requires the ordinary matrix providers")
    target.triton_implementation_model = TritonImplementationModel(
        (*model.implementations, *catalog(WARPS)), model.preferences, model.name + "+local-uniform-iq4-w4-v1")
    target.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)


def _binding(call, group, formal):
    return next(p for p in call[group] if p["formal"] == formal)["buffers"][0]


def _expression(text):
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr,
               ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.USub)
    try:
        tree = ast.parse(str(text), mode="eval")
    except SyntaxError as error:
        raise CodegenError("Uniform IQ4 pipeline requires a static scalar row expression") from error
    names = {"local_n_offsets", "shard_x", "shard_y", "shard_coord0", "shard_coord1", "shard_index"}
    if any(not isinstance(n, allowed) or isinstance(n, ast.Name) and n.id not in names for n in ast.walk(tree)):
        raise CodegenError("Uniform IQ4 row expressions must be CTA/lane independent")
    return compile(tree, "<uniform-iq4-owner>", "eval")


def _workspace(call):
    glu = call["family"] == "matmul_glu"
    k = call["k"]
    components = 2 if glu or call["family"] == "interleaved_projection" else 1
    fields = 2 if glu else 1
    expected = {
        "activation_q8": ((components, k // 32, 36), (k // 32 * 36, 36, 1), components * k // 32 * 36),
        "native_weight_stages": ((2, fields, ROWS, ROW_BYTES),
                                 (fields * ROWS * ROW_BYTES, ROWS * ROW_BYTES, ROW_BYTES, 1),
                                 2 * fields * ROWS * ROW_BYTES),
    }
    actual = call.get("shared_workspaces", ())
    if len(actual) != 2 or {w.get("name") for w in actual} != set(expected):
        raise CodegenError("Uniform IQ4 pipeline requires exactly its two typed Shared workspaces")
    by_name = {}
    for w in actual:
        shape, strides, nbytes = expected[w["name"]]
        offset = w.get("offset_bytes")
        if (w.get("dtype") != "uint8" or tuple(w.get("shape", ())) != shape
                or tuple(w.get("strides", ())) != strides or w.get("nbytes") != nbytes
                or w.get("alignment_bytes") != 16 or type(offset) is not int or offset < 0 or offset % 16
                or not w.get("physical_buffer") or w.get("matrix_compatible", False)):
            raise CodegenError("Uniform IQ4 Shared ABI differs from its exact shape/stride/alignment/offset contract")
        by_name[w["name"]] = w
    q8, weights = by_name["activation_q8"], by_name["native_weight_stages"]
    if not (q8["offset_bytes"] + q8["nbytes"] <= weights["offset_bytes"]
            or weights["offset_bytes"] + weights["nbytes"] <= q8["offset_bytes"]):
        raise CodegenError("Uniform IQ4 Q8 and native weight live spans overlap")
    return {"local_iq4_async_q8_offset": q8["offset_bytes"],
            "local_iq4_async_weight_offset": weights["offset_bytes"],
            "local_iq4_async_arena_end": max(w["offset_bytes"] + w["nbytes"] for w in actual)}


def validate_call(call, descriptor):
    family, k = call.get("family"), call.get("k")
    valid_ids = {i.id: i for i in catalog(WARPS)}
    implementation = valid_ids.get(call.get("implementation"))
    if implementation is None or implementation.family != family or implementation.parameters["required_k"] != k:
        raise CodegenError("Uniform IQ4 pipeline implementation identity was not selected before planning")
    if (descriptor.get("platform") != "nvidia" or descriptor.get("target_architecture") != "sm120"
            or tuple(descriptor.get("mesh_hierarchy", ())) != MESH or tuple(descriptor.get("grid", ())) != (32, 1, 1)
            or descriptor.get("block_threads") != 128 or descriptor.get("num_warps") != WARPS
            or descriptor.get("warp_size") != 32 or descriptor.get("pipeline_schedule") is not None
            or call.get("variant") != "simt" or call.get("weight_encoding") != "IQ4_XS"
            or str(call.get("participation_active")) != "True" or call.get("transfer_pipeline") is not None
            or call.get("pipeline_channels") or call.get("internal_grid_barriers") != 0):
        raise CodegenError("Uniform IQ4 pipeline requires its actual static CPP4x8/block128 uniform schedule")
    params, facts = call.get("parameters", {}), call.get("facts", {})
    if any(params.get(k) != v for k, v in implementation.parameters.items()):
        raise CodegenError("Uniform IQ4 pipeline parameters differ from their versioned implementation")
    if any(facts.get(k) != v for k, v in implementation.facts.items()):
        raise CodegenError("Uniform IQ4 pipeline facts differ from their versioned implementation")
    rows = call["local_n_capacity"]
    barriers = 2 + ((rows + ROWS - 1) // ROWS) * (1 + k // CHUNK)
    if (params.get("local_scalar_rows") != rows or facts.get("internal_cta_barriers") != barriers
            or set(call.get("requires", facts.get("requires", ()))) != {"async_copy", "cooperative_grid"}):
        raise CodegenError("Uniform IQ4 pipeline row/barrier/capability contract changed")
    value = _binding(call, "inputs", "value")
    value_abi = value["abi"]
    if (tuple(value_abi.get("logical_shape", ())) != (1, k)
            or tuple(value_abi.get("local_capacity_shape", ())) != (1, k)
            or tuple(value_abi.get("scalar_storage_strides", ())) != (k, 1)
            or value_abi.get("scalar_dtype") != "bfloat16" or value_abi.get("scalar_lane_count") != 1):
        raise CodegenError("Uniform IQ4 pipeline requires one complete contiguous scalar BF16 activation row")
    weights = [_binding(call, "inputs", name) for name in
               (("gate_weight", "up_weight") if family == "matmul_glu" else ("weight",))]
    row_bytes = k // 256 * 136
    total_rows = None
    for weight in weights:
        abi = weight["abi"]
        shape = tuple(abi.get("logical_shape", ()))
        if (len(shape) != 2 or shape[0] <= 0 or shape[1] != row_bytes
                or tuple(abi.get("scalar_storage_strides", ())) != (row_bytes, 1)
                or abi.get("scalar_dtype") != "uint8" or abi.get("scalar_lane_count") != 1
                or abi.get("coordinate_space") != "canonical_global"):
            raise CodegenError("Uniform IQ4 pipeline requires canonical global native row payloads")
        if total_rows is not None and total_rows != shape[0]:
            raise CodegenError("Uniform GLU weights have unequal row counts")
        total_rows = shape[0]
    result = _binding(call, "outputs", "result")
    result_abi = result["abi"]
    if (result_abi.get("scalar_dtype") not in ("bfloat16", "float32")
            or family == "matmul_glu" and result_abi.get("scalar_dtype") != "bfloat16"):
        raise CodegenError("Uniform IQ4 pipeline requires original BF16 projections or F32 logits")
    space = result_abi.get("coordinate_space")
    if (space not in ("canonical_global", "local")
            or result_abi.get("memory_sharing_scope") != ("chip" if space == "canonical_global" else "block")):
        raise CodegenError("Uniform IQ4 output requires canonical chip storage or a private CTA component")
    shape = result_abi["logical_shape"] if space == "canonical_global" else result_abi["local_capacity_shape"]
    strides = result_abi["scalar_storage_strides"]
    lanes = result_abi["scalar_lane_count"]
    if (len(shape) != 2 or len(strides) != 2 or shape[0] != 1 or lanes not in (1, 8)
            or any(type(n) is not int or n <= 0 for n in (*shape, *strides))
            or prod(result_abi["logical_shape"]) * lanes != total_rows):
        raise CodegenError("Uniform IQ4 output shape/stride does not cover its native row domain")
    output_span = lanes + sum((n - 1) * stride for n, stride in zip(shape, strides))
    for operand in (value, *weights):
        _reject_alias(result, operand)
    # Recompute every ordinary row/pointer/mask expression from the actual ABI.
    from triton.flagmega.codegen.triton import kernel_call_renderers as encoders
    ordinary = encoders._FAMILY_ENCODERS[family](call)
    for key, expected in ordinary.items():
        if call.get(key) != expected:
            raise CodegenError("Uniform IQ4 descriptor differs from ordinary ABI rendering: " + key)
    mask = call["n_mask"] if family == "block_fp8" else f"local_n_offsets < ({call['active_n']})"
    expressions = {"mask": _expression(mask), "row": _expression(call["weight_row_offset"]),
                   "store": _expression(call["destination_offset"])}
    if family == "matmul_glu":
        expressions["up"] = _expression(call["up_weight_row_offset"])
    coverage = 0
    logical_rows = set()
    global_stores = set()
    for y in range(MESH[0]):
        for x in range(MESH[1]):
            env = dict(shard_y=y, shard_x=x, shard_coord0=y, shard_coord1=x, shard_index=y*MESH[1]+x)
            seen = set()
            for row in range(((rows + ROWS - 1) // ROWS) * ROWS):
                env["local_n_offsets"] = row
                active = eval(expressions["mask"], {"__builtins__": {}}, env)
                if row == 0 and not active or row >= rows and active:
                    raise CodegenError("Uniform IQ4 copy mask lacks its safe first row or masks tails incorrectly")
                if not active:
                    continue
                byte = eval(expressions["row"], {"__builtins__": {}}, env)
                if byte % row_bytes or not 0 <= byte < total_rows * row_bytes:
                    raise CodegenError("Uniform IQ4 copy row is misaligned or outside its native matrix")
                logical_row = byte // row_bytes
                if logical_row in logical_rows:
                    raise CodegenError("Uniform IQ4 global native rows have duplicate CTA owners")
                logical_rows.add(logical_row)
                if "up" in expressions and eval(expressions["up"], {"__builtins__": {}}, env) != byte:
                    raise CodegenError("Uniform GLU fields disagree on native row ownership")
                offset = eval(expressions["store"], {"__builtins__": {}}, env)
                if not 0 <= offset < output_span or offset in seen:
                    raise CodegenError("Uniform IQ4 warp rows do not own unique scalar outputs")
                if space == "canonical_global":
                    if offset in global_stores:
                        raise CodegenError("Uniform IQ4 canonical output has duplicate CTA writers")
                    global_stores.add(offset)
                seen.add(offset)
                coverage += 1
    if logical_rows != set(range(total_rows)) or space == "canonical_global" and len(global_stores) != total_rows:
        raise CodegenError("Uniform IQ4 row/output coverage has a hole")
    workspace = _workspace(call)
    return {**workspace, "local_iq4_async": True, "local_iq4_async_warps": WARPS,
            "local_iq4_async_barriers": barriers, "local_iq4_async_owner_rows": coverage}


def configure_descriptor(descriptor):
    # The ordinary flat event lists are serialized synchronous calls. No role
    # pipeline, dynamic loop/event kind, or conditional participation is admitted.
    schedules = [descriptor["entry_events"]]
    for function in descriptor["device_functions"]:
        schedule = function["schedule"]
        if (schedule.get("schema") != "flagmega.triton-pipeline-source-schedule/v3"
                or any(schedule.get(key) for key in ("stages", "handoffs", "producer_events", "auxiliary_events",
                                                     "trip_scoped_drains", "trip_scoped_closes"))):
            raise CodegenError("Uniform IQ4 arena reuse rejects role/overlap schedules")
        schedules.append(schedule["consumer_events"])
    for events in schedules:
        for event in events:
            if (event.get("kind") not in ("barrier", "function_call", "tir.kernel_call")
                    or event.get("pipeline_channels")
                    or event.get("participation_active", "True") != "True"):
                raise CodegenError("Uniform IQ4 arena reuse requires static sequential fully participating calls")
    sites = []
    for call in descriptor["render_calls"]:
        if call.get("family") not in ("block_fp8", "matmul_glu") or call.get("weight_encoding") != "IQ4_XS":
            continue
        fields = validate_call(call, descriptor)
        call.update(fields)
        sites.append({"symbol": call["symbol"], "implementation": call["implementation"],
                      "shared_workspaces": call["shared_workspaces"], **fields})
    if not sites:
        raise CodegenError("Explicit uniform IQ4 recipe selected no native matrix sites")
    legacy = descriptor["local_iq4_shared_bytes"]
    arena = max(legacy, *(s["local_iq4_async_arena_end"] for s in sites))
    if arena > min(49152, descriptor["launch_resource_limits"]["max_shared_memory_bytes"]):
        raise CodegenError("Uniform IQ4 typed arena exceeds the 49152B static/target Shared cap")
    descriptor["local_iq4_shared_bytes"] = arena
    descriptor["local_iq4_recipe"]["async_sites"] = sites
    descriptor["local_iq4_recipe"]["scratch"].update(
        nbytes=arena, legacy_Q8_max_bytes=legacy, static_Shared_limit_bytes=49152,
        aliases="one arena: legacy Q8 at zero or typed Q8/weight offsets in sequential call frames",
        planned_spans="async_sites[].shared_workspaces; preserve each offset and physical buffer identity",
        resource_admission="full compiled static Shared plus actual cooperative occupancy pending")
