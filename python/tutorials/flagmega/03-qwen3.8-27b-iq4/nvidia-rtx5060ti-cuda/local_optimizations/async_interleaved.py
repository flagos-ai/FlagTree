"""Separate explicit dual interleaved extension of the uniform128 pipeline."""
from collections import Counter
from dataclasses import replace

from triton.flagmega.codegen.triton.candidates import (
    InterleavedProjectionCandidateProvider, TritonCandidateProposal, TritonCandidateProviderRegistry,
)
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.codegen.triton.interleaved_projection_renderer import interleaved_projection_call
from triton.flagmega.errors import CodegenError
from triton.flagmega.ir import DType, DistributedType, T, tensor_type
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.targets import get_target
from .async_catalog import candidate
from .async_pipeline import _binding, _expression, _reject_alias, _shape, _workspace

IMPLEMENTATION = "tir.local_iq4_async.interleaved_projection.r16.k5120.w4.v1"
ROWS = 1536
BARRIERS = 2 + (ROWS // 16) * 6


def implementation():
    base = candidate("block_fp8", 5120, 4)
    return replace(base, id=IMPLEMENTATION, family="interleaved_projection",
                   parameters={**base.parameters, "activation_components": 2,
                               "local_scalar_rows": ROWS},
                   contract={**base.contract, "activation_quantization": "existing_local_q8_residual2"},
                   facts={**base.facts, "internal_cta_barriers": BARRIERS},
                   shared_workspaces=(T.shared_workspace_descriptor("activation_q8",
                       tensor_type("uint8", (2, 160, 36)), 16), base.shared_workspaces[1]))


class Provider:
    def __init__(self, original):
        self.original, self.op_names = original, original.op_names

    def propose(self, node, context):
        value = tensor_of(context.module.node_map[node.inputs[0]].type)
        fields = getattr(node.type, "fields", ())
        if (context.codegen_backend != "cpp" or not context.cooperative_grid
                or getattr(context.target, "codegen_architecture", None) != "sm120"
                or node.attrs.get("weight_encoding") != "IQ4_XS"
                or _shape(value) != (1, 5120) or value.dtype != DType.BFLOAT16
                or len(fields) != 2 or any(not isinstance(t, DistributedType)
                    or t.placement.hierarchy != (4, 8) or t.placement.hierarchy_levels != "bb"
                    or t.partial is not None or t.exclusive is not None for t in fields)):
            raise CodegenError("Async interleaved requires its exact CPP4x8/native IQ4 single-token field domain")
        selected = context.configure_implementation(implementation())
        return TritonCandidateProposal((selected,), selected.id)


def enable():
    target = get_target("nvidia-sm120")
    model, policy = target.triton_implementation_model, target.tir_selection_policy
    if model.implementation(IMPLEMENTATION) is not None:
        raise CodegenError("Async interleaved already installed")
    registry = TritonCandidateProviderRegistry()
    count = 0
    for provider in policy.registry.providers:
        if isinstance(provider, InterleavedProjectionCandidateProvider):
            provider = Provider(provider); count += 1
        registry.add(provider)
    if count != 1:
        raise CodegenError("Async interleaved requires the ordinary field projection provider")
    target.triton_implementation_model = TritonImplementationModel(
        (*model.implementations, implementation()), model.preferences, model.name + "+async-interleaved-dual-v1")
    target.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)


def validate_call(call, descriptor):
    impl = implementation()
    if (call.get("implementation") != IMPLEMENTATION or call.get("variant") != "simt"
            or call.get("local_m") != 1 or call.get("k") != 5120 or call.get("local_n_capacity") != ROWS
            or descriptor.get("platform") != "nvidia" or descriptor.get("target_architecture") != "sm120"
            or tuple(descriptor.get("mesh_hierarchy", ())) != (4, 8)
            or tuple(descriptor.get("grid", ())) != (32, 1, 1)
            or descriptor.get("block_threads") != 128 or descriptor.get("num_warps") != 4
            or descriptor.get("warp_size") != 32 or descriptor.get("pipeline_schedule") is not None
            or call.get("transfer_pipeline") is not None or call.get("pipeline_channels")
            or call.get("internal_grid_barriers") != 0 or call.get("participation_active") != "True"
            or call.get("weight_encoding") != "IQ4_XS" or not call.get("local_iq4_dual")):
        raise CodegenError("Async interleaved requires its actual uniform128 dual/native field contract")
    if (any(call.get("parameters", {}).get(k) != v for k, v in impl.parameters.items())
            or any(call.get("facts", {}).get(k) != v for k, v in impl.facts.items())
            or set(call.get("requires", call.get("facts", {}).get("requires", ()))) != {"async_copy", "cooperative_grid"}):
        raise CodegenError("Async interleaved parameters/facts/capabilities differ from its versioned identity")
    ordinary = interleaved_projection_call(call)
    if any(call.get(k) != v for k, v in ordinary.items()):
        raise CodegenError("Async interleaved row/field/pointer expressions differ from actual ordinary ABI")
    outputs = [_binding(call, "outputs", "result_" + str(i)) for i in (0, 1)]
    inputs = [_binding(call, "inputs", name) for name in ("value", "weight", "weight_scale")]
    for output in outputs:
        for operand in inputs:
            _reject_alias(output, operand)
        abi = output["abi"]
        if abi.get("coordinate_space") != "local" or abi.get("memory_sharing_scope") != "block":
            raise CodegenError("Async interleaved requires distinct CTA-private output replicas")
    _reject_alias(*outputs)
    names = ("source_row_offset", "weight_row_offset", "n_mask", "field", "destination_0_offset", "destination_1_offset")
    expr = {name: _expression(call[name].replace("interleaved_offsets", "local_n_offsets").replace("interleaved_m", "0")) for name in names}
    row_visits = Counter()
    stores = set()
    for y in range(4):
        for x in range(8):
            env = dict(shard_y=y, shard_x=x, shard_coord0=y, shard_coord1=x, shard_index=y*8+x)
            if eval(expr["source_row_offset"], {"__builtins__": {}}, env) != 0:
                raise CodegenError("Async interleaved source pointer must select the complete row")
            local_stores = [set(), set()]
            for row in range(ROWS):
                env["local_n_offsets"] = row
                if not eval(expr["n_mask"], {"__builtins__": {}}, env):
                    raise CodegenError("Async interleaved does not admit partial/tail field rows")
                byte = eval(expr["weight_row_offset"], {"__builtins__": {}}, env)
                if byte % 2720 or not 0 <= byte < 12288 * 2720:
                    raise CodegenError("Async interleaved native row is unaligned/out of range")
                native_row = byte // 2720
                field = eval(expr["field"], {"__builtins__": {}}, env)
                head, dim = native_row // 512, native_row % 256
                offset = eval(expr["destination_" + str(field) + "_offset"], {"__builtins__": {}}, env) if field in (0, 1) else -1
                if (field != (native_row // 256) % 2 or head // 3 != x
                        or offset != (head % 3) * 256 + dim or offset in local_stores[field]):
                    raise CodegenError("Async interleaved changed its head/field/output owner map")
                local_stores[field].add(offset)
                row_visits[native_row] += 1
                stores.add((y, x, field, offset))
            if any(s != set(range(768)) for s in local_stores):
                raise CodegenError("Async interleaved output coverage has a hole")
    if len(row_visits) != 12288 or set(row_visits.values()) != {4} or len(stores) != 49152:
        raise CodegenError("Async interleaved must preserve four replicas of all native rows")
    return {**_workspace(call), "local_iq4_async": True, "local_iq4_async_warps": 4,
            "local_iq4_async_barriers": BARRIERS, "local_iq4_async_owner_rows": 49152}


def configure_descriptor(descriptor):
    calls = [c for c in descriptor["render_calls"] if c.get("family") == "interleaved_projection"]
    if not calls:
        raise CodegenError("Explicit async_interleaved selected no interleaved call")
    for call in calls:
        call.update(validate_call(call, descriptor))
        descriptor["local_iq4_recipe"]["async_sites"].append({"symbol": call["symbol"], "implementation": IMPLEMENTATION,
            "shared_workspaces": call["shared_workspaces"], **{k: v for k, v in call.items() if k.startswith("local_iq4_async")}})
        if call["local_iq4_async_arena_end"] > descriptor["local_iq4_shared_bytes"]:
            raise CodegenError("Async interleaved exceeds the already planned GLU arena")
