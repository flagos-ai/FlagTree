"""Fixed t26 terminal NormStats election; normal typed describe/registry only.

The contract JSON is a projection of the real t24 normal descriptor, not a
second planner. Exact equality deliberately rejects a changed physical plan.
No annotation is committed until both typed sites and all 64 callers pass.
"""
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from triton.flagmega.errors import CodegenError

SITES = ((13, "decode_linear_0", "917", 48),
         (28, "decode_attention_1", "936", 16))
FLAG = "local_output_norm_owner"
CONTEXT_KEYS = (
    "entry_id", "symbol", "owning_function", "platform", "target_architecture",
    "grid", "block_threads", "warp_size", "warps", "launch_kind",
    "grid_mesh", "mesh_axes", "mesh_axis_sources", "mesh_coordinate_strides",
    "mesh_size", "mesh_hierarchy", "mesh_levels", "grid_barrier_axis_groups",
    "grid_barrier_counter_slots", "launch_scratch", "scoped_entry_arguments",
    "signature_arguments", "typed_signature_arguments", "argument_types",
    "entry_events", "render_calls", "kernel_templates", "kernel_template_specs",
    "shared_arena_reuse", "pipeline_schedule", "device_functions")


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise CodegenError("output_norm_owner: " + message)


@lru_cache(maxsize=1)
def contract():
    return json.loads(Path(__file__).with_name("output_norm_owner_contract.json").read_text())


def validate_recipe(config):
    expected = contract()["parent_recipe"] | {"output_norm_owner": True}
    require(config == expected, "requires the exact inherited t24 common-Q8 recipe")


def context_projection(context):
    # Strip only our own successful annotation, so repeated describes are safe.
    result = {k: plain(context.get(k)) for k in CONTEXT_KEYS}
    for call in result["render_calls"]:
        call.pop(FLAG, None)
    return result


def validate_and_annotate(module, result):
    expected = contract()
    require(module.stage == "bufferized_tir", "requires final typed module")
    require(result.get("platform") == "nvidia" and result.get("target_architecture") == "sm120",
            "requires cpp NVIDIA SM120")
    contexts = result.get("native_program_contexts", ())
    require(len(contexts) == 33, "requires full 33-entry context")
    programs = result.get("cpp_programs", {})
    require(len(programs.get("entries", ())) == 33 and len(programs.get("invocations", ())) == 869,
            "full program/invocation census differs")
    # Includes every ABI, frame, source/stat binding, launch and scope stride.
    require(digest(programs) == expected["cpp_programs_sha256"], "full 33-entry ABI or 869 callers differ")
    partitions = module.metadata.get("kernel_program_partition", ())
    require(len({p["symbol"] for p in partitions}) == len(partitions), "duplicate partition symbol")
    partition = {p["symbol"]: p for p in partitions}
    by_entry = {c["entry_id"]: c for c in contexts}
    require(len(by_entry) == 33, "duplicate native entry")
    names = {owner + ".output_norm_stats" for _, owner, _, _ in SITES}
    typed_sites = [n for n in module.nodes if n.metadata.get("lowered_from") == "nn.norm_stats"
                   and n.metadata.get("role") == "output"]
    require(len(typed_sites) == 2 and {n.id for n in typed_sites} == names,
            "typed terminal NormStats node set differs")
    rendered = [c for ctx in contexts for c in ctx["render_calls"] if c.get("call") in names]
    require(len(rendered) == 2 and {c["call"] for c in rendered} == names,
            "exact rendered call pair differs")
    require(not any(c.get(FLAG) and c.get("call") not in names
                    for ctx in contexts for c in ctx["render_calls"]), "extra owner annotation")
    pending = []
    for entry, owner, suffix, count in SITES:
        name = owner + ".output_norm_stats"
        producer = owner + "_output.vectorized.compute"
        source = producer + ".reshard" + suffix
        ctx = by_entry.get(entry)
        require(ctx is not None and ctx.get("owning_function") == owner, "native owning function differs")
        p = partition.get(ctx["symbol"])
        require(p is not None and p.get("function") == owner
                and tuple(p.get("calls", ())) == (producer, name), "typed partition closure differs")
        calls = ctx["render_calls"]
        require([c["call"] for c in calls] == [producer, name], "terminal call order differs")
        residual, call = calls
        node, view, output = (module.node_map.get(n) for n in (name, source, producer))
        require(node is not None and view is not None and output is not None, "missing actual typed node")
        require(node.op == "tir.call" and node.inputs == (source,)
                and node.attrs.get("callee") == call["callee"] and node.effect.kind.value == "pure",
                "typed NormStats operand/callee/effect differs")
        require(view.op == "tir.buffer_view" and view.inputs == (producer,)
                and view.attrs.get("alias_kind") == "sharded_view", "input is not residual broadcast view")
        require(output.op == "tir.call" and output.attrs.get("callee") == residual["callee"],
                "actual residual producer differs")
        require(node.type.to_data() == expected["types"]["result"]
                and view.type.to_data() == expected["types"]["source"]
                and output.type.to_data() == expected["types"]["producer"],
                "actual TensorType/VectorType/layout/ownership differs")
        fn = module.function_map.get(owner)
        require(fn is not None and fn.outputs[0] == source and fn.outputs[-1] == name,
                "typed terminal function outputs differ")
        # Bind actual kernel definition, including semantic attrs and retained tree/default settings.
        kernel = module.kernel_callable_map.get(node.attrs["callee"])
        require(kernel is not None and digest(kernel.to_data()) == expected["kernel_definition_sha256"],
                "actual NormStats kernel definition differs")
        require(call.get("family") == "norm_stats" and call.get("variant") == "local"
                and call.get("semantic_op") == "nn.norm_stats" and call.get("semantic_attrs") == {"axis": 1, "use_mean": False}
                and call.get("writer_active") == "(shard_y == 0) & (shard_x == 0)"
                and call.get("outer_capacity") == 1 and call.get("reduction_capacity") == 5120
                and call.get("lane_count") == 8 and call.get("partial_stats") is None
                and call.get("internal_grid_barriers") == 0, "reduction or CTA-only writer differs")
        events = ctx["entry_events"]
        require(len(events) == 3 and events[0].get("call") == producer
                and events[1].get("kind") == "barrier" and events[1].get("scope") == "grid"
                and tuple(events[1].get("after", ())) == (producer,)
                and events[1].get("before") == name and events[2].get("call") == name
                and events[0].get("participation_active") == events[2].get("participation_active") == "True",
                "full residual publication or terminal closure differs")
        require(context_projection(ctx) == expected["contexts"][str(entry)],
                "exact physical render context differs (spans/expressions/barriers/resources)")
        callers = [n for n in module.nodes if n.op == "tir.call" and n.attrs.get("callee") == owner]
        invocations = [i for i in programs["invocations"] if i["entry_id"] == entry]
        require(len(callers) == len(invocations) == count
                and {n.id for n in callers} == {i["call_path"][0] for i in invocations},
                "actual typed dynamic caller census differs")
        pending.append(call)
    invocations = [i for i in programs["invocations"] if i["entry_id"] in (13, 28)]
    require({tuple(i["call_path"]) for i in invocations}
            == {(f"layer_{i}_call",) for i in range(64)}, "all 64 unique layer paths required")
    for call in pending:
        call[FLAG] = True
    return {"entries": [13, 28], "definitions": 2, "invocations": 64,
            "cpp_programs_sha256": expected["cpp_programs_sha256"]}
