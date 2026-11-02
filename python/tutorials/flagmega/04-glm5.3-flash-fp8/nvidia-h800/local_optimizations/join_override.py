"""Explicit full-function template replacement for this frozen typed TP4 join.

The six semantic calls remain the numerical and effect contract. The replacement
owns their complete invocation, all output writes and one planned persistent
communication channel. The ordinary caller retains its device fences and ABI.
"""
from pathlib import Path
import hashlib
import json

from triton.flagmega.errors import CodegenError

OUT = Path(__file__).resolve().parent
FUNCTIONS = {"decode_mla_attention_sparse.join", "decode_linear_attention_sparse.join"}
FAMILIES = ["broadcast_to", "sparse_experts_weighted_sum", "elementwise",
            "reduce_sum", "distributed_boxing", "mhc_post"]
DTYPES = ["float32", "float32", "float32", "float32", "int32",
          "float32", "bfloat16", "float32", "uint8"]


def describe(module, previous):
    planned_hash = json.loads((OUT / 'plan.json').read_text())['semantic_hash']
    if module.semantic_hash != planned_hash:
        raise CodegenError("Persistent join override requires its frozen verified typed module")
    descriptor = previous(module)
    plan = descriptor["persistent_window_resources"]
    resource, = [r for r in plan["resources"] if r["name"] == "ordered_join"]
    if resource["name"] != "ordered_join" or resource["nbytes"] != 8448:
        raise CodegenError("Join channel must contain one BF16 payload and two 16-word epoch arrays")
    actors = [s["symbol"] for s in descriptor["kernels"] if s["entry_function"] in FUNCTIONS]
    if actors != resource["programs"] or len(actors) != 42:
        raise CodegenError("Join invocations disagree with the verified persistent channel DAG")
    receipt = dict(payload=resource["offset"], ready=resource["offset"] + 8192,
                   consumed=resource["offset"] + 8320)
    seen = set()
    definitions = list(descriptor["device_functions"])
    for program in descriptor["programs"]:
        definitions.extend(program.get("device_functions", ()))
    for definition in definitions:
        if definition["function"] not in FUNCTIONS:
            continue
        abi = definition["function_pool_abi"]
        bindings = abi["bindings"]
        schedule = definition["schedule"]
        events = schedule["consumer_events"]
        if (abi["schema"] not in ("flagmega.function-pool-abi/v1", "flagmega.function-pool-abi/v2")
                or abi["record_width"] != 9 or [b["record_column"] for b in bindings] != list(range(9))
                or [b["scalar_dtype"] for b in bindings] != DTYPES
                or any(b["storage"] != "workspace" or b["byte_offset"] != 0
                       or b["alignment_bytes"] != 256 for b in bindings)
                or [e["family"] for e in events if e["kind"] == "tir.kernel_call"] != FAMILIES
                or schedule["producer_events"] or schedule["auxiliary_events"]):
            raise CodegenError("Join typed numerical/layout/call ABI changed; override needs review")
        definition["owner_matched_join"] = receipt
        seen.add(definition["function"])
    if seen != FUNCTIONS:
        raise CodegenError("Both full-model join implementations must be selected")
    assert all(s["num_warps"] == 4 for s in descriptor["kernels"] if s["entry_function"] in FUNCTIONS)
    descriptor["join_warp_schedule"] = dict(warps=4, full42_invocations=True,
        channel_ABI_owner_epoch_release_math_and_DAG_unchanged=True)
    descriptor["implementation_override"] = dict(
        kind="owner-matched-persistent-join/v1", typed_module=planned_hash,
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [OUT / "join_override.py", OUT / "join-body.py.jinja", OUT / "fences.py"]},
    )
    return descriptor


def activate():
    from triton.flagmega.codegen.triton import renderer, tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    original = tir_package.describe_tir_package
    if getattr(original, "_persistent_join_trial", False):
        return
    def wrapped(module):
        return describe(module, original)
    wrapped._persistent_join_trial = True
    tir_package.describe_tir_package = wrapped
    spec = renderer.renderer_registry._specs["bufferized-tir"]
    renderer.renderer_registry._specs[spec.name] = PackageRendererSpec(
        spec.name, spec.kind, spec.matches, wrapped, spec.render)
