# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A56 fixed NormApply/C1 provider and checked ordinary CPP descriptor.

The compiler owns allocation, task launch and tuple lowering. This module
rejects any raw ABI which does not realize the four canonical operands and
two disjoint full results; it never rewrites buffers or emitted source.
"""
from collections.abc import Mapping
from pathlib import Path

from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal, TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.codegen.triton.physical_access import emit_storage_pointer
from triton.flagmega.ir import TupleType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.execution_mesh import effective_execution_mesh, ExecutionMeshKind
from .u4_norm_q8_c1_c2 import CONTRACT_ID, HIDDEN_TYPE, STATS_TYPE, AFFINE_TYPE, Q8_C1_TYPE, Q8_C2_TYPE, NormApplyQ8C1C2
from .u4_q8_abi import require, exact
from .u4_views import TensorLayout
from .u4_encoders import _binding, _typed, _no_alias

FAMILY = "u4_norm_apply_q8_c1_c2"
IMPLEMENTATION = "tir.local_u4.norm_apply_q8_c1_c2.k5120.c2.warp4.v1"
SCHEMA = "flagmega.u4-norm-q8-c1c2-cpp-fields/v1"
INPUTS = ("hidden", "stats", "scale", "bias")
OUTPUTS = ("result_0", "result_1", "result_2")
LAYOUTS = (TensorLayout("bfloat16", (1, 640), (8,)),
           TensorLayout("float32", (1, 1, 1)),
           TensorLayout("float32", (640,), (8,)),
           TensorLayout("float32", (640,), (8,)),
           TensorLayout("bfloat16", (1, 640), (8,)),
           TensorLayout("uint8", (1, 160, 36)),
           TensorLayout("uint8", (2, 160, 36)))


def implementation():
    from .u4_common_q8_kernels import source_identity
    return TritonImplementation(IMPLEMENTATION, FAMILY, "simt",
        {"compute_num_warps": 4, "num_warps": 4, "u4_norm_q8_contract": CONTRACT_ID, "common_q8_source_identity": source_identity()},
        {"required_codegen_backend": "cpp"}, (),
        {"required_codegen_backend": "cpp", "internal_grid_barriers": 0,
         "internal_cta_barriers": 0, "source_memory_scope": "chip", "result_memory_scope": "chip"}, ())


def policy():
    from .u4_execution_plan import TaskPolicy
    return TaskPolicy("ordinary_norm_q8_c1c2", 5120, 2, fields=3)


class NormQ8Provider:
    op_names = frozenset((NormApplyQ8C1C2.op_name,))

    def propose(self, node, context):
        require(context.codegen_backend == "cpp" and context.target.codegen_platform == "nvidia"
                and context.target.codegen_architecture == "sm120", "A56 requires CPP NVIDIA SM120")
        mesh = effective_execution_mesh(context.module, node)
        require(mesh is not None and mesh.kind is ExecutionMeshKind.DYNAMIC,
                "A56 requires regenerated ordinary dynamic execution")
        from .u4_common_q8_distribution import validate_fused_node
        from .u4_norm_q8_c1_c2_vector import validate_norm_types, field_q8_contract
        from .u4_execution_plan import task_policy_for_node
        validate_fused_node(node, context.module)
        require(isinstance(node.type, TupleType) and tuple(tensor_of(t) for t in node.type.fields)
                == (HIDDEN_TYPE, Q8_C1_TYPE, Q8_C2_TYPE), "A56 tuple ABI differs")
        require(tuple(tensor_of(context.module.node_map[n].type) for n in node.inputs)
                == (HIDDEN_TYPE, STATS_TYPE, AFFINE_TYPE, AFFINE_TYPE), "A56 operand types differ")
        attrs = NormApplyQ8C1C2.normalize_attrs(node.attrs)
        p = task_policy_for_node(node)
        require(p == policy(), "A56 task ownership differs")
        candidate = context.configure_implementation(implementation(), semantic_parameters={
            "u4_norm_q8_attrs": attrs, "u4_norm_q8_field_contracts": {str(i):field_q8_contract(node,i) for i in (1,2)}, "u4_task_policy": p.to_data(), "task_mapping": p.recipe.to_data()})
        return TritonCandidateProposal((candidate,), candidate.id)


def encode_call(raw):
    require(isinstance(raw, Mapping) and raw.get("semantic_op") == NormApplyQ8C1C2.op_name
            and raw.get("family") == FAMILY and raw.get("variant") == "simt",
            "A56 raw semantic/family/variant differs")
    impl = implementation()
    params = raw.get("parameters", {})
    require(raw.get("implementation") == impl.id
            and all(exact(params.get(k), v) for k, v in impl.parameters.items())
            and all(exact(raw.get("facts", {}).get(k), v) for k, v in impl.facts.items()),
            "A56 selected implementation facts differ")
    attrs = dict(raw.get("semantic_attrs", {}))
    normalized_attrs = NormApplyQ8C1C2.normalize_attrs(attrs)
    require(exact(normalized_attrs, attrs) and exact(params.get("u4_norm_q8_attrs"), attrs),
            "A56 original arithmetic/Q8/epoch contract differs")
    p = policy()
    require(exact(params.get("u4_task_policy"), p.to_data())
            and exact(params.get("task_mapping"), p.recipe.to_data()), "A56 task mapping differs")
    require(len(raw.get("inputs", ())) == 4 and len(raw.get("outputs", ())) == 3,
            "A56 requires four inputs and three independent result fields")
    bindings = tuple(_binding(raw, "inputs", n) for n in INPUTS) + tuple(
        _binding(raw, "outputs", n) for n in OUTPUTS)
    from .u4_norm_q8_c1_c2_vector import validate_norm_raw_representation
    from .u4_ops import _common, encoding_attrs
    validate_norm_raw_representation(raw)
    require(exact(params.get("u4_norm_q8_field_contracts"), {
        str(i): _common(encoding_attrs(k=5120, components=i, source_epoch=attrs["source_epoch"])) for i in (1,2)}),
        "A56 requires separately bound C1 and C2 field contracts")
    require(not raw.get("workspaces") and not raw.get("shared_workspaces")
            and raw.get("transfer_pipeline") is None, "A56 has zero global/shared scratch and no transfer pipeline")
    return dict(schema=SCHEMA, u4_norm_q8_contract=CONTRACT_ID,
        u4_norm_q8_attrs=attrs, epsilon=attrs["epsilon"],
        cpp_task_mapping="independent-cta", task_launch_recipe=p.recipe.to_data(),
        cpp_trace_sites=[dict(id=2, name="norm_begin"), dict(id=12, name="bf16_round_complete"),
                         dict(id=3, name="q8_stores_issued")],
        u4_pointers={name: emit_storage_pointer(binding["abi"], binding["runtime_argument"])
            for name, binding in zip((*INPUTS, "normalized", "q8_c1", "q8_c2"), bindings)}, shared_workspaces=[])


def validate_raw(raw):
    encode_call(raw)


def validate_context(context):
    calls = [c for c in context["render_calls"] if c["family"] == FAMILY]
    if calls:
        require(context["platform"] == "nvidia" and context["target_architecture"] == "sm120"
                and context["block_threads"] == 128 and context["warp_size"] == 32
                and tuple(context["grid"]) == (40, 1, 1), "A56 physical launch differs")
        require(len(calls) == len(context["render_calls"]) == 1,
                "A56 ordinary entry must own only the fused producer")
        require(calls[0].get("u4_norm_q8_contract") == CONTRACT_ID,
                "A56 checked raw encoder was not selected")
    return [dict(call=c["call"], contract=CONTRACT_ID, grid=[40, 1, 1], threads=128,
                 warps=160, scratch_bytes=0) for c in calls]


def register_templates():
    from jinja2 import ChoiceLoader, FileSystemLoader
    from triton.flagmega.codegen.cpp import renderer
    from triton.flagmega.codegen.triton import tir_package
    cpp_base, tir_base = renderer.CppTemplateRegistry, tir_package.TritonTemplateRegistry
    if getattr(cpp_base, "_norm_q8_c1c2_registered", False):
        return
    root = Path(__file__).parent
    def validate(spec):
        require((spec.variant, spec.platform, spec.architecture) == ("simt", "nvidia", "sm120"),
                "A56 template requires CPP SM120")
    class CppNormQ8Templates(cpp_base):
        _norm_q8_c1c2_registered = True
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.environment.loader = ChoiceLoader([FileSystemLoader(str(root / "templates")), self.environment.loader])
        def available(self, spec):
            return spec.kernel == FAMILY or super().available(spec)
        def resolve(self, spec):
            if spec.kernel == FAMILY:
                validate(spec)
                return "kernels/u4_norm_apply_q8_c1_c2.cpp.jinja"
            return super().resolve(spec)
    class NormQ8MetadataTemplates(tir_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.environment.loader = ChoiceLoader([FileSystemLoader(str(root / "tir_metadata")), self.environment.loader])
        def resolve(self, spec):
            if spec.kernel == FAMILY:
                validate(spec)
                return "kernels/u4_norm_apply_q8_c1_c2.py.jinja"
            return super().resolve(spec)
    renderer.CppTemplateRegistry = CppNormQ8Templates
    tir_package.TritonTemplateRegistry = NormQ8MetadataTemplates


def enable(target=None):
    from triton.flagmega.targets import get_target
    from triton.flagmega.codegen.triton.kernel_call_renderers import register_family_encoder
    target = target or get_target("nvidia-sm120")
    model, selection = target.triton_implementation_model, target.tir_selection_policy
    if model.implementation(IMPLEMENTATION) is None:
        registry = TritonCandidateProviderRegistry()
        for provider in selection.registry.providers:
            registry.add(provider)
        registry.add(NormQ8Provider())
        target.triton_implementation_model = TritonImplementationModel(
            (*model.implementations, implementation()), model.preferences, model.name + "+norm-q8-v1")
        target.tir_selection_policy = TritonTirSelectionPolicy(selection.workspace_annotator, registry)
    register_family_encoder(FAMILY, encode_call, schema=SCHEMA, validate_raw=validate_raw)
    register_templates()
