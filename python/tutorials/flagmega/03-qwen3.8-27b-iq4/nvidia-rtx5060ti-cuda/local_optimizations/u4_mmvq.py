# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Frozen U4-L providers and local template registration, CPP SM120 only.

No launch tuning, live registration at import, or GPU/native module loading.
I0 explicitly calls enable() before normal candidate selection/code generation.
"""
from pathlib import Path
from math import prod
from triton.flagmega.codegen.triton.implementation import TritonImplementation, TritonImplementationModel
from triton.flagmega.codegen.triton.candidates import TritonCandidateProposal, TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.ir import T, tensor_type, TupleType
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.execution_mesh import effective_execution_mesh, ExecutionMeshKind
from .u4_q8_abi import NUMERICAL_ID, Q8Layout, primitive_fields, validate_primitive_fields, exact, require
from .u4_views import TensorLayout

FAMILIES = {"encode": "u4_q8_encode", "matmul": "u4_iq4_matmul", "glu": "u4_iq4_glu",
            "interleaved": "u4_iq4_interleaved", "gather": "u4_gdn_z_gather"}
OPS = {"u4.q8_encode": "encode", "u4.iq4_matmul": "matmul", "u4.iq4_glu": "glu",
       "u4.iq4_interleaved": "interleaved", "u4.gdn_z_gather": "gather"}


def implementation(kind, k=5120, components=1, *, fullhead48=False):
    require(kind in FAMILIES, "unknown U4 provider kind")
    require(type(fullhead48) is bool and (not fullhead48 or kind == "gather"), "fullhead48 is only an explicit gather variant")
    if kind != "gather":
        Q8Layout(k, components)
        require(kind != "matmul" or components == 1, "full12 simple matrices require C1")
        require(kind not in ("glu", "interleaved") or (k, components) == (5120, 2), "dual projection geometry differs")
    fields = 2 if kind == "glu" else 1
    partial = kind in ("matmul", "interleaved")
    parameters = {"compute_num_warps": 4, "num_warps": 4, "u4_kind": kind}
    if fullhead48:
        parameters.update(compute_num_warps=8, num_warps=8, u4_private_values_per_cta=128)
    if kind == "glu":
        parameters.update(u4_schedule="warp_private_row4")
    if kind != "gather":
        parameters.update(u4_k=k, u4_components=components, numerical_id=NUMERICAL_ID)
    return TritonImplementation(
        "tir.local_u4.glu.k5120.c2.warp_private_row4.v1" if kind == "glu" else
        f"tir.local_u4.{kind}.k{k}.c{components}.v1" if kind != "gather" else ("tir.local_u4.gdn_z_gather.fullhead48.v1" if fullhead48 else "tir.local_u4.gdn_z_gather.fixed96.v1"),
        FAMILIES[kind], "simt", parameters,
        {"required_codegen_backend": "cpp", "numerical_id": NUMERICAL_ID},
        (), {"required_codegen_backend": "cpp", "internal_grid_barriers": 0,
             "internal_cta_barriers": 1 if partial or kind == "gather" else 0,
             "source_memory_scope": "chip", "result_memory_scope": "block" if kind == "gather" else "chip"},
        (T.shared_workspace_descriptor("row_partials", tensor_type("float32", (3, fields, components, 32)), 16),)
        if partial else ())


def catalog():
    return tuple(implementation("encode", k, c) for k in (5120, 6144, 17408) for c in (1, 2)) + tuple(
        implementation("matmul", k, 1) for k in (5120, 6144, 17408)) + (
        implementation("glu", 5120, 2), implementation("interleaved", 5120, 2), implementation("gather"), implementation("gather", fullhead48=True))


def _layout(t):
    t = tensor_of(t)
    return TensorLayout(getattr(t.dtype, "elem_type", t.dtype).value,
                        tuple(d.fixed_value for d in t.shape), tuple(getattr(t.dtype, "lanes", ())))


class U4Provider:
    op_names = frozenset(OPS)

    def propose(self, node, context):
        require(context.codegen_backend == "cpp" and context.target.codegen_platform == "nvidia"
                and context.target.codegen_architecture == "sm120", "U4 requires CPP NVIDIA SM120")
        kind = OPS[node.op]
        mesh = effective_execution_mesh(context.module, node)
        if kind == "gather" and node.metadata.get("u4_fuse_into_fullhead48") is True:
            require(mesh is not None and mesh.kind is ExecutionMeshKind.FIXED
                    and tuple(mesh.shape) == (4, 12) and mesh.mapping == "row_major"
                    and node.metadata.get("kernel_num_warps") == 8
                    and node.metadata.get("u4_private_values_per_cta") == 128
                    and not node.metadata.get("u4_fuse_into_fixed96"),
                    "fullhead48 Z gather requires explicit fixed4x12/8warp/private128 continuation")
            candidate = context.configure_implementation(implementation("gather", fullhead48=True), semantic_parameters={
                "u4_source_epoch": node.attrs["source_epoch"], "u4_fuse_into_fullhead48": True})
        elif kind == "gather":
            require(mesh is not None and mesh.kind is ExecutionMeshKind.FIXED
                    and tuple(mesh.shape) == (4, 24), "Z gather requires actual fixed4x24 execution")
            require(node.metadata.get("u4_fuse_into_fixed96") is True,
                    "Z gather must be explicitly fused into the following fixed96 continuation")
            candidate = context.configure_implementation(implementation("gather"), semantic_parameters={
                "u4_source_epoch": node.attrs["source_epoch"], "u4_fuse_into_fixed96": True})
        else:
            from .u4_execution_plan import task_policy_for_node
            require(mesh is not None and mesh.kind is ExecutionMeshKind.DYNAMIC,
                    "U4 ordinary provider requires regenerated dynamic execution mesh")
            fields = node.metadata.get("u4_primitive")
            layout = validate_primitive_fields(fields)
            types = node.type.fields if isinstance(node.type, TupleType) else (node.type,)
            expected = primitive_fields(kind=kind, k=node.attrs["k"], components=node.attrs["components"],
                source_ssa=node.metadata.get("u4_source_ssa"), source_epoch=node.attrs["source_epoch"],
                output_layouts=() if kind == "encode" else tuple(_layout(t) for t in types))
            require(exact(fields, expected) and exact(node.attrs["q8_schema"], fields["q8"]),
                    "actual selected node differs from Graph/Vector primitive ABI")
            if kind == "encode":
                require(_layout(types[0]) == TensorLayout("uint8", layout.shape), "wrong actual Q8 result")
            policy = task_policy_for_node(node)
            candidate = context.configure_implementation(implementation(kind, layout.k, layout.components),
                semantic_parameters={"u4_primitive": fields, "u4_task_policy": policy.to_data(),
                                     "task_mapping": policy.recipe.to_data()})
        return TritonCandidateProposal((candidate,), candidate.id)


def register_templates():
    """Scoped local registry adapters; source remains owned by this package."""
    from jinja2 import ChoiceLoader, FileSystemLoader
    from triton.flagmega.codegen.cpp import renderer as cpp_renderer
    from triton.flagmega.codegen.triton import tir_package
    cpp_base, triton_base = cpp_renderer.CppTemplateRegistry, tir_package.TritonTemplateRegistry
    if getattr(cpp_base, "_u4_registered", False):
        return
    root = Path(__file__).parent
    files = {FAMILIES["encode"]: "u4_q8_encode", FAMILIES["gather"]: "u4_gdn_z_gather",
             **{FAMILIES[k]: "u4_mmvq" for k in ("matmul", "glu", "interleaved")}}
    class CppU4Templates(cpp_base):
        _u4_registered = True
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.environment.loader = ChoiceLoader([FileSystemLoader(str(root / "templates")), self.environment.loader])
        def available(self, spec):
            return spec.kernel in files or super().available(spec)
        def resolve(self, spec):
            if spec.kernel in files:
                require((spec.variant, spec.platform, spec.architecture) == ("simt", "nvidia", "sm120"),
                        "U4 template requires frozen CPP SM120 identity")
                return "kernels/" + files[spec.kernel] + ".cpp.jinja"
            return super().resolve(spec)
    class U4MetadataTemplates(triton_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.environment.loader = ChoiceLoader([FileSystemLoader(str(root / "tir_metadata")), self.environment.loader])
        def resolve(self, spec):
            if spec.kernel in files:
                require((spec.variant, spec.platform, spec.architecture) == ("simt", "nvidia", "sm120"),
                        "U4 metadata requires frozen CPP SM120 identity")
                return "kernels/u4_cpp_only.py.jinja"
            return super().resolve(spec)
    cpp_renderer.CppTemplateRegistry = CppU4Templates
    # CPP multi-program rendering uses a separate module with the same registry
    # lookup; C0 owns any generic alias, I0 passes this registry explicitly there.
    tir_package.TritonTemplateRegistry = U4MetadataTemplates


def enable(target=None):
    from triton.flagmega.targets import get_target
    from .u4_encoders import register_encoders
    target = target or get_target("nvidia-sm120")
    model, policy = target.triton_implementation_model, target.tir_selection_policy
    if model.implementation(catalog()[0].id) is None:
        registry = TritonCandidateProviderRegistry()
        for provider in policy.registry.providers:
            registry.add(provider)
        registry.add(U4Provider())
        target.triton_implementation_model = TritonImplementationModel(
            (*model.implementations, *catalog()), model.preferences, model.name + "+u4l-v1")
        target.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)
    register_encoders()
    register_templates()
