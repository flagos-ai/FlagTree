# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""U4-L ownership and audits of regenerated serial program plans.

Local policies feed K0/I0. They do not allocate buffers, select native launches
from distributed_entry, or stand in for the normal compiler.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, replace
from math import prod
from triton.flagmega.ir.execution_mesh import ExecutionMesh, KernelProgramKind
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
from triton.flagmega.ir.program_execution import ProgramExecution, ProgramInvocation, CallInvocation
from .u4_q8_abi import Q8Layout, NUMERICAL_ID, exact, integer, require
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidateProviderBase, DistributedCandidate,
)

SUPPORTED_ROWS = (1024, 5120, 6144, 10240, 12288, 17408, 248320)
EXPECTED_COUNTS = {
    "full12_norm_common_q8_fullhead48": dict(definitions=33, fixed96=227, fixed48=48, ordinary_q8=193,
                          ordinary_norm_q8_c1c2=64, ordinary_matrix=337,
                          compute=869, resets=275, total_nodes=1144),
    "motif": dict(definitions=14, fixed96=130, ordinary_q8=128, ordinary_matrix=128,
                  compute=386, resets=130, total_nodes=516),
    "full12": dict(definitions=36, fixed96=339, ordinary_q8=273, ordinary_matrix=337,
                   compute=949, resets=339, total_nodes=1288),
    "full12_norm_q8_fullhead48": dict(definitions=34, fixed96=227, fixed48=48, ordinary_q8=209,
                          ordinary_norm_q8=64, ordinary_matrix=337,
                          compute=885, resets=275, total_nodes=1160),
    "full12_norm_q8": dict(definitions=34, fixed96=275, ordinary_q8=209,
                          ordinary_norm_q8=64, ordinary_matrix=337,
                          compute=885, resets=275, total_nodes=1160),
}


class U4DistributionProvider(DistributedCandidateProviderBase):
    """The frozen local type relations in the ordinary target-owned search."""
    op_names = frozenset({"u4.q8_encode", "u4.iq4_matmul", "u4.iq4_glu",
                          "u4.iq4_interleaved", "u4.gdn_z_gather", "u4.norm_apply_q8_c1", "u4.norm_apply_q8_c1_c2"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        from triton.flagmega.ir.distributed_inference import broadcast_ir_type
        from triton.flagmega.ir.execution_mesh import effective_execution_mesh, placement_for_execution_mesh
        from triton.flagmega.ir.ops.core import tensor_nbytes
        from .u4_transitions import gdn_z_reader_type
        node, placement = context.source_call, context.placement
        mesh = effective_execution_mesh(context.module, node)
        if node.op == "u4.gdn_z_gather":
            fullhead48 = node.metadata.get("u4_fuse_into_fullhead48") is True
            shape = (4, 12) if fullhead48 else (4, 24)
            flag = "u4_fuse_into_fullhead48" if fullhead48 else "u4_fuse_into_fixed96"
            require(mesh is not None and mesh.kind.value == "fixed" and mesh.shape == shape
                    and node.metadata.get(flag) is True
                    and node.metadata.get("kernel_num_warps") == (8 if fullhead48 else 4)
                    and not (fullhead48 and node.metadata.get("u4_fuse_into_fixed96")),
                    "GDN Z requires its explicit fused owner mesh/warps")
            output = gdn_z_reader_type(node.type, fullhead48=fullhead48)
            require(output.placement == placement, "GDN Z provider needs its exact local placement")
            source_placement = placement_for_execution_mesh(placement, ExecutionMesh.dynamic())
            inputs = (broadcast_ir_type(context.module.node_map[node.inputs[0]].type, source_placement),)
            cost, reason = 24576, ("u4-fused-canonical-to-private128-gather" if fullhead48
                                   else "u4-fused-canonical-to-private64-gather")
        else:
            require(mesh is not None and mesh.kind.value == "dynamic"
                    and all(n == 1 for n, level in zip(placement.hierarchy, placement.hierarchy_levels)
                            if level == "b"), "ordinary U4 provider requires dynamic unit block placement")
            task_policy_for_node(node)
            output = broadcast_ir_type(node.type, placement)
            inputs = tuple(broadcast_ir_type(context.module.node_map[n].type, placement) for n in node.inputs)
            from triton.flagmega.ir import TupleType
            from triton.flagmega.ir.distributed_inference import tensor_of
            fields = node.type.fields if isinstance(node.type, TupleType) else (node.type,)
            cost = min(sum(tensor_nbytes(tensor_of(t)) for t in fields), 2_000_000_000)
            reason = "u4-complete-chip-value-independent-cta-tasks"
        return (DistributedCandidate("distribution." + node.id + ".u4", output, inputs, cost, reason,
                                     objective_kind="analytic", objective_model="flagmega.u4-logical-bytes/v1",
                                     objective_evidence=("unmeasured-logical-bytes", "frozen-u4-owners")),)


class _FullHead48ReadonlyViewPolicy:
    """Preserve complete chip-global readonly bytes across A55's local meshes.

    The normal boundary planner asks for this realization before constant
    materialization. Core ShardedView type and physical-backing checks still
    verify the result; no materialized-source proof rule is changed.
    """
    def __init__(self, parent):
        self.parent = parent

    def uses_sharded_views_for_constants(self):
        return self.parent.uses_sharded_views_for_constants()

    def classify(self, context):
        from triton.flagmega import ir as fm
        from triton.flagmega.ir.distributed_type import SBPBroadCast, sharded_view_error
        from triton.flagmega.passes.auto_distributed.realization import (
            DistributedReshardSourceKind, DistributedReshardRealization)
        source, target = context.source_type, context.target_type
        if (context.source_kind is DistributedReshardSourceKind.CONSTANT
                and isinstance(source, fm.DistributedType) and isinstance(target, fm.DistributedType)
                and source.placement == fm.Placement((4, 24), "yx", "bb")
                and target.placement == fm.Placement((4, 12), "yx", "bb")
                and source.tensor == target.tensor
                and source.partial is None and target.partial is None
                and source.exclusive is None and target.exclusive is None
                and all(isinstance(p, SBPBroadCast) for t in (source, target) for p in t.axis_policies)
                and source.tensor in (fm.tensor_type("uint8", (48, 2720)),
                                      fm.tensor_type("float32", (48,)),
                                      fm.tensor_type("float32", (128,)))
                and self.parent.uses_sharded_views_for_constants()
                and sharded_view_error(source, target) is None):
            return DistributedReshardRealization.SHARDED_VIEW
        return self.parent.classify(context)


class _U4DistributionPolicy:
    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def identity(self):
        return self.parent.identity + "+u4l-owners-v6-common-q8-provenance-fullhead48"

    def reshard_value_metadata(self, module, source, target_type, op):
        from .u4_q8_boundary import q8_reshard_metadata
        return q8_reshard_metadata(module, source, target_type, op)

    def reshard_realization_policy(self):
        return _FullHead48ReadonlyViewPolicy(self.parent.reshard_realization_policy())

    def register_candidate_providers(self, registry):
        from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidateProviderRegistry
        base = DistributedCandidateProviderRegistry()
        self.parent.register_candidate_providers(base)
        registry.set_split_candidate_provider(base.split_candidate_provider)
        registered = set()
        for name in base.op_names:
            provider = base.try_get(name)
            if id(provider) not in registered:
                registered.add(id(provider))
                registry.add(_U4ReaderProvider(provider) if "math.vectorized_binary" in provider.op_names
                             else provider)
        registry.add(U4DistributionProvider())
        from .u4_common_q8_distribution import GDNCommonQ8DistributionProvider
        registry.add(GDNCommonQ8DistributionProvider())


def cyclic32_suffix_type(value, placement):
    """The fixed96 reader contract in the result's actual element units."""
    from triton.flagmega import ir as fm
    from triton.flagmega.ir.distributed_inference import tensor_of
    tensor = tensor_of(value)
    lanes = prod(getattr(tensor.dtype, "lanes", ()))
    require(placement == fm.Placement((4, 24), "yx", "bb") and lanes in (1, 8)
            and tuple(d.fixed_value for d in tensor.shape) == (1, 5120 // lanes)
            and getattr(tensor.dtype, "elem_type", tensor.dtype) == fm.DType.BFLOAT16,
            "cyclic32 residual requires the frozen BF16[1,5120] representation")
    return fm.DistributedType(tensor, (fm.SBP.broadcast(),
                fm.SBP.split_block_cyclic((0, 1), 32 // lanes)), placement)


class _CyclicReaderCandidates(DistributedCandidateProviderBase):
    op_names = frozenset({"math.vectorized_binary"})
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        from triton.flagmega.passes.auto_distributed.inference_providers import TypeInferenceCandidateProvider
        require(context.source_call.op in self.op_names, "cyclic32 reader must retain vectorized arithmetic")
        output = cyclic32_suffix_type(context.source_call.type, context.placement)
        # Use the real Pack/Unpack inverse AND its forward inference check.
        # Scalar RHS grain32 and packed BF16x8 grain4 name the same owners.
        candidates = TypeInferenceCandidateProvider(self.op_names)._inverse_candidates(context, output)
        require(len(candidates) == 1, "retained residual arithmetic cannot realize frozen cyclic32 owners")
        for source, actual in zip(context.source_call.inputs, candidates[0].input_types, strict=True):
            expected = cyclic32_suffix_type(context.module.node_map[source].type, context.placement)
            require(actual == expected, "both residual operands must retain exact cyclic32 scalar owners")
        return candidates


class _U4ReaderProvider:
    """Restrict only explicitly marked U4 readers; preserve all parent queries."""
    def __init__(self, parent):
        self.parent = parent
        self.fixed = _CyclicReaderCandidates()
        self.op_names = parent.op_names
        self.allows_partial_inputs = parent.allows_partial_inputs
        self.is_exhaustive = parent.is_exhaustive

    def _provider(self, context):
        return self.fixed if context.source_call.metadata.get("u4_cyclic32_reader") is True else self.parent

    def get_return_candidate_types(self, context, defaults):
        return self._provider(context).get_return_candidate_types(context, defaults)

    def try_get_input_type_tuples(self, context, return_type):
        return self._provider(context).try_get_input_type_tuples(context, return_type)

    def create_candidate_target(self, context, return_type):
        return self._provider(context).create_candidate_target(context, return_type)

    def create_candidate_attrs(self, context, return_type):
        return self._provider(context).create_candidate_attrs(context, return_type)

    def create_candidate(self, context, return_type, inputs):
        return self._provider(context).create_candidate(context, return_type, inputs)

    def get_candidates(self, context):
        return self._provider(context).get_candidates(context)


def enable_distribution(target=None):
    """Explicit local registration before normal AutoDistributed compilation.

    Wrap only the supplied target's policy; preserve existing providers except
    the frozen owner constraint on marked residual readers. Core target copies
    retain this wrapper and ordinary reshard realization.
    """
    from triton.flagmega.targets import get_target
    target = target or get_target("nvidia-sm120")
    if not isinstance(target.distribution_policy, _U4DistributionPolicy):
        target.distribution_policy = _U4DistributionPolicy(target.distribution_policy)
        # with_placement deliberately reconstructs generic distribution policy;
        # reinstall this explicit local extension on that new target as well.
        configure = target.with_placement

        def with_placement(placement):
            return enable_distribution(configure(placement))

        target.with_placement = with_placement
    return target


@dataclass(frozen=True)
class TaskPolicy:
    kind: str
    k: int
    components: int
    rows: int = 0
    fields: int = 1

    def __post_init__(self):
        Q8Layout(self.k, self.components)
        require(self.kind in ("ordinary_q8", "ordinary_norm_q8", "ordinary_norm_q8_c1c2", "ordinary_matrix", "ordinary_glu_warp_private_row4"),
                "unknown ordinary task")
        require(type(self.rows) is int and type(self.fields) is int, "noninteger task geometry")
        if self.kind == "ordinary_norm_q8":
            require((self.k, self.components, self.rows, self.fields) == (5120, 1, 0, 2),
                    "fused norm/C1 requires two distinct fixed full5120 result fields")
        elif self.kind == "ordinary_norm_q8_c1c2":
            require((self.k, self.components, self.rows, self.fields) == (5120, 2, 0, 3),
                    "common Norm/C1/C2 requires three distinct fixed full5120 result fields")
        elif self.kind == "ordinary_q8":
            require(self.rows == 0 and self.fields == 1, "Q8 encode has no row/GLU fields")
        elif self.kind == "ordinary_glu_warp_private_row4":
            require((self.k, self.rows, self.components, self.fields) == (5120, 17408, 2, 2),
                    "GLU row4 requires frozen dual field geometry")
        else:
            require(self.rows in SUPPORTED_ROWS and self.fields == 1, "unsupported ordinary matrix task")

    @property
    def recipe(self):
        return TaskLaunchRecipe(self.k // 32 if self.kind in ("ordinary_q8", "ordinary_norm_q8", "ordinary_norm_q8_c1c2") else self.rows,
                                4 if self.kind in ("ordinary_q8", "ordinary_norm_q8", "ordinary_norm_q8_c1c2", "ordinary_glu_warp_private_row4") else 1,
                                KernelProgramKind.ORDINARY)

    @property
    def partial_shape(self):
        return (3, self.fields, self.components, 32) if self.kind == "ordinary_matrix" else ()

    def to_data(self):
        data = dict(schema="flagmega.u4-task-policy/v1", kind=self.kind, k=self.k,
                    components=self.components, rows=self.rows, fields=self.fields,
                    execution_mesh=ExecutionMesh.dynamic().to_data(),
                    kernel_program_kind="ordinary", launch_kind="ordinary",
                    num_warps=4, block_threads=128, launch_recipe=self.recipe.to_data(),
                    partial_shape=list(self.partial_shape),
                    partial_nbytes=4 * prod(self.partial_shape) if self.partial_shape else 0,
                    partial_alignment=16, partial_publications=1 if self.partial_shape else 0,
                    grid_waits=False, grid_scratch_nbytes=0, numerical_id=NUMERICAL_ID)
        if self.kind == "ordinary_glu_warp_private_row4":
            data.update(schema="flagmega.u4-task-policy/v2", u4_schedule="warp_private_row4")
        if self.kind == "ordinary_norm_q8":
            data.update(schema="flagmega.u4-task-policy/v3", u4_schedule="norm_apply_q8_c1",
                        normalized_field=0, q8_field=1, result_nbytes=[10240, 5760],
                        output_aliases=[], canonical_global_fields=[0, 1])
        if self.kind == "ordinary_norm_q8_c1c2":
            data.update(schema="flagmega.u4-task-policy/v4", u4_schedule="norm_apply_q8_c1_c2",
                        normalized_field=0, q8_fields=[1, 2], field_components={"1": 1, "2": 2},
                        result_nbytes=[10240, 5760, 11520], output_aliases=[],
                        canonical_global_fields=[0, 1, 2], retain_all_result_stores=True)
        return data

    def result_byte_writes(self, field, cta, warp, lane):
        """A56 field-local byte owners; C1 cannot inherit producer top-level C2."""
        require(self.kind == "ordinary_norm_q8_c1c2" and type(field) is int and field in (0, 1, 2),
                "common Q8 writes require an explicit valid result field")
        self._thread(cta, warp, lane)
        require(cta < 40, "common Q8 has exactly40 physical CTAs")
        if field == 0:
            return self.normalized_byte_writes(cta, warp, lane)
        return q8_task_policy(self.k, field).q8_byte_writes(cta, warp, lane)

    def q8_byte_writes(self, cta, warp, lane):
        """Concrete byte offsets owned by one encoder thread, all components."""
        require(self.kind in ("ordinary_q8", "ordinary_norm_q8"), "Q8 writers require encode policy")
        self._thread(cta, warp, lane)
        layout = Q8Layout(self.k, self.components)
        group = 4 * cta + warp
        if group >= layout.groups:
            return ()
        result = []
        for component in range(self.components):
            base = layout.record_offset(component, group)
            if lane == 0:
                result.extend(range(base, base + 4))
            result.append(base + 4 + lane)
        return tuple(result)

    def normalized_byte_writes(self, cta, warp, lane):
        require(self.kind in ("ordinary_norm_q8", "ordinary_norm_q8_c1c2"),
                "normalized writers require fused Norm/Q8 policy")
        self._thread(cta, warp, lane)
        if self.kind == "ordinary_norm_q8_c1c2":
            require(cta < 40, "common Q8 has exactly40 physical CTAs")
        h = 32 * (4 * cta + warp) + lane
        return (2 * h, 2 * h + 1)

    def row_groups(self, warp, lane):
        require(self.kind in ("ordinary_matrix", "ordinary_glu_warp_private_row4"),
                "group reduction requires row policy")
        self._thread(0, warp, lane)
        if self.kind == "ordinary_glu_warp_private_row4":
            # Traversal order only: the last three groups each require a fresh
            # +0 temporary and an ordered RN merge into the first two groups.
            return (lane, 128 + lane, 32 + lane, 64 + lane, 96 + lane)
        return tuple(range(32 * warp + lane, self.k // 32, 128))

    def row_owner(self, cta, warp):
        require(self.kind == "ordinary_glu_warp_private_row4", "row4 owner requires GLU policy")
        self._thread(cta, warp, 0)
        return 4 * cta + warp

    def row_output_byte_writes(self, cta, warp, lane):
        require(self.kind == "ordinary_glu_warp_private_row4", "row4 BF16 writers require GLU policy")
        self._thread(cta, warp, lane)
        row = self.row_owner(cta, warp)
        return (2 * row, 2 * row + 1) if lane == 0 else ()

    def partial_byte_offset(self, warp, field, component, lane):
        require(self.kind == "ordinary_matrix", "partials require row policy")
        self._thread(0, warp, lane)
        require(1 <= warp <= 3 and integer(field) and field < self.fields
                and integer(component) and component < self.components, "invalid shared partial slot")
        return ((((warp - 1) * self.fields + field) * self.components + component) * 32 + lane) * 4

    def publication_threads(self):
        # Empty final iterations contribute +0; all threads join this barrier.
        return tuple(range(128)) if self.partial_shape else ()

    def _thread(self, cta, warp, lane):
        require(integer(cta) and cta < self.recipe.task_count and integer(warp) and warp < 4
                and integer(lane) and lane < 32, "task/thread coordinate out of bounds")


def q8_task_policy(k, components):
    return TaskPolicy("ordinary_q8", k, components)


def row_task_policy(k, rows, *, components=1, fields=1):
    kind = "ordinary_glu_warp_private_row4" if fields == 2 else "ordinary_matrix"
    return TaskPolicy(kind, k, components, rows, fields)


def task_policy_for_node(node):
    from triton.flagmega.ir.model import TupleType
    from triton.flagmega.ir.distributed_inference import tensor_of
    require(node.op in ("u4.q8_encode", "u4.iq4_matmul", "u4.iq4_glu", "u4.iq4_interleaved", "u4.norm_apply_q8_c1", "u4.norm_apply_q8_c1_c2"),
            "task policy requires an ordinary U4 operation")
    k, c = node.attrs["k"], node.attrs["components"]
    if node.op == "u4.norm_apply_q8_c1_c2":
        from .u4_common_q8_distribution import validate_fused_node
        validate_fused_node(node)
        return TaskPolicy("ordinary_norm_q8_c1c2", k, c, fields=3)
    if node.op == "u4.norm_apply_q8_c1":
        from .u4_norm_q8_distribution import validate_fused_node
        validate_fused_node(node)
        return TaskPolicy("ordinary_norm_q8", k, c, fields=2)
    if node.op == "u4.q8_encode":
        return q8_task_policy(k, c)
    values = node.type.fields if isinstance(node.type, TupleType) else (node.type,)
    tensors = tuple(tensor_of(t) for t in values)
    require(all(all(d.is_fixed for d in t.shape) for t in tensors), "dynamic row extent")
    rows = sum(prod(d.fixed_value for d in t.shape) * prod(getattr(t.dtype, "lanes", ()))
               for t in tensors)
    return row_task_policy(k, rows, components=c, fields=2 if node.op == "u4.iq4_glu" else 1)


def annotate_u4_task(node):
    if node.op == "u4.gdn_z_gather":
        return annotate_u4_gather(node)
    policy = task_policy_for_node(node)
    return replace(node, metadata={**node.metadata, "execution_mesh": ExecutionMesh.dynamic().to_data(),
                                  "kernel_program_kind": "ordinary", "kernel_num_warps": 4,
                                  "u4_task_policy": policy.to_data()})


def annotate_u4_gather(node, *, fullhead48=False):
    require(node.op == "u4.gdn_z_gather" and node.metadata.get("canonical_global") is not True,
            "private Z gather cannot be a canonical result")
    require(type(fullhead48) is bool, "fullhead48 must be an explicit boolean request")
    metadata = {k: v for k, v in node.metadata.items()
                if k not in ("u4_fuse_into_fixed96", "u4_fuse_into_fullhead48")}
    return replace(node, attrs={**node.attrs, "mapping_id": ("gdn-z-c128-private/v1" if fullhead48 else "gdn-z-c64-private/v1")},
                   metadata={**metadata,
                   "execution_mesh": ExecutionMesh.fixed((4, 12) if fullhead48 else (4, 24), axes=("y", "x")).to_data(),
                   "kernel_num_warps": 8 if fullhead48 else 4,
                   ("u4_fuse_into_fullhead48" if fullhead48 else "u4_fuse_into_fixed96"): True,
                   "u4_private_values_per_cta": 128 if fullhead48 else 64,
                   "bufferization.memory_space": "block_local_data"})


def prepare_u4_distribution(module, *, gdn_fullhead48=False):
    """Apply physical task/pool requests before ordinary compiler planning."""
    require(module.stage == "stats_combined", "U4 owner requests must precede constant freeze/distribution")
    if module.metadata.get("u4_input_norm_q8_c1c2") is not None:
        from .u4_common_q8_distribution import prepare_common_q8_ownership
        require(gdn_fullhead48 is True, "common Q8 requires fixed fullhead48 GDN")
        module = prepare_common_q8_ownership(module)
    elif module.metadata.get("u4_input_norm_q8") is not None:
        from .u4_norm_q8_distribution import prepare_norm_q8_ownership
        module = prepare_norm_q8_ownership(module)
    require(type(gdn_fullhead48) is bool, "fullhead48 requires an explicit owner request")
    nodes = {n.id: (annotate_u4_gather(n, fullhead48=gdn_fullhead48) if n.op == "u4.gdn_z_gather"
                   else annotate_u4_task(n) if n.op in U4DistributionProvider.op_names else n)
             for n in module.nodes}
    # A fusion request belongs to both operations. The program partitioner uses
    # the exact mesh identity AND the explicit resource requests in its key;
    # an inherited b0/b1 mesh or inherited four warps is not the same key.
    # Bind only the retained recurrent core reading this gather as its Z input.
    for gather in tuple(nodes.values()):
        if gather.op != "u4.gdn_z_gather":
            continue
        users = [n for n in nodes.values() if gather.id in n.inputs]
        require(len(users) == 1 and (users[0].op == "nn.gdn_recurrent_core" or
                (users[0].op == "u4.gdn_recurrent_core_q8_c2" and gdn_fullhead48 is True
                 and module.metadata.get("u4_input_norm_q8_c1c2") is not None))
                and users[0].inputs[2] == gather.id,
                "GDN Z fusion requires one retained recurrent Z reader")
        reader = users[0]
        for key in ("execution_mesh", "kernel_num_warps", "kernel_maxnreg", "kernel_program_kind"):
            if key in reader.metadata:
                old_request = ((key == "execution_mesh" and reader.metadata[key] ==
                                ExecutionMesh.fixed((4, 24), axes=("y", "x")).to_data()) or
                               (key == "kernel_num_warps" and reader.metadata[key] == 4))
                require(reader.metadata[key] == gather.metadata.get(key) or
                        (gdn_fullhead48 and old_request and reader.metadata.get("u4_gdn_z_fusion_source") == gather.id),
                        "GDN Z fusion has conflicting retained execution policy: " + key)
        nodes[reader.id] = replace(reader, metadata={**reader.metadata,
            "execution_mesh": gather.metadata["execution_mesh"],
            "kernel_num_warps": gather.metadata["kernel_num_warps"],
            "u4_gdn_z_fusion_source": gather.id,
            **({"u4_gdn_fullhead48": True} if gdn_fullhead48 else {})})
    for source in tuple(nodes.values()):
        if source.metadata.get("u4_boundary") != "canonical-to-fixed96-cyclic32":
            continue
        users = [n for n in nodes.values() if source.id in n.inputs]
        require(source.metadata.get("canonical_global") is True and len(users) == 1
                and users[0].op == "math.vectorized_binary" and users[0].attrs.get("binary_op") == "add",
                "canonical U4 output needs one retained vectorized residual reader")
        reader = users[0]
        nodes[reader.id] = replace(reader, metadata={**reader.metadata,
            "u4_cyclic32_reader": True, "u4_canonical_reader_source": source.id})
    return replace(module, nodes=tuple(nodes[n.id] for n in module.nodes))


def fixed96_launch_facts():
    return dict(schema="flagmega.u4-fixed96-launch/v1",
                execution_mesh=ExecutionMesh.fixed((4, 24), axes=("y", "x")).to_data(),
                grid=[96, 1, 1], block_threads=128, num_warps=4, launch_kind="cooperative",
                scratch_slots=25, scratch_nbytes=100, scratch_alignment=4,
                reset_before_each_invocation=True, same_stream=True,
                admission="query each native entry; all96 CTAs resident",
                scratch_lifetime="through final graph completion; no concurrent invocation reuse")


def flatten_execution(execution):
    """Expand typed serial calls into (symbol, call_path); never parse pointers."""
    require(isinstance(execution, ProgramExecution), "expected actual core ProgramExecution")
    functions = {f.function: f for f in execution.functions}
    defined, visited, calls, trace = set(), set(), [], []
    for function in execution.functions:
        for step in function.steps:
            if isinstance(step, ProgramInvocation):
                require(step.symbol not in defined, "program definition repeated between scopes")
                defined.add(step.symbol)
            else:
                require(isinstance(step, CallInvocation), "U4 has no loops; explicit control pricing required")

    def visit(name, path, active):
        require(name in functions and name not in active, "unknown/recursive function scope")
        visited.add(name)
        local = set()
        for step in functions[name].steps:
            if isinstance(step, ProgramInvocation):
                trace.append((step.symbol, path))
            else:
                require(step.call not in local, "duplicate caller identity in scope")
                local.add(step.call)
                call_path = path + (step.call,)
                calls.append((call_path, step.callee))
                visit(step.callee, call_path, active + (name,))
    visit(execution.entry, (), ())
    require(visited == set(functions), "unreachable function scope")
    return tuple(trace), tuple(calls)


def audit_program_execution(execution, program_kinds, *, scope, caller_map=None):
    """Compare supplied program structure to frozen counts, without inventing it."""
    require(scope in EXPECTED_COUNTS, "unknown U4 scope")
    trace, calls = flatten_execution(execution)
    require(set(program_kinds) == {symbol for symbol, _ in trace}, "missing/unused program definition")
    allowed = {"fixed96", "ordinary_q8", "ordinary_matrix"}
    if scope in ("full12_norm_q8", "full12_norm_q8_fullhead48"):
        allowed.add("ordinary_norm_q8")
    if scope == "full12_norm_common_q8_fullhead48":
        allowed.add("ordinary_norm_q8_c1c2")
    if scope in ("full12_norm_q8_fullhead48", "full12_norm_common_q8_fullhead48"):
        allowed.add("fixed48")
    require(set(program_kinds.values()) <= allowed,
            "unexpected bridge/control launch requires explicit pricing")
    original_calls = tuple(f"layer_{i}_call" for i in range(64))
    if caller_map is None:
        caller_map = {name: name for name in original_calls}
    require(set(caller_map) == set(original_calls) and len(set(caller_map.values())) == 64
            and all(type(value) is str and bool(value) for value in caller_map.values()),
            "explicit original-to-regenerated caller map is incomplete")
    expected_calls = tuple(((caller_map[f"layer_{i}_call"],), "decode_attention_1" if i % 4 == 3 else "decode_linear_0")
                           for i in range(64))
    require(calls == expected_calls, "all64 typed call order/callee identities differ")
    counts = Counter(program_kinds[symbol] for symbol, _ in trace)
    result = dict(definitions=len(program_kinds), **{k: counts[k] for k in sorted(allowed)}, compute=len(trace),
                  resets=counts["fixed96"] + counts["fixed48"],
                  total_nodes=len(trace) + counts["fixed96"] + counts["fixed48"])
    require(exact(result, EXPECTED_COUNTS[scope]), "program/bridge/reset census differs from frozen U4 scope")
    return result


def validate_cpp_distribution(descriptor, *, scope, program_kinds=None, module=None, caller_map=None):
    """Audit C0's actual descriptor, retaining native-source/device gates.

    scope=None permits a small real compiler falsifier; complete U4 checking
    requires a scope and matching per-definition Q8/row classifications.
    """
    from triton.flagmega.codegen.cpp.program_abi import identity as digest
    manifest = descriptor.get("cpp_programs", {})
    require(manifest.get("schema") == "flagmega.cpp-programs/v1", "missing typed CPP program ABI")
    require(manifest.get("plan_sha256") == digest({k: v for k, v in manifest.items() if k != "plan_sha256"}),
            "CPP plan hash differs")
    if module is not None:
        from triton.flagmega.runtime.cpp_programs import verified_cpp_programs
        verified_cpp_programs(descriptor, module)
    elif scope is not None:
        require(False, "full U4 validation needs authoritative regenerated module")
    contexts = descriptor.get("native_program_contexts", ())
    entries, invocations = manifest["entries"], manifest["invocations"]
    require(len(contexts) == len(entries) and bool(entries), "missing entry contexts")
    by_id = {}
    for entry, context in zip(entries, contexts):
        eid = entry["entry_id"]
        require(type(eid) is int and eid == len(by_id) and context["entry_id"] == eid, "entry identity differs")
        by_id[eid] = entry
        ordinary = entry["launch_kind"] == "ordinary"
        require(entry["launch_kind"] in ("ordinary", "cooperative") and
                context["launch_kind"] == entry["launch_kind"], "missing explicit native launch kind")
        fullhead48 = any(c.get("implementation") in (
            "tir.gdn_recurrent.persistent_warp_state_head_globalq8_qkregs_fullhead48_v1",
            "tir.gdn_recurrent.persistent_warp_state_head_externalq8_qkregs_fullhead48_v1")
            for c in context.get("render_calls", ()))
        require(tuple(entry["grid"]) == tuple(context["grid"])
                and entry["block_threads"] == (256 if fullhead48 else 128),
                "entry grid/block-thread launch differs")
        if fullhead48:
            require(not ordinary and entry["execution_mesh"] ==
                    ExecutionMesh.fixed((4, 12), axes=("y", "x")).to_data()
                    and context["num_warps"] == 8
                    and any(c.get("implementation") == "tir.local_u4.gdn_z_gather.fullhead48.v1"
                            for c in context.get("render_calls", ())),
                    "fullhead48 requires its exact cooperative4x12/8-warp entry")
        if ordinary:
            recipe = TaskLaunchRecipe.from_data(entry["launch_recipe"])
            require(recipe.program_kind is KernelProgramKind.ORDINARY
                    and tuple(entry["grid"]) == recipe.grid_upper_bound
                    and entry["program_kind"] == "ordinary", "ordinary task mapping/grid differs")
            require(entry["requires_grid_wide_sync"] is False and entry["scratch_nbytes"] == 0
                    and context.get("barrier_abi") is False and context.get("launch_scratch") is None,
                    "ordinary entry has a grid wait/reset")
        else:
            from triton.flagmega.codegen.cpp.renderer import _grid_barrier_counter_slots
            slots = _grid_barrier_counter_slots(context)
            require(tuple(entry["grid"]) == ((48, 1, 1) if fullhead48 else (96, 1, 1)) and entry["scratch_nbytes"] == 4 * slots
                    and context.get("barrier_abi") is True
                    and context.get("grid_barrier_counter_slots") == slots,
                    "fixed96 owning entry lost scratch/mesh")
            if scope is not None and not fullhead48:
                require(slots == 25, "U4 fixed96 entry lost25-slot barrier protocol")
        from triton.flagmega.codegen.cpp.programs import _check_controls
        _check_controls(context, ordinary)
    epochs, trace = [], []
    for ordinal, invocation in enumerate(invocations):
        require(invocation["ordinal"] == ordinal and invocation["entry_id"] in by_id,
                "missing/out-of-order invocation")
        entry = by_id[invocation["entry_id"]]
        require("stream_index" not in invocation and "depends_on" not in invocation,
                "concurrent stream invocation is outside serial U4")
        require(invocation.get("invocation_kind") == "program" and invocation.get("transfer") is None,
                "unexpected loop-copy/bridge/control launch")
        bindings = invocation["argument_bindings"]
        require(invocation["frame_identity"] == digest(bindings) and len(bindings) == len(entry["signature"]),
                "typed caller/frame signature differs")
        for binding in bindings:
            require(binding.get("kind") in ("argument", "pointer", "immediate", "scalar_argument"),
                    "host device scalar or untyped binding")
            require("view_byte_offset" not in binding, "untyped pointer expression")
            if entry["launch_kind"] == "ordinary" and binding["kind"] == "pointer":
                require(binding.get("scope_stride_bytes", 0) == 0, "ordinary value retains a private scope stride")
        if entry["launch_kind"] == "cooperative":
            epochs.append(dict(invocation=ordinal, entry_id=entry["entry_id"], epoch=len(epochs),
                               reset_nbytes=entry["scratch_nbytes"], stream_index=0, reset_before_entry=True))
        trace.append((entry["symbol"], tuple(invocation["call_path"])))
    require(set(by_id) == {v["entry_id"] for v in invocations}, "uninvoked native definition")
    result = dict(compute=len(invocations), resets=len(epochs), epochs=epochs)
    if scope is not None:
        require(program_kinds is not None, "full census needs actual per-definition task classification")
        execution = ProgramExecution.from_data(manifest["program_execution"])
        require(tuple(trace) == flatten_execution(execution)[0], "native invocations lost typed controls")
        counts = audit_program_execution(execution, program_kinds, scope=scope, caller_map=caller_map)
        require(counts["resets"] == len(epochs), "native reset census differs")
        result.update(counts)
    return result

