"""Model-local provider for a measured implementation of existing P1/S2 rows."""
from dataclasses import replace

from local_optimizations.native_tp4 import NativeTp4DistributionTarget
from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal,TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.targets.nvidia.partitioned_linear import PartitionedLinearCandidateProvider
from local_optimizations.mla_input_projection import MARKER,replaced_projections

IDENTITY = "tir.partitioned_linear.wgmma_parallel_splits"


class InputProvider:
    op_names = frozenset({"nn.partitioned_linear"})

    def __init__(self,baseline):
        self.baseline = baseline

    def propose(self,node,context):
        ordinary = PartitionedLinearCandidateProvider().propose(node,replace(context,implementation_model=self.baseline))
        if MARKER not in node.metadata:
            return ordinary
        if (node.metadata[MARKER]["original"] not in replaced_projections(context.module)
                or node.attrs["partitions"] != 1 or node.attrs.get("accumulation_splits",1) != 2):
            raise ValueError("Parallel input provider requires the complete measured P1/S2 numerical contract.")
        kind = fm.logical_type(node.type)
        rows = kind.shape[-1].fixed_value * getattr(kind.dtype,"lane_count",1)
        implementation = context.implementation_model.implementation(IDENTITY)
        if implementation is None:
            raise ValueError("Requested parallel input implementation is absent.")
        workspace = dict(name="projection_partials",type=fm.tensor_type("float32",(1,2,rows)),
                         memory_space="workspace",alignment=256,lifetime="invocation")
        candidate = context.configure_implementation(implementation,semantic_parameters={"workspaces":(workspace,)})
        return TritonCandidateProposal((*ordinary.candidates,candidate),candidate.id)


class InputTarget(NativeTp4DistributionTarget):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        baseline = self.triton_implementation_model
        original = baseline.implementation("tir.partitioned_linear.wgmma_local")
        if original is None or baseline.implementation(IDENTITY) is not None:
            raise ValueError("Parallel input trial requires one known ordinary TMA row implementation.")
        implementation = replace(original,id=IDENTITY,
            parameters={**original.parameters,"parallel_accumulation_splits":True},
            facts={**original.facts,"internal_grid_barriers":1},
            requires=tuple(dict.fromkeys((*original.requires,"grid_sync"))))
        self.triton_implementation_model = replace(baseline,implementations=(*baseline.implementations,implementation))
        policy = self.tir_selection_policy
        registry = TritonCandidateProviderRegistry()
        for provider in policy.registry.providers:
            registry.add(InputProvider(baseline) if provider.op_names==InputProvider.op_names else provider)
        self.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator,registry)
