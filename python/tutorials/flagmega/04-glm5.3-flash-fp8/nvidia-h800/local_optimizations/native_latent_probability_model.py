"""Declare chip-owned probability metadata and ordered long32 PV before planning."""
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.candidates.core import TritonCandidateProposal, TritonCandidateProviderRegistry
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.ir import kernel_dispatch_of
from triton.flagmega.selection import apply_plan, override_plan

IDENTITY = 'tir.dsa_paged_attention.decode.native_parallel_probability_long32'
SCHEDULE = 'complete_heads_contiguous_k_parallel_probability_long32'
SCORE_SHAPE = (4, 33, 16, 64)
METADATA_SHAPE = (4, 132, 16)
METADATA_CHIP_STRIDE = 2112
METADATA_PAYLOAD_FLOATS = 2080


def workspaces():
    return tuple(dict(name=name, type=fm.tensor_type('float32', shape),
                      memory_space='workspace', alignment=256, lifetime='invocation')
                 for name, shape in (('latent_qk_scores', SCORE_SHAPE),
                                     ('latent_probability_metadata', METADATA_SHAPE)))


def declared_model(model):
    original = model.implementation('tir.dsa_paged_attention.decode')
    if original is None or model.implementation(IDENTITY) is not None:
        raise ValueError('Probability scheduling requires one original complete decode implementation')
    selected = replace(original, id=IDENTITY,
        parameters={**original.parameters, 'compute_num_warps': 8,
                    'latent_work_schedule': SCHEDULE, 'latent_qk_tile': 256,
                    'latent_pv_tile': 64, 'latent_column_tile': 64,
                    'latent_long_column_tile': 32, 'latent_compact_pipeline': True,
                    'latent_parallel_score_tile': 64, 'latent_parallel_score_threshold': 256,
                    'latent_parallel_score_chips': 4, 'latent_parallel_score_heads': 16,
                    'latent_parallel_score_max_slots': 2051,
                    'latent_probability_metadata_stride': METADATA_CHIP_STRIDE,
                    'latent_probability_metadata_payload': METADATA_PAYLOAD_FLOATS},
        facts={**original.facts, 'internal_grid_barriers': 3},
        requires=tuple(dict.fromkeys((*original.requires, 'grid_sync'))))
    return replace(model, implementations=(*model.implementations, selected))


class ProbabilityProvider:
    def __init__(self, original):
        self.original = original
        self.op_names = original.op_names

    def propose(self, node, context):
        if node.op != 'nn.dsa_paged_attention':
            return self.original.propose(node, context)
        implementation = context.implementation_model.implementation(IDENTITY)
        if (implementation is None or node.attrs.get('probability_rounding') != 'online_bf16'
                or node.attrs.get('index_traversal') != 'compact'
                or tuple(node.attrs['layout']) != ('seq', 'head', 'dim')):
            raise ValueError('Probability scheduling requires the original complete BF16 contract')
        candidate = context.configure_implementation(implementation)
        if candidate.parameters.get('workspaces'):
            raise ValueError('Probability workspaces must be declared once before TIR materialization')
        candidate = replace(candidate, parameters={**candidate.parameters, 'workspaces': workspaces()})
        return TritonCandidateProposal((candidate,), IDENTITY, selection_kind='tir')


class ProbabilityTargetMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        policy = self.tir_selection_policy
        registry = TritonCandidateProviderRegistry()
        owners = 0
        for provider in policy.registry.providers:
            if 'nn.dsa_paged_attention' in provider.op_names:
                provider = ProbabilityProvider(provider)
                owners += 1
            registry.add(provider)
        if owners != 1:
            raise ValueError('Probability scheduling requires one complete attention provider')
        self.tir_selection_policy = TritonTirSelectionPolicy(policy.workspace_annotator, registry)

    def select_microkernels(self, module):
        choices = []
        seen = 0
        for name, function in module.kernel_callable_map.items():
            dispatch = kernel_dispatch_of(function)
            if dispatch is None or dispatch.semantic_op != 'nn.dsa_paged_attention':
                continue
            if dispatch.candidate != IDENTITY or len(dispatch.workspaces) != 2:
                raise ValueError('Probability scheduling requires normally materialized typed workspaces')
            for actual, expected in zip(dispatch.workspaces, workspaces(), strict=True):
                if actual.name != expected['name'] or actual.type != expected['type']:
                    raise ValueError('Probability workspace geometry disagrees with its declared ABI')
            seen += 1
            if dispatch.microkernel is not None:
                if dispatch.microkernel.implementation != IDENTITY:
                    raise ValueError('Physical probability selection disagrees with its TIR identity')
                continue
            choices.append((self.microkernel_selection_policy.point_id(name), IDENTITY))
        if seen != 1:
            raise ValueError('Expected one shared complete MLA attention definition')
        selected = apply_plan(module, override_plan(module, tuple(choices),
            policy='glm53.parallel-probability-long32/v1',
            rationale='Select ordered W8 PV and independent invocation-owned score/metadata workspaces.')) if choices else module
        return super().select_microkernels(selected)
