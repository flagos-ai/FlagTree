"""Select the verified visibility and join schedules in complete final IR."""
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.package_plan import plain_package_value

from .visibility_publish import SCHEDULE
from .parallel_dsa_pool_sort import SCHEDULE as PARALLEL_SORT
from .batch_dsa_pool_scores import SCHEDULE as BATCH_SCORE, MIN_BASE as BATCH_MIN_BASE, BATCH
from .joint_dsa_absorbed_pool_scores import LONG_QUERY_TILE_N, LONG_QUERY_SINGLE_BATCH_POOLS

JOIN_FUNCTIONS = {'decode_mla_attention_sparse.join', 'decode_linear_attention_sparse.join'}


def select(module):
    fm.verify_module(module)
    if module.metadata['num_hidden_layers'] != 45 or module.metadata['num_experts'] != 288:
        raise ValueError('This schedule requires the complete GLM model')
    definitions, visibility, latent = [], [], []
    from .native_latent_probability_model import LEGACY_SCHEDULE, SCHEDULE as LATENT_SCHEDULE
    from .latent_segmented_compaction import SCHEDULE as COMPACTION_SCHEDULE
    from .indexer_complete_publication import SCHEDULE as PUBLICATION_SCHEDULE
    from .paired_pv_loop import SCHEDULE as PV_LOOP_SCHEDULE, MIN_COUNT as PV_PAIR_MIN_COUNT
    for definition in module.kernel_definitions:
        selection = definition.dispatch.microkernel
        if selection.family == 'dsa_indexer' and selection.variant == 'decode':
            parameters = selection.parameters
            if (parameters['long_pool_score_ctas'] != 96
                    or parameters['long_absorbed_query_ctas'] != 32
                    or parameters['pool_score_schedule'] != 'joint_adaptive_complete_pools_absorbed_v1'):
                raise ValueError('Visibility publication requires the verified complete pool schedule')
            if selection.facts.get('internal_grid_barriers') != 3:
                raise ValueError('Expected the complete three-phase input indexer')
            definition = replace(definition, dispatch=replace(definition.dispatch,
                microkernel=replace(selection,
                    parameters={**parameters, 'visibility_publish_schedule': SCHEDULE,
                                'pool_selection_schedule': PARALLEL_SORT,
                                'indexer_publication_schedule': PUBLICATION_SCHEDULE,
                                'pool_score_batch_schedule': BATCH_SCORE,
                                'pool_score_batch':BATCH,
                                'long_absorbed_query_tile_n':LONG_QUERY_TILE_N,
                                'long_absorbed_query_single_batch_pools':LONG_QUERY_SINGLE_BATCH_POOLS,
                                'pool_score_batch_min_base':BATCH_MIN_BASE},
                    facts={**selection.facts, 'internal_grid_barriers': 4})))
            visibility.append(definition.name)
        if selection.family == 'dsa_paged_attention' and selection.variant == 'decode':
            if (selection.parameters.get('latent_work_schedule') not in (LEGACY_SCHEDULE, LATENT_SCHEDULE)
                    or selection.facts.get('internal_grid_barriers') not in (2, 3)):
                raise ValueError('Immutable prefix requires the complete declared probability schedule')
            definition = replace(definition, dispatch=replace(definition.dispatch,
                microkernel=replace(selection,
                    parameters={**selection.parameters, 'latent_work_schedule': LATENT_SCHEDULE,
                                'latent_compaction_schedule': COMPACTION_SCHEDULE,
                                'latent_pv_loop_schedule': PV_LOOP_SCHEDULE,
                                'latent_pv_pair_min_count': PV_PAIR_MIN_COUNT},
                    facts={**selection.facts, 'internal_grid_barriers': 2})))
            latent.append(definition.name)
        definitions.append(definition)
    if len(latent) != 1:
        raise ValueError('Expected one complete declared latent probability definition')
    if len(visibility) != 1:
        raise ValueError('Expected one shared DSA indexer definition')
    from .lm_head_pipeline import select_definition as select_logits_definition
    definitions, logits_definition = select_logits_definition(module, definitions)
    requests = module.metadata['kernel_program_partition']
    package = module.metadata['codegen_package_plan']
    if len(requests) != 228 or plain_package_value(package['programs']) != plain_package_value(requests):
        raise ValueError('Typed program requests and package plan must agree')
    revised, joins = [], []
    for request in requests:
        callees = [module.node_map[call].attrs.get('callee') for call in request['calls']]
        if len(callees) == 1 and callees[0] in JOIN_FUNCTIONS:
            if request['num_warps'] != 8:
                raise ValueError('Expected the original eight-warp join request')
            revised.append({**request, 'num_warps': 4})
            joins.append(request['symbol'])
        else:
            revised.append(request)
    if len(joins) != 42:
        raise ValueError('Expected all 42 join invocations')
    from .lm_head_pipeline import select_programs as select_logits_programs
    revised, logits = select_logits_programs(revised)
    metadata = {**module.metadata, 'kernel_program_partition': tuple(revised),
                'kernel_program_stream_policy': 'critical_path_v2',
                'codegen_package_plan': {**package, 'programs': tuple(revised)}}
    selected = fm.verify_module(replace(module, kernel_definitions=tuple(definitions), metadata=metadata))
    assert replace(selected, kernel_definitions=module.kernel_definitions, metadata=module.metadata) == module
    return selected, dict(visibility_definitions=visibility, latent_probability_definitions=latent,
                         latent_probability_schedule=LATENT_SCHEDULE, latent_grid_barriers_max=2, join_programs=joins,
                         join_warps_before=8, join_warps_after=4,
                         logits_programs=logits, logits_warps_before=8, logits_warps_after=4,
                         logits_definition=logits_definition,
                         pool_selection_schedule=PARALLEL_SORT,
                         pool_score_batch_schedule=BATCH_SCORE,
                         pool_score_batch_min_base=BATCH_MIN_BASE,
                         indexer_grid_barriers_max=4, extra_workspace_bytes=0,
                         stream_policy='critical_path_v2',
                         all_calls_DAG_buffers_effects_unchanged=True)
