"""Select the verified visibility and join schedules in complete final IR."""
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.codegen.triton.package_plan import plain_package_value

from .visibility_publish import SCHEDULE

JOIN_FUNCTIONS = {'decode_mla_attention_sparse.join', 'decode_linear_attention_sparse.join'}


def select(module):
    fm.verify_module(module)
    if module.metadata['num_hidden_layers'] != 45 or module.metadata['num_experts'] != 288:
        raise ValueError('This schedule requires the complete GLM model')
    definitions, visibility = [], []
    for definition in module.kernel_definitions:
        selection = definition.dispatch.microkernel
        if selection.family == 'dsa_indexer' and selection.variant == 'decode':
            parameters = selection.parameters
            if (parameters['long_pool_score_ctas'] != 96
                    or parameters['long_absorbed_query_ctas'] != 32
                    or parameters['pool_score_schedule'] != 'joint_adaptive_complete_pools_absorbed_v1'):
                raise ValueError('Visibility publication requires the verified complete pool schedule')
            definition = replace(definition, dispatch=replace(definition.dispatch,
                microkernel=replace(selection, parameters={**parameters, 'visibility_publish_schedule': SCHEDULE})))
            visibility.append(definition.name)
        definitions.append(definition)
    if len(visibility) != 1:
        raise ValueError('Expected one shared DSA indexer definition')
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
    metadata = {**module.metadata, 'kernel_program_partition': tuple(revised),
                'codegen_package_plan': {**package, 'programs': tuple(revised)}}
    selected = fm.verify_module(replace(module, kernel_definitions=tuple(definitions), metadata=metadata))
    assert replace(selected, kernel_definitions=module.kernel_definitions, metadata=module.metadata) == module
    return selected, dict(visibility_definitions=visibility, join_programs=joins,
                         join_warps_before=8, join_warps_after=4,
                         all_calls_DAG_buffers_effects_unchanged=True)
