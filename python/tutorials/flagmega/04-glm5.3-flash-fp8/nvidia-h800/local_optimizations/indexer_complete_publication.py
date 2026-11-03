"""Publish indices, absorbed query and sequence in one final grid phase."""
import copy
from triton.flagmega.errors import CodegenError

SCHEDULE = 'indices_query_sequence_before_final_grid_v1'


def remove_redundant_caller_grid(definition, calls):
    if definition['function'] != 'decode_mla_attention_sparse.prefix':
        return None
    events = definition['schedule']['consumer_events']
    positions = [i for i, e in enumerate(events) if e.get('family') == 'dsa_indexer']
    if len(positions) != 1:
        raise CodegenError('Complete indexer publication requires one original indexer')
    i = positions[0]
    if i+2 >= len(events):
        raise CodegenError('Complete indexer publication lacks its consumer')
    event, barrier, consumer = events[i:i+3]
    ix, latent = calls.get(event['symbol']), calls.get(consumer.get('symbol'))
    if ix is None or latent is None or not ix.get('dsa_absorbed_overlap'):
        raise CodegenError('Complete publication requires typed indexer and consumer calls')
    absorbed = calls.get(ix['dsa_absorbed_overlap']['symbol'])
    if absorbed is None:
        raise CodegenError('Complete publication requires the absorbed query producer')
    if (barrier.get('kind') != 'barrier' or barrier.get('scope') != 'grid'
            or consumer.get('kind') != 'tir.kernel_call'
            or consumer.get('family') != 'dsa_paged_attention'
            or ix['parameters'].get('indexer_publication_schedule') != SCHEDULE
            or ix['internal_grid_barriers'] != 4 or not ix.get('joint_dsa_pool_scores')
            or ix['result'] != latent['topk'] or absorbed['result'] != latent['query']):
        raise CodegenError('Final grid must publish the complete typed index/query ABI')
    removed = copy.deepcopy(barrier)
    del events[i+1]
    return dict(function=definition['function'], indexer=event['call'], consumer=consumer['call'],
                removed_caller_grid=removed, final_grid_publishes_all_indices_and_query_and_sequence=True,
                original_rank_capture_grid_retained=True, original_ordered_effects_retained=True)
