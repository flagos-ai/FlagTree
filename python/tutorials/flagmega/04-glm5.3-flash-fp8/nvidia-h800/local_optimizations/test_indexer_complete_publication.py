"""All indexer effects must precede the final publication grid."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_optimizations.joint_dsa_absorbed_pool_scores import wrapper


def test_sequence_write_precedes_final_indexer_publication():
    source = wrapper('def original(): pass')
    selected = source.split('{% else %}\ndef original(): pass')[0]
    assert selected.index('__advance(') < selected.rindex('{{ entry.stage_barrier() }}')
    assert selected.index('__parallel_select(') < selected.index('__advance(')


def fixture():
    from local_optimizations.indexer_complete_publication import SCHEDULE
    definition = dict(function='decode_mla_attention_sparse.prefix', schedule=dict(consumer_events=[
        dict(kind='tir.kernel_call', family='dsa_indexer', symbol='ix', call='indexer'),
        dict(kind='barrier', scope='grid', after=['indexer', 'absorbed'], ranges=['all_index_and_query']),
        dict(kind='tir.kernel_call', family='dsa_paged_attention', symbol='latent', call='attention.latent')]))
    calls = dict(ix=dict(parameters=dict(indexer_publication_schedule=SCHEDULE),
                         internal_grid_barriers=4, joint_dsa_pool_scores=dict(count=128),
                         result='Index', dsa_absorbed_overlap=dict(symbol='absorbed')),
                 absorbed=dict(result='Query'), latent=dict(topk='Index', query='Query'))
    return definition, calls


def test_only_complete_matching_indexer_consumer_grid_is_removed():
    from local_optimizations.indexer_complete_publication import remove_redundant_caller_grid
    definition, calls = fixture()
    proof = remove_redundant_caller_grid(definition, calls)
    assert len(definition['schedule']['consumer_events']) == 2
    assert proof['removed_caller_grid']['ranges'] == ['all_index_and_query']
    assert proof['final_grid_publishes_all_indices_and_query_and_sequence']
    definition['function'] = 'decode_linear_attention_sparse.prefix'
    assert remove_redundant_caller_grid(definition, calls) is None


def test_unproven_barrier_and_pointer_aliases_are_rejected():
    import pytest
    from triton.flagmega.errors import CodegenError
    from local_optimizations.indexer_complete_publication import remove_redundant_caller_grid
    for field, value in [('scope', 'device'), ('kind', 'tir.kernel_call')]:
        definition, calls = fixture()
        definition['schedule']['consumer_events'][1][field] = value
        with pytest.raises(CodegenError):
            remove_redundant_caller_grid(definition, calls)
    for field in ('topk', 'query'):
        definition, calls = fixture()
        calls['latent'][field] = 'different_buffer'
        with pytest.raises(CodegenError):
            remove_redundant_caller_grid(definition, calls)
    definition, calls = fixture()
    calls['ix']['parameters'] = {}
    with pytest.raises(CodegenError):
        remove_redundant_caller_grid(definition, calls)
