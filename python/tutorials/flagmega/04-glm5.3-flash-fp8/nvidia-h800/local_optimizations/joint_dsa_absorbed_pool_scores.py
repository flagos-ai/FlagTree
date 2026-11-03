"""Parallel complete DSA pools alongside the original absorbed-query work."""
from dataclasses import replace
import hashlib
from pathlib import Path
from triton.flagmega.errors import CodegenError
from .replica_work import replica_group
from . import parallel_dsa_pool_sort
from . import batch_dsa_pool_scores
from .indexer_complete_publication import SCHEDULE as PUBLICATION_SCHEDULE, remove_redundant_caller_grid

SCHEDULE='joint_adaptive_complete_pools_absorbed_v1'
LONG_SCORE_CTAS=96
LONG_QUERY_CTAS=32
LONG_MIN_BASE=1024
SCORE_CTAS=64
QUERY_CTAS=128-SCORE_CTAS
LONG_QUERY_TILE_N=256
LONG_QUERY_SINGLE_BATCH_POOLS=LONG_SCORE_CTAS*batch_dsa_pool_scores.BATCH

def implementation_model(model):
    originals=[v for v in model.implementations if v.family=='dsa_indexer' and v.variant=='decode']
    if not originals or any(v.facts.get('internal_grid_barriers')!=2 for v in originals):
        raise CodegenError('Joint DSA requires the synchronized complete original indexer')
    selected=tuple(replace(v,id=v.id+'.joint_pool_query',
        parameters={**v.parameters,'pool_score_schedule':SCHEDULE,'pool_selection_schedule':parallel_dsa_pool_sort.SCHEDULE,
                    'indexer_publication_schedule':PUBLICATION_SCHEDULE,
                    'pool_score_ctas':SCORE_CTAS,'absorbed_query_ctas':QUERY_CTAS,
                    'long_pool_score_ctas':LONG_SCORE_CTAS,'long_absorbed_query_ctas':LONG_QUERY_CTAS,
                    'long_min_base':LONG_MIN_BASE,
                    'long_absorbed_query_tile_n':LONG_QUERY_TILE_N,
                    'long_absorbed_query_single_batch_pools':LONG_QUERY_SINGLE_BATCH_POOLS,
                    'pool_score_batch_schedule':batch_dsa_pool_scores.SCHEDULE,
                    'pool_score_batch':batch_dsa_pool_scores.BATCH,'pool_score_batch_min_base':batch_dsa_pool_scores.MIN_BASE},
        facts={**v.facts,'internal_grid_barriers':4}) for v in originals)
    return replace(model,implementations=(*model.implementations,*selected),preferences={**model.preferences,
        'dsa_indexer':tuple(dict.fromkeys((*[v.id for v in selected],*model.preferences.get('dsa_indexer',()))))})

def helpers(source):
    from .native_dsa_pool_scores import _template
    from .native_dsa_pool_sort import template
    changed=template(_template(source))
    anchor='@triton.jit\ndef {{ call.symbol }}({{ call.signature }}):'
    if source.count(anchor)!=1 or changed.count(anchor)!=1:
        raise CodegenError('Complete original DSA wrapper changed')
    result=changed[:changed.index(anchor)]+source[source.index(anchor):]
    result=result.replace('call.pool_score_group','call.joint_dsa_pool_scores').replace('call.pool_sort_tiles','call.joint_dsa_sort_tiles')
    start=result.index('@triton.jit(noinline=True)\ndef {{ call.symbol }}__pool_scores(')
    end=result.index('{% if call.joint_dsa_sort_tiles | default(none) %}',start)
    long_score=result[start:end].replace('__pool_scores(', '__pool_scores__long96(').replace('call.joint_dsa_pool_scores.count','call.joint_dsa_pool_scores.long_count')
    result=result[:end]+long_score+result[end:]
    from .visibility_publish import helpers as visibility_helpers
    return batch_dsa_pool_scores.helpers(parallel_dsa_pool_sort.helpers(visibility_helpers(result)))

def _long_region(query_symbol,indent):
    region='''if tl.program_id(0) < {{ call.joint_dsa_pool_scores.long_count }}:
{% if call.batch_dsa_pool_scores | default(false) %}
    {{ call.symbol }}__pool_scores__batched_long96({{ call.signature }})
{% else %}
    {{ call.symbol }}__pool_scores__long96({{ call.signature }})
{% endif %}
else:
    '''+query_symbol+'''({{ call.dsa_absorbed_overlap.arguments }})'''
    return '\n'.join(' '*indent+line if not line.startswith('{%') else line for line in region.splitlines())

def wrapper(original_wrapper):
    original_long=_long_region('{{ call.dsa_absorbed_overlap.symbol }}__long96',8)
    wide_long=_long_region('{{ call.dsa_absorbed_overlap.symbol }}__long96__wide',12)
    original_long_nested=_long_region('{{ call.dsa_absorbed_overlap.symbol }}__long96',12)
    return '''{% if call.joint_dsa_pool_scores | default(none) %}
def {{ call.symbol }}({{ call.signature }}):
    if tl.program_id(0) == 0:
        {{ call.symbol }}__write({{ call.signature }})
{{ entry.stage_barrier() }}
    if tl.load({{ call.sequence_lengths }}).to(tl.int32) >= {{ call.joint_dsa_pool_scores.long_min_base }}:
{% if call.adaptive_absorbed_query | default(none) %}
        _fm_absorbed_pool_count = tl.maximum((tl.load({{ call.sequence_lengths }}).to(tl.int32) + 1 - tl.load({{ call.result }} + {{ call.index_topk }}).to(tl.int32) + 3) // 4, 0)
        if _fm_absorbed_pool_count <= {{ call.adaptive_absorbed_query.single_batch_pools }}:
'''+wide_long+'''
        else:
'''+original_long_nested+'''
{% else %}
'''+original_long+'''
{% endif %}
    else:
        if tl.program_id(0) < {{ call.joint_dsa_pool_scores.count }}:
{% if call.batch_dsa_pool_scores | default(false) %}
            if tl.load({{ call.sequence_lengths }}).to(tl.int32) >= {{ call.pool_score_batch_min_base }}:
                {{ call.symbol }}__pool_scores__batched({{ call.signature }})
            else:
                {{ call.symbol }}__pool_scores({{ call.signature }})
{% else %}
            {{ call.symbol }}__pool_scores({{ call.signature }})
{% endif %}
        else:
            {{ call.dsa_absorbed_overlap.symbol }}({{ call.dsa_absorbed_overlap.arguments }})
{{ entry.stage_barrier() }}
'''+parallel_dsa_pool_sort.selection()+'''
    if (tl.program_id(0) == 0) & ({{ call.advance_sequence }}):
        {{ call.symbol }}__advance({{ call.signature }})
{{ entry.stage_barrier() }}
{% else %}
'''+original_wrapper+'\n{% endif %}'

def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    previous=tir_package.describe_tir_package
    def describe(module):
        result=previous(module);selected=[]
        for call in result['render_calls']:
            if call['family']!='dsa_indexer' or call['parameters'].get('pool_score_schedule')!=SCHEDULE:continue
            if (not str(module.metadata.get('numerical_contract','')).startswith('glm5-vllm-0.31-tp4/')
                    or call['variant']!='decode' or call['internal_grid_barriers']!=4
                    or call['parameters'].get('pool_selection_schedule')!=parallel_dsa_pool_sort.SCHEDULE
                    or call['parameters'].get('indexer_publication_schedule')!=PUBLICATION_SCHEDULE
                    or call['pool_tile']!=1024 or call['output_width']!=2051 or call['value_type']!='tl.bfloat16'
                    or call['parameters'].get('pool_score_ctas')!=SCORE_CTAS
                    or call['parameters'].get('absorbed_query_ctas')!=QUERY_CTAS
                    or call['parameters'].get('long_pool_score_ctas')!=LONG_SCORE_CTAS
                    or call['parameters'].get('long_absorbed_query_ctas')!=LONG_QUERY_CTAS
                    or call['parameters'].get('long_min_base')!=LONG_MIN_BASE
                    or call['parameters'].get('long_absorbed_query_tile_n')!=LONG_QUERY_TILE_N
                    or call['parameters'].get('long_absorbed_query_single_batch_pools')!=LONG_QUERY_SINGLE_BATCH_POOLS
                    or call['parameters'].get('pool_score_batch_schedule')!=batch_dsa_pool_scores.SCHEDULE
                    or call['parameters'].get('pool_score_batch')!=batch_dsa_pool_scores.BATCH
                    or call['parameters'].get('pool_score_batch_min_base')!=batch_dsa_pool_scores.MIN_BASE
                    or call['pool']!=4 or call['index_heads']!=32 or call['index_dim']!=128
                    or call['q_width']!=4096 or not call.get('dsa_absorbed_overlap')):
                raise CodegenError('Joint DSA needs the complete native TP4 score/query contract')
            group=replica_group(call,result,expressions=tuple(call[k] for k in ('q','weights','state','sequence_lengths','compress_ape','result','layer_id')))
            if group is None or group['count']!=128:
                raise CodegenError('Joint DSA needs 128 proven replicas of one complete chip owner')
            ab=next(c for c in result['render_calls'] if c['symbol']==call['dsa_absorbed_overlap']['symbol'])
            addresses=tuple(ab[k] for k in ('value','weight','scale','result','global_head','head_active','source_offset','result_offset','weight_offset','scale_offset') if isinstance(ab.get(k),str))
            query_group=replica_group(ab,result,expressions=addresses,private_canonical_inputs=('weight','scale'))
            if query_group is None or query_group['count']!=128:
                raise CodegenError('Joint DSA query needs complete owned-head operands')
            call['joint_dsa_pool_scores']=dict(index='tl.program_id(0)',count=SCORE_CTAS,long_count=LONG_SCORE_CTAS,long_min_base=LONG_MIN_BASE)
            call['parallel_dsa_sort']=True
            call['batch_dsa_pool_scores']=True
            call['pool_score_batch_min_base']=batch_dsa_pool_scores.MIN_BASE
            from .visibility_publish import configure as configure_visibility
            configure_visibility(call)
            call['joint_dsa_sort_tiles']=(16,64,256,512,1024)
            ab['dsa_absorbed_score_ctas']=SCORE_CTAS
            ab['dsa_adaptive_long_score_ctas']=LONG_SCORE_CTAS
            ab['dsa_adaptive_query_tile_n']=LONG_QUERY_TILE_N
            call['adaptive_absorbed_query']=dict(tile_n=LONG_QUERY_TILE_N,single_batch_pools=LONG_QUERY_SINGLE_BATCH_POOLS)
            selected.append(dict(call=call['call'],score_ctas=SCORE_CTAS,query_ctas=QUERY_CTAS,
                long_score_ctas=LONG_SCORE_CTAS,long_query_ctas=LONG_QUERY_CTAS,long_min_base=LONG_MIN_BASE,
                long_query_tile_n=LONG_QUERY_TILE_N,long_query_single_batch_pools=LONG_QUERY_SINGLE_BATCH_POOLS,
                score_owner_proof=group,query_owner_proof=query_group,original_128_query_tasks_complete=True,internal_grid_phases=4))
        if selected:
            functions=list(result['device_functions'])
            for program in result['programs']:functions.extend(program.get('device_functions',()))
            functions=list({id(f):f for f in functions}.values())
            calls={c['symbol']:c for c in result['render_calls']}
            publication=[r for f in functions if (r:=remove_redundant_caller_grid(f,calls)) is not None]
            if len(publication)!=1:raise CodegenError('Expected one complete indexer publication owner')
            result['indexer_complete_publication']=dict(schedule=PUBLICATION_SCHEDULE,selected=publication)
            result['joint_DSA_absorbed_pool_scores']=dict(schedule=SCHEDULE,selected=selected,
                score_math='Original complete-pool reductions and BF16 rounding',selection='Bounded score order with original invalid sentinel and low-ID ties',
                scratch='all complete score reads precede one grid barrier and disjoint int32 publication',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            result['implementation_override']['inputs']['parallel_dsa_pool_sort.py']=parallel_dsa_pool_sort.source_digest()
            result['implementation_override']['inputs']['indexer_complete_publication.py']=hashlib.sha256((Path(__file__).parent/'indexer_complete_publication.py').read_bytes()).hexdigest()
            for name in ('batch_dsa_pool_scores.py','dsa-pool-score-batch.py.jinja'):
                result['implementation_override']['inputs'][name]=hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        return result
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=replace(spec,describe=describe)
