"""Parallel complete DSA pools alongside the original absorbed-query work."""
from dataclasses import replace
import hashlib
from pathlib import Path
from triton.flagmega.errors import CodegenError
from .replica_work import replica_group

SCHEDULE='joint_adaptive_complete_pools_absorbed_v1'
LONG_SCORE_CTAS=96
LONG_QUERY_CTAS=32
LONG_MIN_BASE=1024
SCORE_CTAS=64
QUERY_CTAS=128-SCORE_CTAS

def implementation_model(model):
    originals=[v for v in model.implementations if v.family=='dsa_indexer' and v.variant=='decode']
    if not originals or any(v.facts.get('internal_grid_barriers')!=2 for v in originals):
        raise CodegenError('Joint DSA requires the synchronized complete original indexer')
    selected=tuple(replace(v,id=v.id+'.joint_pool_query',
        parameters={**v.parameters,'pool_score_schedule':SCHEDULE,'pool_selection_schedule':'bounded_complete_pool_sort_v1',
                    'pool_score_ctas':SCORE_CTAS,'absorbed_query_ctas':QUERY_CTAS,
                    'long_pool_score_ctas':LONG_SCORE_CTAS,'long_absorbed_query_ctas':LONG_QUERY_CTAS,
                    'long_min_base':LONG_MIN_BASE},
        facts={**v.facts,'internal_grid_barriers':3}) for v in originals)
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
    return result

def wrapper(original_wrapper):
    return '''{% if call.joint_dsa_pool_scores | default(none) %}
def {{ call.symbol }}({{ call.signature }}):
    if tl.program_id(0) == 0:
        {{ call.symbol }}__write({{ call.signature }})
{{ entry.stage_barrier() }}
    if tl.load({{ call.sequence_lengths }}).to(tl.int32) >= {{ call.joint_dsa_pool_scores.long_min_base }}:
        if tl.program_id(0) < {{ call.joint_dsa_pool_scores.long_count }}:
            {{ call.symbol }}__pool_scores__long96({{ call.signature }})
        else:
            {{ call.dsa_absorbed_overlap.symbol }}__long96({{ call.dsa_absorbed_overlap.arguments }})
    else:
        if tl.program_id(0) < {{ call.joint_dsa_pool_scores.count }}:
            {{ call.symbol }}__pool_scores({{ call.signature }})
        else:
            {{ call.dsa_absorbed_overlap.symbol }}({{ call.dsa_absorbed_overlap.arguments }})
{{ entry.stage_barrier() }}
    if tl.program_id(0) == 0:
        {{ call.symbol }}__select({{ call.signature }})
{{ entry.stage_barrier() }}
    if (tl.program_id(0) == 0) & ({{ call.advance_sequence }}):
        {{ call.symbol }}__advance({{ call.signature }})
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
                    or call['variant']!='decode' or call['internal_grid_barriers']!=3
                    or call['pool_tile']!=1024 or call['output_width']!=2051 or call['value_type']!='tl.bfloat16'
                    or call['parameters'].get('pool_score_ctas')!=SCORE_CTAS
                    or call['parameters'].get('absorbed_query_ctas')!=QUERY_CTAS
                    or call['parameters'].get('long_pool_score_ctas')!=LONG_SCORE_CTAS
                    or call['parameters'].get('long_absorbed_query_ctas')!=LONG_QUERY_CTAS
                    or call['parameters'].get('long_min_base')!=LONG_MIN_BASE or not call.get('dsa_absorbed_overlap')):
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
            call['joint_dsa_sort_tiles']=(16,64,256,512,1024)
            ab['dsa_absorbed_score_ctas']=SCORE_CTAS
            ab['dsa_adaptive_long_score_ctas']=LONG_SCORE_CTAS
            selected.append(dict(call=call['call'],score_ctas=SCORE_CTAS,query_ctas=QUERY_CTAS,
                long_score_ctas=LONG_SCORE_CTAS,long_query_ctas=LONG_QUERY_CTAS,long_min_base=LONG_MIN_BASE,
                score_owner_proof=group,query_owner_proof=query_group,original_128_query_tasks_complete=True,internal_grid_phases=3))
        if selected:
            result['joint_DSA_absorbed_pool_scores']=dict(schedule=SCHEDULE,selected=selected,
                score_math='Original complete-pool reductions and BF16 rounding',selection='Bounded score order with original invalid sentinel and low-ID ties',
                scratch='float32 output prefix captured before destructive int32 selection',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return result
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=replace(spec,describe=describe)
