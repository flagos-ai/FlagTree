"""Use original selector CTA0 and 127 other CTAs for complete absorbed-query tasks."""
from pathlib import Path
import ast,copy,hashlib
from jinja2 import Environment,ChoiceLoader,DictLoader
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='kernels/mla_projection/platforms/nvidia/sm90/mma.py.jinja'
def args(s):return [ast.unparse(v) for v in ast.parse('f('+s+')').body[0].value.args]
def transform(source):
    loop='    for _fm_mla_task in tl.range(({{ call.mla_replica_group.index }}), {{ call.heads * ((call.output + call.tile_n - 1) // call.tile_n) }}, {{ call.mla_replica_group.count }}):'
    replacement='    for _fm_mla_task in tl.range({% if call.dsa_absorbed_overlap | default(false) %}tl.program_id(0) - {{ call.dsa_absorbed_score_ctas | default(1) }}{% else %}({{ call.mla_replica_group.index }}){% endif %}, {{ call.heads * ((call.output + call.tile_n - 1) // call.tile_n) }}, {% if call.dsa_absorbed_overlap | default(false) %}{{ 128 - (call.dsa_absorbed_score_ctas | default(1)) }}{% else %}{{ call.mla_replica_group.count }}{% endif %}):'
    if source.count(loop)!=1:raise CodegenError('Original cooperative MLA task loop changed')
    return source.replace(loop,replacement)

def adaptive_helpers(short):
    long=short.replace('call.symbol }}','call.symbol }}__long96')
    long=long.replace('call.dsa_absorbed_score_ctas | default(1)','call.dsa_adaptive_long_score_ctas')
    selector='call.family == "mla_projection" and call.variant == "mma"'
    if long.count(selector)!=1:raise CodegenError('Complete cooperative query template changed')
    long=long.replace(selector,selector+' and call.dsa_adaptive_long_score_ctas | default(none)')
    long=long.replace('import triton.experimental.tle.language as tle\n','')
    return short+'{% if render_calls | selectattr("dsa_adaptive_long_score_ctas","defined") | list %}\n'+long+'\n{% endif %}'
def activate():
    from triton.flagmega.codegen.triton import renderer,tir_package
    from triton.flagmega.codegen.triton.registry import PackageRendererSpec
    previous=tir_package.describe_tir_package
    def describe(module):
        d=previous(module);fs=list(d['device_functions'])
        for p in d['programs']:fs.extend(p.get('device_functions',()))
        fs=list({id(f):f for f in fs}.values());calls={c['symbol']:c for c in d['render_calls']}
        f,=[f for f in fs if f['function']=='decode_mla_attention_sparse.prefix']
        events=f['schedule']['consumer_events']
        indices=[i for i,e in enumerate(events) if e.get('family')=='dsa_indexer']
        if len(indices)!=1:raise CodegenError('Expected one original complete DSA indexer')
        i=indices[0];original=copy.deepcopy(events[i:i+3])
        if [(e['kind'],e.get('family',e.get('scope'))) for e in original]!=[('tir.kernel_call','dsa_indexer'),('tir.kernel_call','mla_projection'),('barrier','grid')]:
            raise CodegenError('Original indexer/absorbed publication schedule changed')
        ix,ab=calls[events[i]['symbol']],calls[events[i+1]['symbol']]
        if not (ix['variant']=='decode' and ix['writer_active']=='(shard_coord1 == 0) & (shard_coord2 == 0)' and ix['state_capacity']==4096 and ix['index_heads']==32 and ix['index_dim']==128 and ix['index_topk']==2048 and ix['output_width']==2051):
            raise CodegenError('Full original indexer contract changed')
        if not (ab['direction']=='absorb_query' and ab['variant']=='mma' and ab['heads']==16 and ab['head_sharded'] and ab['reduction']==256 and ab['output']==512 and ab['tile_n']==64 and ab['reduction_tile']==128 and not ab['quantized']):
            raise CodegenError('Original full absorbed-query geometry changed')
        ia,aa=args(ix['signature']),args(ab['signature'])
        actual=args(events[i+1]['arguments'])
        if len(aa)!=3 or aa[-1]!='device_dptr' or ia[-1]!='device_dptr':
            raise CodegenError('Original shared full-prefix pointer ABI changed')
        ix['signature']+=', _dsa_absorbed_workspace, _dsa_absorbed_weight'
        events[i]['arguments']+=', '+', '.join(actual[:2])
        ix['dsa_absorbed_overlap']=dict(symbol=ab['symbol'],arguments='_dsa_absorbed_workspace, _dsa_absorbed_weight, device_dptr')
        ab['dsa_absorbed_overlap']=True
        del events[i+1]
        # Original indexer two inner grids and absorbed-result outer grid remain.
        if events[i+1]['kind']!='barrier' or events[i+1]['scope']!='grid':
            raise CodegenError('Original absorbed query full-grid completion missing')
        d['dsa_select_absorbed_overlap']=dict(original_region=original,
            select_owner=0,absorbed_owners=127,original128tasks_complete=True,
            original_indexer_state_sequence_and_numerics=True,original_three_grids=True,
            output_ranges_disjoint=True,original_270abi_and_windows=True)
        d['implementation_override']['inputs']['dsa_absorbed_overlap.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d
    tir_package.describe_tir_package=describe
    spec=renderer.renderer_registry._specs['bufferized-tir']
    renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)
    prior=Environment.get_template
    def selected(self,name,*a,**kw):
        if name in {'module.py.jinja',KEY}:
            source=self.loader.get_source(self,KEY)[0]
            if '_fm_mla_task' in source and 'dsa_absorbed_overlap' not in source:
                self.loader=ChoiceLoader([DictLoader({KEY:adaptive_helpers(transform(source))}),self.loader])
        return prior(self,name,*a,**kw)
    Environment.get_template=selected

