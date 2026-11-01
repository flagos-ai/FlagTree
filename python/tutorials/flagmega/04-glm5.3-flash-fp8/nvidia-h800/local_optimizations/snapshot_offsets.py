"""Materialize immutable dynamic record columns in the ordinary entry descriptor."""
from pathlib import Path
import hashlib,re
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent
KEY='entrypoints/call_graph.py.jinja'
PATTERN=re.compile(r"tl\.load\((?:_flagmega_call_offsets \+ |"
                   r"_flagmega_function_offsets\.to\(tl\.pointer_type\(tl\.int64\)\) \+ _flagmega_call_offsets \+ )(\d+)(?:, volatile=True)?\)")
def transform_descriptor(d):
 defs=list(d['device_functions'])
 for p in d['programs']:defs+=p.get('device_functions',[])
 defs=list({id(v):v for v in defs}.values());receipts=[]
 def collect(x):
  if isinstance(x,str):return [int(m.group(1)) for m in PATTERN.finditer(x)]
  if isinstance(x,dict):return [c for v in x.values() for c in collect(v)]
  if isinstance(x,(list,tuple)):return [c for v in x for c in collect(v)]
  return []
 def rewrite(x):
  if isinstance(x,str):return PATTERN.sub(lambda m:'_flagmega_record_column_'+m.group(1),x)
  if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
  if isinstance(x,list):return [rewrite(v) for v in x]
  if isinstance(x,tuple):return tuple(rewrite(v) for v in x)
  return x
 for definition in defs:
  if definition.get('immutable_call_record_bindings'):continue
  if definition.get('owner_matched_join'):continue
  plan=definition.get('function_pool_abi');cols=collect(definition['schedule'])
  if not cols or len(cols)==len(set(cols)):continue
  if not plan or '_flagmega_call_offsets' not in definition['signature_arguments']:raise CodegenError('No complete dynamic function record')
  if any(c<0 or c>=plan['record_width'] for c in cols):raise CodegenError('Offset outside record width')
  if any(definition['schedule'].get(k) for k in ['producer_events','auxiliary_events']):raise CodegenError('Snapshot currently requires one consumer role')
  if any(e['kind']=='loop' for e in definition['schedule']['consumer_events']):raise CodegenError('Snapshot requires immutable per-invocation record')
  columns=sorted(set(cols));definition['immutable_call_record_bindings']=[dict(column=c,name='_flagmega_record_column_'+str(c)) for c in columns]
  definition['schedule']=rewrite(definition['schedule'])
  receipts.append(dict(function=definition['function'],source_alias_reads=len(cols),snapshot_columns=columns,record_width=plan['record_width'],runtime_dynamic=True))
 if not receipts:raise CodegenError('Expected duplicate immutable offset uses')
 d['immutable_call_record_snapshot']=receipts
 d['implementation_override']['inputs']['snapshot_offsets.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 return d
def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry
 from jinja2 import ChoiceLoader,DictLoader
 previous=tir_package.describe_tir_package
 def describe(module):return transform_descriptor(previous(module))
 tir_package.describe_tir_package=describe
 spec=renderer.renderer_registry._specs['bufferized-tir'];renderer.renderer_registry._specs[spec.name]=PackageRendererSpec(spec.name,spec.kind,spec.matches,describe,spec.render)

