"""Pass module-owned record values as unspecialized i64 launch data."""
from pathlib import Path
import hashlib
from triton.flagmega.errors import CodegenError
OUT=Path(__file__).resolve().parent

def activate():
 from triton.flagmega.codegen.triton import renderer,tir_package,function_programs
 from triton.flagmega.codegen.triton.registry import PackageRendererSpec
 previous=tir_package.describe_tir_package
 original=function_programs._actual_bindings
 def actual(module,root,definition,call,signature,callee,physical):
  extra=definition.get('runtime_record_scalar_bindings',[]);names={x['name'] for x in extra}
  values=original(module,root,definition,call,[n for n in signature if n not in names],callee,physical)
  if extra:
   row=next(x for x in definition['function_pool_abi']['calls'] if x['caller']==module.entry and x['call']==call['call'])
   table=definition['runtime_record_scalar_payload']
   for field in extra:
    value=table[row['record_offset']+field['column']]
    if type(value) is not int or not -(2**63)<=value<2**63:raise CodegenError('Invalid signed i64 callrecord')
    values[field['name']]=dict(kind='immediate',dtype='int64',value=value)
  return values
 function_programs._actual_bindings=actual
 original_project=function_programs.project_function_programs
 def project(module,descriptor,programs,launches):
  from local_optimizations.snapshot_offsets import transform_descriptor
  # Join-owned templates resolve their address contracts afterentryprojection.
  # Their private flag/payload lifetimes are auditedseparately.
  def included(f):return not f['function'].endswith('.join')
  intermediate={**descriptor,'device_functions':[f for f in descriptor['device_functions'] if included(f)],'programs':[{**p,'device_functions':[f for f in p.get('device_functions',[]) if included(f)]} for p in programs]}
  intermediate.setdefault('implementation_override',{}).setdefault('inputs',{})
  transform_descriptor(intermediate)
  descriptor['immutable_call_record_snapshot']=intermediate['immutable_call_record_snapshot']
  descriptor['implementation_override']=intermediate['implementation_override']
  table,=descriptor['function_offset_tables'];receipts=[];seen=set()
  for p in programs:
   for f in p.get('device_functions',[]):
    if id(f) in seen:continue
    seen.add(id(f))
    if f['function'].endswith('.join'):continue
    fields=f.get('immutable_call_record_bindings')
    if not fields:continue
    if f.get('runtime_record_scalar_bindings'):continue
    if any(f['schedule'].get(k) for k in ['producer_events','auxiliary_events']):raise CodegenError('Requiresone consumer role')
    if any(e['kind']=='loop' for e in f['schedule']['consumer_events']):raise CodegenError('Requiresroot perinvocationrecord')
    names=[x['name'] for x in fields];assert not set(names)&set(f['signature_arguments'])
    f['runtime_record_scalar_bindings']=fields;f['runtime_record_scalar_payload']=table['values']
    f['typed_signature_arguments']=[*f.get('typed_signature_arguments',f['signature_arguments']),*[n+': tl.int64' for n in names]]
    f['signature_arguments']=[*f['signature_arguments'],*names];f['signature']=', '.join(f['typed_signature_arguments'])
    receipts.append(dict(function=f['function'],columns=[x['column'] for x in fields],dtype='int64',unspecialized=True,no_constexpr=True))
  result=original_project(module,descriptor,programs,launches)
  descriptor['runtime_record_scalar_inputs']=receipts
  descriptor.setdefault('implementation_override',{}).setdefault('inputs',{})['scalar_offsets.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  return result
 function_programs.project_function_programs=project
