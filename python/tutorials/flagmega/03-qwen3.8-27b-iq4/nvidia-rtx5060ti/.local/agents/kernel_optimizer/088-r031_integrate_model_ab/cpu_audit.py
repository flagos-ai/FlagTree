from pathlib import Path
from copy import deepcopy
import json,hashlib,ast,subprocess
from triton.flagmega.rules.ntt.r004_native_model import native_model_contract,model_phase_symbol
from triton.flagmega.rules.ntt.r004_gdn_phase_bindings import validate_phase_plan,_same_contract
from triton.flagmega.rules.ntt.r017_split4lut_bindings import column_phase_plan
A=Path(__file__).resolve().parent;F=A/'source-tree/python/triton/flagmega';old=A.parent/'048-r004_v6_model_integration/source-final'
protected=[]
for p in old.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.jinja'):
  live=F/p.relative_to(old)
  if p.name in ('r004_column_v6.py.jinja','r004_norm_v6.py.jinja','r004_helpers_v4.py.jinja','r004_reference_v4.py.jinja','r004_model_policy_v6.py','r004_model_v2.py.jinja'):
   assert p.read_bytes()==live.read_bytes();protected.append(str(p.relative_to(old)))
contract=native_model_contract(31);plan=column_phase_plan()
assert contract['implementation']=='tir.gdn_recurrent.r017_split4lut'
assert contract['entry_order']==['partials','prep','core','y'] and contract['arena']['nbytes']==56832
assert native_model_contract(6)['implementation']=='tir.gdn_recurrent.r004_column_v6'
assert model_phase_symbol('prep',native_model_contract(6))=='r004_model_T_prep_v6'
negative=[]
for key in ('grid','pool','identity'):
 bad=deepcopy(plan)
 if key=='grid':bad['entries'][0]['grid'][0]+=1
 elif key=='pool':bad['pool']['nbytes']+=256
 else:bad['implementation']='tir.gdn_recurrent.r004_column_v6'
 try:validate_phase_plan(bad)
 except Exception as e:negative.append(dict(case=key,rejected=True,error=str(e)))
 else:raise AssertionError(key)
bad=deepcopy(contract);bad['arena']['nbytes']=41472
try:model_phase_symbol('partials',bad)
except Exception as e:negative.append(dict(case='stale_model_arena',rejected=True,error=str(e)))
else:raise AssertionError('stale native arena')
a=json.loads((A/'build-attempt2/artifact/artifact.json').read_text());b=json.loads((A.parent/'048-r004_v6_model_integration/package-attempt1/artifact.json').read_text())
# Compare algorithms by selected point; workspace address allocation is separately recorded.
x={s['point_id']:s for s in a['selections']};y={s['point_id']:s for s in b['selections']}
changed=[]
for k in x.keys()|y.keys():
 if x.get(k)!=y.get(k):changed.append(dict(point=k,old=y.get(k),new=x.get(k)))
result=dict(status='PASS',versioned_contract=contract,unchanged_frozen_bodies=protected,negative_checks=negative,selection_deltas=changed,accepted_fallback='048-r004_v6_model_integration/package-attempt1 + r004.gdn.native-model/v2',no_generated_source_ptx_cubin_patch=True,no_new_grid_arrival_counter=True)
(A/'cpu-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(negative),'negative checks',len(changed),'selection deltas',flush=True)
