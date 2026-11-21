from pathlib import Path
import sys, os, json, hashlib, traceback, gc
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A.parent/'048-r004_v6_model_integration'))
from qualify import ModelContext, active_check
import torch
from triton.flagmega.runtime import prepared as pm
out=A/sys.argv[1];out.mkdir(exist_ok=False)
os.environ['TRITON_CACHE_DIR']=str(out/'jit-cache')
def save(n,v):(out/n).write_text(json.dumps(v,indent=2,default=str)+'\n')
rows=[]
validate=pm._validate_resources
def audit(k,contract,grid=(1,)):
 k._init_handles()
 row=dict(name=k.name,grid=list(grid),hash=k.hash,cubin_sha256=hashlib.sha256(k.asm['cubin']).hexdigest(),metadata=k.metadata._asdict(),n_regs=k.n_regs)
 rows.append(row);save('resources.json',rows)
 return validate(k,contract,grid)
pm._validate_resources=audit
torch.set_num_threads(1)
c=None
status=dict(status='STARTED',MODEL_TOKEN10={},model_claim=None)
save('status.json',status)
try:
 save('profiler-before.json',active_check())
 c=ModelContext(A/'build-attempt2/artifact')
 print('PREPARED',len(c.runtime._prepared),flush=True)
 affected=[r for r in rows if r['metadata'].get('consumer_barrier_abi','').endswith('v26-global-pf2') or '_program_7_' in r['name']]
 save('affected-resources.json',affected)
 assert all(r['metadata']['ptxas_spill_load_bytes']==r['metadata']['ptxas_spill_store_bytes']==0 for r in affected)
 for name,steps,start,want in [('synthetic32',32,0,[5328,3165,264,220,16,15,15,15,36093,8627]),('chinese64',64,16,[271,248068,198,97237,96094,96876,4960,98832,105258,122826])]:
  gold=json.loads((A.parents[3]/'.local/milestones/m000/revisions/r000/measurements/r000-r4-20261116T103555Z/golden'/name/'metadata.json').read_text())
  prompt=gold['input_tokens'][:gold['prompt_token_count']]
  c.reset();tokens=[]
  for step in range(steps):
   if step<len(prompt):c.values['input_ids'].fill_(prompt[step])
   c.launch();tokens.append(int(c.token.item()))
   if step in (9,25):print(name,'STEP',step,tokens[start:start+10],flush=True)
  eager=c.snapshot(steps-1)
  result=dict(outcome='PASS' if tokens[start:start+10]==want else 'FAIL',eligible_token_ids=tokens[start:start+10],expected_token_ids=want,all_tokens=tokens,endpoint=eager)
  status['MODEL_TOKEN10'][name]=result;save('status.json',status)
  c.capture();c.reset();graph=[]
  for step in range(steps):
   if step<len(prompt):c.values['input_ids'].fill_(prompt[step])
   c.graph.replay();graph.append(int(c.token.item()))
  result['graph_tokens']=graph;result['graph_endpoint']=c.snapshot(steps-1)
  assert graph==tokens and result['graph_endpoint']==eager
  result['eager_graph_repeatability']='PASS';save('status.json',status)
  print('MODEL_TOKEN10',name,result['outcome'],flush=True)
 status['status']='PASS' if all(r['outcome']=='PASS' for r in status['MODEL_TOKEN10'].values()) else 'INVALID_TOKEN10'
 save('launch-specs.json',c.runtime.codegen['kernels'])
 # Capture typed caller arguments for complete-row verification.
 save('prepared-argument-layout.json',[dict(index=i,name=p.compiled_kernel.name,args=[dict(type=type(x).__name__,shape=list(x.shape),dtype=str(x.dtype)) if isinstance(x,torch.Tensor) else dict(type=type(x).__name__,value=str(x)) for x in p._arguments]) for i,p in enumerate(c.runtime._prepared[:38])])
 save('profiler-after.json',active_check())
except BaseException as e:
 status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
 save('status.json',status)
 if c is not None:c.close()
 gc.collect();torch.cuda.empty_cache()
print(status['status'],flush=True)
