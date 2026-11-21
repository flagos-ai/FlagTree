import sys, os, json, gc, hashlib, traceback, gzip, numpy as np
from pathlib import Path
A=Path(__file__).resolve().parent
sys.path.insert(0,str(A.parent/'048-r004_v6_model_integration'))
from qualify import ModelContext
from triton.flagmega.codegen.triton.tir_package import describe_tir_package
import torch
out=A/(sys.argv[1] if len(sys.argv)>1 else 'integrated-row-verification');out.mkdir(exist_ok=False)
os.environ['TRITON_CACHE_DIR']=str(A/'model-gates-attempt2/jit-cache')
torch.set_num_threads(1)
paths={'r004':A.parent/'048-r004_v6_model_integration/package-attempt1','r031':A/'build-attempt2/artifact'}
def save(n,v):(out/n).write_text(json.dumps(v,indent=2,default=str)+'\n')
status=dict(status='STARTED',seeds=[20261120,1,37],all_rows_bitwise_equal=None,failures=[],model_claim=None)
save('status.json',status)
def fingerprint(raw,rowbytes):
 b=raw.cpu().numpy().tobytes()
 return dict(sha256=hashlib.sha256(b).hexdigest(),row_sha256=[hashlib.sha256(b[i:i+rowbytes]).hexdigest() for i in range(0,len(b),rowbytes)])
try:
 for arm,path in paths.items():
  c=ModelContext(path);d=describe_tir_package(c.runtime.ir_module)
  specs=c.runtime.codegen['kernels'];programs={p['symbol']:p for p in d['programs']}
  definitions={x['function']:x for x in d['device_functions']}
  captures=[]
  for i,(p,s) in enumerate(zip(c.runtime._prepared,specs,strict=True)):
   pr=programs[s['entry_symbol']];call=pr['render_calls'][0] if len(pr['render_calls'])==1 else None
   if call is None or not (call['family']=='iq4_q8_glu' or call.get('phase') in ('prep','core','y')):continue
   fn=pr['owning_function'];defs=definitions[fn];names=pr['signature_arguments'];args=dict(zip(names,p._arguments,strict=True))
   pool=defs['function_pool_abi'];field=next(b for b in pool['bindings'] if b['formal']=='workspace')
   base=field['byte_offset']
   if 'record_column' in field:base+=int(args['_flagmega_function_offsets'][int(args['_flagmega_call_offsets'])+field['record_column']].item())
   storage=args['_flagmega_call_pool_workspace'];views=[]
   if call['family']=='iq4_q8_glu':
    abi=call['outputs'][0]['buffers'][0]['abi'];views=[('glu',storage,base+abi['pool_byte_offset'],17408*2,2)]
   else:
    arena=call['workspaces'][0]['buffers'][0]['abi']['pool_byte_offset']
    phase=call['phase']
    if phase=='prep':
     views=[(name,storage,base+arena+off,size,row) for name,off,size,row in [('qn',0,8192,512),('kn',8192,8192,512),('beta',16384,192,4),('decay_log',16640,192,4)]]
    elif phase=='core':
     views=[('core',storage,base+arena+16896,24576,512)]
     state=args['decode_linear_0_state_1'].view(torch.uint8);layer=int(args['decode_linear_0_layer_id']);views.append(('state',state,layer*3145728,3145728,512))
    else:
     abi=call['outputs'][0]['buffers'][0]['abi'];views=[('y',storage,base+abi['pool_byte_offset'],12288,256)]
   captures.append((i,s['call_path'],views))
  print('CAPTURE_SITES',arm,len(captures),flush=True);assert len(captures)==64+48*3
  lookup={i:(layer,views) for i,layer,views in captures}
  for seed in (20261120,1,37):
   c.reset();token=int(np.random.default_rng(seed).integers(1,248000));c.values['input_ids'].fill_(token)
   dest=out/f'{arm}-seed-{seed}.jsonl.gz'
   reference=None
   if arm=='r031':reference=iter(gzip.open(out/f'r004-seed-{seed}.jsonl.gz','rt'))
   with gzip.open(dest,'wt',compresslevel=1) as f:
    for i,p in enumerate(c.runtime._prepared):
     assert not c.runtime._dynamic_descriptor_indices
     p.launch(*c.arguments,stream=c.stream.cuda_stream)
     if i in lookup:
      layer,views=lookup[i];c.stream.synchronize()
      row=dict(seed=seed,input_token=token,layer=layer,outputs={})
      for name,t,start,n,rowbytes in views:
       raw=t.view(torch.uint8).reshape(-1)[start:start+n]
       assert raw.numel()==n
       if name=='glu' or name=='y':assert bool(torch.isfinite(raw.view(torch.bfloat16)).all().item())
       else:assert bool(torch.isfinite(raw.view(torch.float32)).all().item())
       row['outputs'][name]=fingerprint(raw,rowbytes)
      f.write(json.dumps(row)+'\n')
      if reference is not None:
       expected=json.loads(next(reference));assert row==expected, 'ROW_SHA_MISMATCH '+str((seed,layer,list(row['outputs'])))
   status.setdefault('endpoints',{}).setdefault(arm,{})[str(seed)]=c.snapshot(0)
   save('status.json',status);print('ALL_ROWS',arm,seed,'PASS',flush=True)
  c.close();c=None
  p=None;args=None;storage=None;state=None;t=None;raw=None
  lookup=None;captures=None;views=None
  gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
 status['all_rows_bitwise_equal']=True;status['status']='PASS'
except BaseException:
 status.update(status='FAILED',error=traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:save('status.json',status)
