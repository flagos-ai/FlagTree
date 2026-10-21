"""Compare native/FlagMega graph intervals without assuming engine output alignment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant',choices=['native','flagmega'],required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--artifact',type=Path,default=Path('/tmp/r33_formal_tail_gig_aligned'))
parser.add_argument('--reference',type=Path)
parser.add_argument('--require-outside-no-worse',action='store_true')
parser.add_argument('--repeats',type=int,default=3)
args=parser.parse_args()
assert not args.output.exists()
os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0',CUDA_MODULE_LOADING='0')
import torch
from vllm import LLM,SamplingParams
from flagos_adapter import FlagMegaExtension

torch.set_num_threads(1)
artifact=str(args.artifact.resolve())
is_native=args.variant=='native'
extra={} if is_native else {'worker_extension_cls':'flagos_adapter.FlagMegaExtension'}
llm=LLM(model='/root/models/Qwen3-1.7B',dtype='bfloat16',max_model_len=3072,max_num_seqs=1,
        max_num_batched_tokens=3072,gpu_memory_utilization=.35,enforce_eager=not is_native,
        compilation_config={'mode':0,'cudagraph_mode':'FULL' if is_native else 'NONE',
                            'cudagraph_capture_sizes':[1] if is_native else []},
        block_size=256,num_gpu_blocks_override=16,enable_prefix_caching=False,
        enable_chunked_prefill=False,disable_log_stats=True,seed=0,**extra)
if not is_native:llm.collective_rpc('flagmega_install',args=(artifact,))
engine=llm.llm_engine
runner=engine.model_executor.driver_worker.worker.model_runner
sampling=SamplingParams(temperature=0,max_tokens=64,ignore_eos=True)
source=llm.get_tokenizer().encode('Explain why the sky is blue, and how sunlight interacts with air. ',add_special_tokens=False)
prompt=(source*3)[:32]
assert len(prompt)==32
current=None
all_requests=[]
original_replay=torch.cuda.CUDAGraph.replay
original_sampler=runner.sampler.forward
original_logits=runner.model.compute_logits
pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(4096)]
# Materialize CUDA event handles before timing.
for a,b in pairs:a.record();b.record()
torch.cuda.synchronize()
pair_index=0

def timed(call):
 global pair_index
 a,b=pairs[pair_index];pair_index+=1
 a.record()
 start=time.perf_counter()
 result=call()
 host=(time.perf_counter()-start)*1000
 b.record()
 return result,{'events':(a,b),'host_ms':host}

def replay(self,*a,**kw):
 if current is None:return original_replay(self,*a,**kw)
 result,interval=timed(lambda:original_replay(self,*a,**kw))
 current.setdefault('graphs',[]).append(interval)
 return result

def sampler(*a,**kw):
 if current is None:return original_sampler(*a,**kw)
 result,interval=timed(lambda:original_sampler(*a,**kw))
 current['sampler']=interval
 return result

def logits(*a,**kw):
 if current is None:return original_logits(*a,**kw)
 result,interval=timed(lambda:original_logits(*a,**kw))
 current['logits']=interval
 return result

torch.cuda.CUDAGraph.replay=replay
runner.sampler.forward=sampler
runner.model.compute_logits=logits
try:
 truth=None
 for request_id in range(args.repeats+1):
  engine.add_request(str(request_id),{'prompt_token_ids':prompt},sampling)
  rows=[];output=None
  while engine.has_unfinished_requests():
   current={}
   values,step=timed(engine.step)
   current['step']=step
   if values:
    assert len(values)==1
    output=values[0]
    current['output_length']=len(output.outputs[0].token_ids)
   rows.append(current)
   current=None
  assert output is not None and len(output.outputs[0].token_ids)==64
  tokens=list(output.outputs[0].token_ids)
  if truth is None:truth=tokens
  assert truth==tokens
  graph_rows=[r for r in rows if r.get('graphs')]
  assert len(graph_rows)==63,(request_id,len(graph_rows))
  for row in graph_rows:
   assert len(row['graphs'])==1 and 'sampler' in row and 'logits' in row,row.keys()
  if request_id:all_requests.append({'request_id':request_id,'token_ids':tokens,'steps':rows})
 torch.cuda.synchronize()
 samples=[]
 for request in all_requests:
  index=0
  for row in request['steps']:
   if not row.get('graphs'):continue
   index+=1
   step,graph=row['step'],row['graphs'][0]
   out={'request_id':request['request_id'],'decode_index':index,'context':32+index,
        'engine_host_ms':step['host_ms'],'graph_device_ms':graph['events'][0].elapsed_time(graph['events'][1]),
        'graph_host_launch_ms':graph['host_ms'],
        'before_graph_device_timeline_ms':step['events'][0].elapsed_time(graph['events'][0]),
        'after_graph_device_timeline_ms':graph['events'][1].elapsed_time(step['events'][1])}
   for key in ['sampler','logits']:
    interval=row[key]
    out[key+'_device_interval_ms']=interval['events'][0].elapsed_time(interval['events'][1])
    out[key+'_host_launch_ms']=interval['host_ms']
   out['outside_graph_residual_ms']=out['engine_host_ms']-out['graph_device_ms']
   samples.append(out)
 assert len(samples)==args.repeats*63
 if is_native:
  graph_description='Native FlagOS FULL model forward graph; logits head and sampler outside graph'
 else:
  graph_description='FlagMega staging+complete C++ model including logits; native sampler outside graph'
 keys=[k for k in samples[0] if k not in ['request_id','decode_index','context']]
 report={'variant':args.variant,'prompt_length':32,'decode_tokens':64,'warmups':1,'repeats':args.repeats,
         'device':torch.cuda.get_device_name(),'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
         'graph_boundary':graph_description,'boundary':'Events immediately surround graph replay, logits and sampler; host timer surrounds engine.step; no per-step event synchronization',
         'note':'Means are used for decomposition. Residual is host elapsed minus graph device duration, including CPU/GPU overlap; it is not the sum of isolated CPU costs. Timeline outside graph includes GPU idle while CPU submits work. Diagnostics leave published figures unchanged.',
         'tokens_match_across_requests':True,'token_ids':truth,
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'summary':{k:{'mean':statistics.mean(s[k] for s in samples),'median':statistics.median(s[k] for s in samples)} for k in keys},
         'samples':samples}
 report['adapter_sha256']=hashlib.sha256(Path(__file__).with_name('flagos_adapter.py').read_bytes()).hexdigest()
 report['decode_inputs_sha256']=hashlib.sha256(Path(__file__).with_name('decode_inputs.py').read_bytes()).hexdigest()
 if not is_native:report['executor_stats']=llm.collective_rpc('flagmega_stats')
 if args.reference:
  reference=json.loads(args.reference.read_text())
  assert truth==reference['token_ids'], 'Tokens differ from the native reference'
  report['native_token_agreement']=True
  native=reference['summary']
  native_nonmodel=native['outside_graph_residual_ms']['mean']-native['logits_device_interval_ms']['mean']
  report['native_nonmodel_residual_ms']=native_nonmodel
  report['outside_no_worse']=report['summary']['outside_graph_residual_ms']['mean'] <= native_nonmodel

 args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report['summary'],indent=2),flush=True)
 if args.require_outside_no_worse:
  assert args.reference and report['outside_no_worse'], 'Decode integration overhead exceeds native non-model residual'
finally:
 current=None
 torch.cuda.CUDAGraph.replay=original_replay
 runner.sampler.forward=original_sampler
 runner.model.compute_logits=original_logits
 engine.engine_core.shutdown()
 from vllm.distributed.parallel_state import destroy_model_parallel,destroy_distributed_environment
 destroy_model_parallel();destroy_distributed_environment()
