import hashlib,json,os,sys,time,statistics
from pathlib import Path
import torch
from triton.flagmega.serving.backend import ArtifactBackend
from triton.flagmega.serving import TextGenerationEngine
from triton.flagmega.serving.chat import TextStream,load_tokenizer

torch.set_num_threads(1)
b=ArtifactBackend.load('/tmp/r36_final_hoisted_query',cuda_graph=True)
t,eos=load_tokenizer('/root/models/Qwen3-1.7B',b)
engine=TextGenerationEngine(b,eos_token_ids=eos)
src=t.encode('Explain why the sky is blue, and how sunlight interacts with air. ',add_special_tokens=False)
phases={}; events=[]; active=False; current={}; pool=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(512)]
for a,z in pool:a.record();z.record()
b.synchronize();idx=0
orig_replay=b._replay;orig_sample=b.sample;orig_sync=b.synchronize;orig_append=b.append

def replay():
 global idx
 if not active:return orig_replay()
 a,z=pool[idx];idx+=1
 a.record();st=time.perf_counter();result=orig_replay();current['graph_launch_ms']=(time.perf_counter()-st)*1000;z.record();current['graph_events']=(a,z)
 return result

def sync():
 if not active:return orig_sync()
 st=time.perf_counter();result=orig_sync();current['synchronize_ms']=(time.perf_counter()-st)*1000;return result

def sample():
 global current
 st=time.perf_counter();result=orig_sample()
 if active:
  current['sample_ms']=(time.perf_counter()-st)*1000
 return result

def append(token):
 st=time.perf_counter();result=orig_append(token)
 if active:current['append_ms']=(time.perf_counter()-st)*1000
 return result
b._replay=replay;b.synchronize=sync;b.sample=sample;b.append=append
reports=[]
try:
 for label,length,profile in [('short_before',32,True),('long_warmup',2048,False),('short_after',32,True)]:
  engine.reset();b.synchronize();prompt=(src*((length+len(src)-1)//len(src)))[:length]
  rows=[];stream=TextStream(t,lambda text:None)
  def on_token(token):
   global active,current
   st=time.perf_counter();stream.put(token)
   if active:
    current['text_ms']=(time.perf_counter()-st)*1000
    rows.append(current)
   current={};active=profile
  result=engine.generate(prompt,max_new_tokens=64,ignore_eos=True,on_token=on_token);active=False;stream.finish();b.synchronize()
  for row in rows:
   a,z=row.pop('graph_events');row['graph_device_ms']=a.elapsed_time(z)
   row['outside_device_ms']=row['append_ms']+row['sample_ms']+row['text_ms']-row['graph_device_ms']
  summary={k:{'mean':statistics.mean(r[k] for r in rows),'median':statistics.median(r[k] for r in rows)} for k in rows[0]} if rows else {}
  report={'label':label,'prompt_length':length,'result':result.to_dict(),'summary':summary,'samples':rows}
  reports.append(report);print(json.dumps({'label':label,'summary':summary,'ttft_ms':result.metrics.ttft_ms}),flush=True)
finally:b.close()
Path(sys.argv[1]).write_text(json.dumps(reports,indent=2)+'\n')
