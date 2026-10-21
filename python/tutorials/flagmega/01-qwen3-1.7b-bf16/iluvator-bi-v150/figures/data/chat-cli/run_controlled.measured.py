import os,sys,subprocess,json,gzip
from pathlib import Path
b=Path(__file__).resolve().parents[2]
root=Path(__file__).resolve().parent
out=root/'controlled'
out.mkdir(exist_ok=False)
cli=out/'chat-cli';cli.mkdir()
ref=b/'.local/chat-cli/native-scan-reference.json'
env=dict(os.environ)
env.update(VLLM_PLUGINS='fl',FLAGOS_DEVICE_CONTROL_ENV_VAR='CUDA_VISIBLE_DEVICES',VLLM_FL_FLAGOS_BLACKLIST_APPEND='sort,sort_stable',VLLM_NO_USAGE_STATS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',CUDA_MODULE_LOADING='0',VLLM_ENABLE_V1_MULTIPROCESSING='0')
checkpoint=['--checkpoint','/root/models/Qwen3-1.7B']
artifact=['--artifact','/tmp/r36_final_hoisted_query']
for rnd in range(3):
 order=['native','flagmega','chat_cli']
 order=order[rnd:]+order[:rnd]
 for variant in order:
  if variant=='chat_cli':
   d=cli/f'process{rnd:02d}'
   cmd=['/root/miniconda3/envs/flagtree/bin/python',str(b/'benchmark_chat_cli.py'),*checkpoint,*artifact,'--native-reference',str(ref),'--results',str(d),'--rounds','1','--round-offset',str(rnd),'--repeats','5','--warmups','1','--max-start-temperature','70','--cuda-graph']
   logpath=cli/f'round{rnd:02d}.chat_cli.log'
  else:
   d=out/f'round{rnd:02d}.{variant}.json'
   cmd=[sys.executable,str(b/'benchmark_flagos.py'),*checkpoint,'--output',str(d),'--repeats','5','--warmups','1','--rotation',str(rnd),'--max-start-temperature','70']
   if variant=='native':cmd+=['--cuda-graph']
   else:cmd+=artifact
   if variant!='native' or rnd:cmd+=['--reference',str(out/'round00.native.json')]
   logpath=d.with_suffix('.log')
  with logpath.open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  if variant=='chat_cli':
   src=d/f'round{rnd:02d}.chat_cli.json'
   (cli/src.name).write_bytes(src.read_bytes())
  print(json.dumps({'round':rnd,'variant':variant,'accepted':True}),flush=True)
