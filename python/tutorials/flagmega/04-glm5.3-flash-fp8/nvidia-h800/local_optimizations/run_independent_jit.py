"""Compile the independently published artifact on one reserved idle device."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time
HERE=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=5);a=p.parse_args()
files={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for directory in ['local_optimizations'] for x in (HERE/directory).rglob('*') if x.is_file() and x.suffix in {'.py','.jinja'}}
for name in ['dag-final.py','artifact/generated_kernels.py','artifact/artifact.json','local_optimizations/reproduction-runtime-dependencies.json']:
 x=HERE/name;files[str(x)]=hashlib.sha256(x.read_bytes()).hexdigest()
(HERE/'independent-jit-inputs.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
env=dict(os.environ,BASH_ENV='/dev/null',ENV='/dev/null',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
env.pop('TRITON_PTXAS_PATH',None);env.pop('TRITON_CACHE_DIR',None)
command=[sys.executable,str(HERE/'local_optimizations/reserve_jit.py'),'--device',str(a.device),'--output',str(HERE/'independent-jit-owned'),'--',sys.executable,str(HERE/'local_optimizations/jit.py')]
started=time.monotonic()
with (HERE/'independent-jit-controller.log').open('w') as log:
 child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps(dict(event='independent_jit_started',pid=child.pid)),flush=True)
 code=child.wait()
receipt=dict(actual_exit_code=code,real_child_wait=True,pid=child.pid,seconds=time.monotonic()-started,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_receipts_unchanged=all(hashlib.sha256(Path(x).read_bytes()).hexdigest()==h for x,h in files.items()))
(HERE/'independent-jit-exit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
raise SystemExit(code)
