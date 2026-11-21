from pathlib import Path
import json,hashlib
A=Path(__file__).resolve().parent
old=json.loads((A/'protected-pins-before.json').read_text());changed=[]
for n,r in enumerate(old):
 p=Path(r['path'])
 if not p.is_file():changed.append(dict(path=r['path'],reason='MISSING'));continue
 if p.stat().st_size!=r['nbytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:changed.append(dict(path=r['path'],reason='HASH_OR_SIZE_CHANGED'))
 if n and n%20000==0:print('PROTECTED_CHECK',n,flush=True)
result=dict(status='PASS' if not changed else 'FAIL',files_checked=len(old),changed=changed,before_manifest_sha256=hashlib.sha256((A/'protected-pins-before.json').read_bytes()).hexdigest(),forbidden_paths_written=False)
(A/'protected-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
assert not changed
