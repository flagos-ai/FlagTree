from pathlib import Path
import json,hashlib,subprocess
import struct
A=Path(__file__).resolve().parent
frozen=A.parent/'048-r004_v6_model_integration/qualification-attempt1'
cache=A/'ab-p3/ordinary-cache'
current={hashlib.sha256(p.read_bytes()).hexdigest():p for p in cache.rglob('*.cubin')}
loaded=json.loads((A/'ab-p3/synthetic32/round-0/r004/loaded-identity.json').read_text());new={r['name']:r for r in loaded}
def sections(p):
 data=p.read_bytes();assert data[:6]==b'\x7fELF\x02\x01'
 start=struct.unpack_from('<Q',data,40)[0];size,count,names=struct.unpack_from('<HHH',data,58)
 records=[struct.unpack_from('<IIQQQQIIQQ',data,start+i*size) for i in range(count)]
 table=records[names];strings=data[table[4]:table[4]+table[5]];out={}
 for r in records:
  name=strings[r[0]:].split(b'\0',1)[0].decode()
  if r[1]!=8:out[name]=hashlib.sha256(data[r[4]:r[4]+r[5]]).hexdigest()
 return out
rows=[]
for folder in sorted(frozen.glob('function-*')):
 old=json.loads((folder/'loaded.json').read_text());name=old['name'];p=current[new[name]['cubin_sha256']]
 x=sections(folder/'kernel.cubin');y=sections(p)
 diff=[n for n in x.keys()|y.keys() if x.get(n)!=y.get(n)]
 code=all(x.get(n)==y.get(n) for n in x if n.startswith(('.text.','.nv.constant','.nv.info.','.nv.shared.')))
 rows.append(dict(name=name,frozen_path=str(folder/'kernel.cubin'),fresh_path=str(p),cubin_same=hashlib.sha256((folder/'kernel.cubin').read_bytes()).hexdigest()==new[name]['cubin_sha256'],machine_and_launch_sections_same=code,changed_ELF_sections=diff,frozen_sections=x,fresh_sections=y))
(A/'fallback-machine-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print('MACHINE_EQUAL',sum(r['machine_and_launch_sections_same'] for r in rows),'/',len(rows))
print([(r['name'],r['changed_ELF_sections']) for r in rows if not r['cubin_same']])
