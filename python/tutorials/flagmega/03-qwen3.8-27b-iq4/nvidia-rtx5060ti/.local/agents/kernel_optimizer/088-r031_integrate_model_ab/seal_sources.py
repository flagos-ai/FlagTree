from pathlib import Path
import json,hashlib,difflib,subprocess
A=Path(__file__).resolve().parent;R27=A.parent/'085-r027_register_40_no_spill';R17=A.parent/'077-r006_r17_gdn_prep';F=A/'source-tree/python/triton/flagmega';W=A.parents[3]
def pin(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return dict(path=str(p.absolute()),nbytes=p.stat().st_size,sha256=h.hexdigest())
closure=[];patch=[];deltas=[]
for p in sorted(F.rglob('*')):
 if not p.is_file() or p.suffix not in ('.py','.jinja'):continue
 rel=p.relative_to(F);ref=R27/'source-tree/python/triton/flagmega'/rel
 r=pin(p)
 if p.is_symlink():r['symlink_target']=str(p.resolve())
 closure.append(r)
 if not ref.exists() or p.read_bytes()!=ref.read_bytes():
  deltas.append(str(rel));old=ref.read_text() if ref.exists() else ''
  patch.extend(difflib.unified_diff(old.splitlines(True),p.read_text().splitlines(True),fromfile='a/python/triton/flagmega/'+str(rel),tofile='b/python/triton/flagmega/'+str(rel)))
(A/'integration-source.patch').write_text(''.join(patch))
(A/'source-closure.json').write_text(json.dumps(dict(branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),source_overlay=closure,harness_sources=[pin(A/n) for n in ('bootstrap.py','build.py','setup_integration.py','qualify_model.py','ab_worker.py','verify_rows.py','analyze_ab.py','write_delivery.py','verify_delivery.py','audit_fallback_machine.py','check_protected.py','build_llama_bridge.py','cpu_audit.py','seal_sources.py','RUN.md','llama_bridge.cpp','llama_bridge.so')],accepted_model_harness_sources=[pin(A.parent/'048-r004_v6_model_integration/qualify.py'),pin(W/'benchmark/benchmark_model.py')],integration_deltas_from_r27=deltas,r27_native_compiler=pin(R27/'compiler-terminal/production/libtriton.so'),r27_upstream_receipt=pin(W/'.local/agents/kernel_profiler/013-r030_r27_k9_qualify/r27-k9-receipt.json'),r17_upstream_receipt=pin(W/'.local/agents/kernel_profiler/012-r019_full_p5_industry_boundary/full-p5-k9-receipt.json'),r17_development_receipt=pin(R17/'gdn-prep-receipt.json'),r004_accepted_receipt=pin(A.parent/'048-r004_v6_model_integration/v6-model-receipt.json'),r004_accepted_ab=pin(W/'.local/agents/graph_profiler/028-r004_fullmodel_ab/fullmodel-ab-packet.json')),indent=2)+'\n')
print('SOURCE_CLOSURE',len(closure),'INTEGRATION_DELTA',len(deltas),flush=True)
