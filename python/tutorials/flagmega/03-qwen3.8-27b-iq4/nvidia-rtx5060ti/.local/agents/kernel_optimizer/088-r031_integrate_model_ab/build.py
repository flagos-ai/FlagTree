import os, sys, json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
A=Path(__file__).resolve().parent
W=A.parents[3]
sys.path.insert(0,str(W))
sys.path.insert(1,str(A.parent/'085-r027_register_40_no_spill'))
from experiment import configure, persistent_view
from triton.flagmega import Compiler, CompileOptions, ir as fm
from triton.flagmega.rules.ntt.r031_model_policy import prepare_model_column_distribution
from triton.flagmega.rules.ntt.r004_native_model import prepare_native_model
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.importer.checkpoint import checkpoint_from_path
import ortools
assert ortools.__version__=='9.10.4067'
boundary=W/'.local/milestones/m000/revisions/r003/profiles/model/r003-fullmodel-ab/builds/candidate/attempts/construction-20261117T151026325610Z/unused_functions_removed.py'
out=A/sys.argv[1];out.mkdir(exist_ok=False)
m=fm.load_module(boundary)
from agent_optimizations.iq4_streaming.graph import Q8Config,prepare_q8_distribution
m=prepare_q8_distribution(m,Q8Config());m=persistent_view(m)
m=prepare_model_column_distribution(m);m=prepare_native_model(m)
fm.emit_module(m,out/'configured.py')
c=Compiler(CompileOptions(target='nvidia-sm120',codegen_backend='triton'));c.target=configure(c.target)
m=c.compile(m).module
fm.emit_module(m,out/'compiled.py');print('COMPILED',m.semantic_hash,flush=True)
write_artifact(m,out/'artifact',target='nvidia-sm120',checkpoint=checkpoint_from_path('/home/sunnycase/.cache/flagtree-gguf-validation/qwen3.8-27b-IQ4_XS-pure.gguf'),emit_executable=True,rdata_cache_dir=W/'.local/cache/weights')
# Same-byte hard link stays within the artifact's required path boundary.
import json, os
frozen=A.parent/'048-r004_v6_model_integration/package-attempt1'
new_manifest=json.loads((out/'artifact/artifact.json').read_text())
old_manifest=json.loads((frozen/'artifact.json').read_text())
x=next(x for x in new_manifest['sections'] if x['kind']=='rdata.image/v1')
y=next(x for x in old_manifest['sections'] if x['kind']=='rdata.image/v1')
assert x==y
(out/'artifact/assets/rdata.bin').unlink()
os.link(frozen/'assets/rdata.bin',out/'artifact/assets/rdata.bin')
print('EXPORTED',flush=True)
