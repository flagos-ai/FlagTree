import sys; sys.path.insert(0,'python')
import torch
from triton.flagmega.artifacts import write_artifact
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.importer import import_qwen3_model
from triton.flagmega.runtime import load as load_runtime
from triton.flagmega.evaluator import TorchEvaluator, CheckpointWeightResolver, create_paged_attention_state
from python.test.flagmega.qwen3.helpers import full_checkpoint_wmma
import tempfile, pathlib

TGT="amd-gfx1201"; TOK=16
ck=full_checkpoint_wmma(num_hidden_layers=2)
imported=import_qwen3_model(ck, execution_phase='prefill', num_tokens=TOK)
compiled=Compiler(CompileOptions(target=TGT)).compile(imported).module
print('compiled stage', compiled.stage)
td=pathlib.Path(tempfile.mkdtemp())
art=write_artifact(compiled, td/'pf', target=TGT, checkpoint=ck, emit_executable=True)
src=(td/'pf'/'generated_kernels.py').read_text()
print('tl.dot in kernels:', 'tl.dot' in src)
rt=load_runtime(art, device='cuda:0')
if 'workspace' in rt._pool_values: rt._pool_values['workspace'].fill_(0xA5)
state=rt.create_state()
ref_state=create_paged_attention_state(rt.state_config, device='cuda:0')
input_ids=torch.arange(TOK, dtype=torch.int32, device='cuda:0') % ck.config['vocab_size']
ev=TorchEvaluator(CheckpointWeightResolver(ck, device='cuda:0'))
exp=ev.run(imported, {'input_ids':input_ids, 'paged_attention_kv_cache':ref_state})
exp_logits=exp[0] if isinstance(exp,(tuple,list)) else exp
# runtime outputs: prefill returns [logits, state]
outs=rt.create_outputs()
logits=outs[0] if isinstance(outs,(tuple,list)) else outs
rt.prepare(input_ids, state, logits=logits)
rt.run_into(logits, input_ids, state)
torch.cuda.synchronize()
err=(logits.float()-exp_logits.float()).abs().max().item()
print(f'prefill e2e max_abs_err: {err:.6f}  logits shape {tuple(logits.shape)}')
