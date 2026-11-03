"""Preserve assumption predicates while folding distinct implied comparisons."""
from triton._C.libtriton import ir,hcu

def test_assumptions_survive_range_folding(tmp_path):
 output=tmp_path/"ir"
 output.mkdir();ctx=ir.context();ir.load_dialects(ctx);hcu.load_dialects(ctx);rows=[]
 def run(name,body,args='%k: i32',return_type='i32'):
  text='module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {\n tt.func public @probe('+args+') -> '+return_type+' {\n'+body+'\n }\n}'
  f=output/(name+'-before.mlir');f.write_text(text);mod=ir.parse_mlir_module(str(f),ctx);pm=ir.pass_manager(ctx);hcu.passes.ttgpuir.add_fold_true_cmpi(pm);pm.run(mod,'C540_assumption_preservation');after=str(mod);(output/(name+'-after.mlir')).write_text(after)
  row=dict(name=name,after_cmp=after.count('arith.cmpi'),assume_lines=[l.strip()for l in after.splitlines()if'llvm.intr.assume'in l]);rows.append(row);return after,row
 for name,predicate in [('positive','sgt'),('nonnegative','sge'),('negative','slt'),('unsigned-upper','ult')]:
  after,row=run(name,'%c0 = arith.constant 7 : i32\n%p = arith.cmpi '+predicate+', %k, %c0 : i32\nllvm.intr.assume %p : i1\ntt.return %k : i32')
  assert row['after_cmp']==1 and len(row['assume_lines'])==1 and '%true'not in row['assume_lines'][0]
 # The original fact survives while a distinct implied comparison still folds.
 after,row=run('derived-still-folds','''%c0 = arith.constant 0 : i32
 %a = arith.cmpi sgt, %k, %c0 : i32
 llvm.intr.assume %a : i1
 %derived = arith.cmpi sge, %k, %c0 : i32
 %out = arith.select %derived, %k, %c0 : i32
 tt.return %out : i32''')
 assert row['after_cmp']==1 and 'arith.select'not in after and '%true'not in row['assume_lines'][0]
 after,row=run('shared-assume-and-data','''%c0 = arith.constant 0 : i32
 %a = arith.cmpi sgt, %k, %c0 : i32
 llvm.intr.assume %a : i1
 %out = arith.select %a, %k, %c0 : i32
 tt.return %out : i32''')
 assert row['after_cmp']==1 and '%true'not in row['assume_lines'][0]
 after,row=run('unconstrained-control','''%c0 = arith.constant 0 : i32
 %a = arith.cmpi sgt, %k, %c0 : i32
 %out = arith.select %a, %k, %c0 : i32
 tt.return %out : i32''')
 assert row['after_cmp']==1 and 'arith.select'in after and not row['assume_lines']
 # A genuinely constant predicate is allowed to fold independently.
 after,row=run('constant-assume','''%c0 = arith.constant 0 : i32
 %c1 = arith.constant 1 : i32
 %a = arith.cmpi sgt, %c1, %c0 : i32
 llvm.intr.assume %a : i1
 tt.return %k : i32''')
 assert row['after_cmp']==0
