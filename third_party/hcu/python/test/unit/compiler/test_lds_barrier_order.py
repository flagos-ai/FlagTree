"""Regression for the native MachineSink barrier workaround."""
import re
import pytest
from test_ready_shared_loads import load_fixture
from triton._C.libtriton import hcu

BASE = '''
declare void @llvm.amdgcn.s.barrier()
declare void @llvm.amdgcn.wave.barrier()
declare i1 @input_pred()
define void @fixture() {
entry:
  %p = call i1 @input_pred()
  %v = load i32, ptr addrspace(3) poison
  call void @llvm.amdgcn.s.barrier()
  br i1 %p, label %use, label %exit
use:
  store i32 %v, ptr addrspace(1) poison
  br label %exit
exit:
  call void @llvm.amdgcn.wave.barrier()
  ret void
}
'''

def test_fence_before_cta_barrier_preserves_memory_operations(tmp_path):
    module, keep = load_fixture(tmp_path, BASE)
    before = str(module)
    assert hcu.preserve_lds_barrier_order(module, 'gfx936') == 1
    after = str(module)
    assert re.search(r'call void asm sideeffect "", "~\{memory\}"\(\)\n  call void @llvm.amdgcn.s.barrier', after)
    assert before.count('load i32') == after.count('load i32') == 1
    assert before.count('store i32') == after.count('store i32') == 1
    assert after.count('call void @llvm.amdgcn.wave.barrier') == 1
    assert before.count('call void @llvm.amdgcn.s.barrier') == after.count('call void @llvm.amdgcn.s.barrier') == 1

@pytest.mark.parametrize('arch', ['gfx928', 'gfx942', 'gfx950', 'gfx1250'])
def test_other_architectures_unchanged(tmp_path, arch):
    module, keep = load_fixture(tmp_path, BASE)
    before = str(module)
    assert hcu.preserve_lds_barrier_order(module, arch) == 0
    assert str(module) == before

def test_wave_barrier_only_unchanged(tmp_path):
    module, keep = load_fixture(tmp_path, BASE.replace('  call void @llvm.amdgcn.s.barrier()\n', ''))
    before = str(module)
    assert hcu.preserve_lds_barrier_order(module, 'gfx936') == 0
    assert str(module) == before
