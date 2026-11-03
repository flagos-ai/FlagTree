"""Protect the final buffer-load scale without touching other memory paths."""
import re

import pytest

from test_ready_shared_loads import load_fixture
from triton._C.libtriton import hcu

BASE = '''
declare i32 @input()
declare i1 @input_pred()
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8), i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
define void @fixture() {
entry:
  %base = call i32 @input()
  %delta = call i32 @input()
  %pred = call i1 @input_pred()
  %off = add i32 %base, %delta
  %bytes = shl i32 %off, 1
  %masked = select i1 %pred, i32 %bytes, i32 -2147483648
  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)
  store i32 %value, ptr addrspace(1) poison
  ret void
}
'''


@pytest.mark.parametrize('case', ['masked_true', 'masked_false', 'unmasked', 'shared_scale'])
def test_preserve_scale_and_idempotence(tmp_path, case):
    text = BASE
    if case == 'masked_false':
        text = text.replace('i32 %bytes, i32 -2147483648', 'i32 -2147483648, i32 %bytes')
    elif case == 'unmasked':
        text = text.replace('i32 %masked, i32 0, i32 0)', 'i32 %bytes, i32 0, i32 0)')
    elif case == 'shared_scale':
        text = text.replace('  store i32 %value,', '  %other = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %bytes, i32 0, i32 0)\n  store i32 %value,')
    module, keep = load_fixture(tmp_path, text)
    before = str(module)
    assert hcu.preserve_buffer_load_byte_offsets(module, 'gfx936') == 1
    after = str(module)
    assert len(re.findall(r'freeze i32', after)) == 1
    assert re.search(r'(%[^ ]+) = freeze i32 %off\n  %bytes = shl i32 \1, 1', after)
    assert before.count('call i32 @llvm.amdgcn.raw.ptr.buffer.load') == after.count('call i32 @llvm.amdgcn.raw.ptr.buffer.load')
    assert hcu.preserve_buffer_load_byte_offsets(module, 'gfx936') == 0
    assert str(module) == after


@pytest.mark.parametrize('case', ['other_arch', 'other_scale', 'add_offset', 'already_frozen', 'non_kernel', 'to_lds', 'user_call', 'buffer_store'])
def test_unrelated_operations_unchanged(tmp_path, case):
    text = BASE
    if case == 'other_scale':
        text = text.replace('%bytes = shl i32 %off, 1', '%bytes = shl i32 %off, 2')
    elif case == 'add_offset':
        text = text.replace('%bytes = shl i32 %off, 1', '%bytes = add i32 %off, 1')
    elif case == 'already_frozen':
        text = text.replace('  %bytes = shl i32 %off, 1', '  %ready = freeze i32 %off\n  %bytes = shl i32 %ready, 1')
    elif case == 'to_lds':
        text = text.replace('  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)\n  store i32 %value, ptr addrspace(1) poison', '  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) poison, ptr addrspace(3) poison, i32 4, i32 %masked, i32 0, i32 0, i32 0)')
    elif case == 'user_call':
        text = text.replace('@llvm.amdgcn.raw.ptr.buffer.load.i32', '@user_load')
    elif case == 'buffer_store':
        text = text.replace('define void @fixture()', 'declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8), i32, i32, i32 immarg)\ndefine void @fixture()')
        text = text.replace('  %value = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)\n  store i32 %value, ptr addrspace(1) poison', '  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 1, ptr addrspace(8) poison, i32 %masked, i32 0, i32 0)')
    module, keep = load_fixture(tmp_path, text)
    if case == 'non_kernel':
        next(fn for fn in module.get_functions() if fn.name == 'fixture').set_calling_conv(0)
    before = str(module)
    assert hcu.preserve_buffer_load_byte_offsets(module, 'gfx928' if case == 'other_arch' else 'gfx936') == 0
    assert str(module) == before
