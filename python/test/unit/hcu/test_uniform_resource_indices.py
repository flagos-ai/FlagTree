"""Semantic guards for the late, defined wave-uniform descriptor indices."""
from importlib import import_module
import pytest
from triton._C import libtriton
from triton.backends.compiler import GPUTarget
if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU backend is not built", allow_module_level=True)
HIPBackend = import_module("triton.backends.hcu.compiler").HIPBackend


def lower(tmp_path, mode='uniform', bits=64, arch='gfx936', kernel=True):
    input_value = 'workgroup'
    count = '    %count = llvm.mlir.constant(1024 : i64) : i64'
    prepare = '    %wide = llvm.zext %id : i32 to i64'
    index = '%wide'
    if mode == 'varying': input_value = 'workitem'
    if bits == 32:
        prepare = ''
        index = '%id'
    elif bits < 32:
        prepare = f'    %narrow = llvm.trunc %id : i32 to i{bits}'
        index = '%narrow'
    elif bits > 64:
        prepare = f'    %wide = llvm.zext %id : i32 to i{bits}'
    if mode == 'undef':
        prepare = f'    %wide = llvm.mlir.undef : i{bits}'
        index = '%wide'
    if mode == 'freeze':
        prepare = f'    %unknown = llvm.mlir.undef : i{bits}\n    %wide = llvm.freeze %unknown : i{bits}'
        index = '%wide'
    if mode == 'arg':
        prepare = ''
        index = '%arg'
        bits = 64
    if mode == 'wide_count': count = '    %count = llvm.mlir.constant(4294967296 : i64) : i64'
    if mode == 'varying_count':
        count = '    %count32 = llvm.call @llvm.amdgcn.workitem.id.x() : () -> i32\n    %count = llvm.zext %count32 : i32 to i64'
    if mode == 'arg_count': count = '    %count = llvm.and %arg, %mask : i64'
    if mode == 'constant':
        prepare = f'    %wide = llvm.mlir.constant(8 : i{bits}) : i{bits}'
        index = '%wide'
    text = f'''module {{
  llvm.func @llvm.amdgcn.workgroup.id.x() -> i32
  llvm.func @llvm.amdgcn.workitem.id.x() -> i32
  llvm.func @llvm.amdgcn.make.buffer.rsrc.p8.p1(!llvm.ptr<1>, i16, i64, i32) -> !llvm.ptr<8>
  llvm.func @fixture(%base: !llvm.ptr<1>, %arg: i64) {{
    %id = llvm.call @llvm.amdgcn.{input_value}.id.x() : () -> i32
    %stride = llvm.mlir.constant(16448 : i16) : i16
    %flags = llvm.mlir.constant(131072 : i32) : i32
    %mask = llvm.mlir.constant(4294967295 : i64) : i64
{prepare}
{count}
    %address = llvm.getelementptr %base[{index}] : (!llvm.ptr<1>, i{bits}) -> !llvm.ptr<1>, i8
    %resource = llvm.call @llvm.amdgcn.make.buffer.rsrc.p8.p1(%address, %stride, %count, %flags)
      : (!llvm.ptr<1>, i16, i64, i32) -> !llvm.ptr<8>
    llvm.return
  }}
}}'''
    path = tmp_path / 'resource.mlir'
    path.write_text(text)
    ctx = libtriton.ir.context()
    libtriton.ir.load_dialects(ctx)
    HIPBackend(GPUTarget('hip', 'gfx936', 64)).load_dialects(ctx)
    mod = libtriton.ir.parse_mlir_module(str(path), ctx)
    libtriton.llvm.init_targets()
    llctx = libtriton.llvm.context()
    ll = libtriton.llvm.to_module(mod, llctx)
    libtriton.hcu.attach_target_triple(ll)
    libtriton.llvm.attach_datalayout(ll, libtriton.hcu.TARGET_TRIPLE, 'gfx936', '')
    for function in ll.get_functions():
        if function.name == 'fixture' and kernel: function.set_calling_conv(libtriton.hcu.CALLING_CONV_AMDGPU_KERNEL)
    before = str(ll)
    changed = libtriton.hcu.preserve_uniform_resource_indices(ll, arch)
    after = str(ll)
    again = libtriton.hcu.preserve_uniform_resource_indices(ll, arch)
    assert again == 0
    return changed, before, after


@pytest.mark.parametrize('bits', [1, 16, 32, 64])
def test_preserves_defined_scalar_indices_and_is_idempotent(tmp_path, bits):
    changed, before, after = lower(tmp_path, bits=bits)
    assert changed == 1
    assert 'getelementptr i8' in after and 'llvm.amdgcn.readfirstlane' in after
    assert 'hcu.uniform.resource.indices.done' in after
    assert 'ptrtoint' not in after and 'inttoptr' not in after


@pytest.mark.parametrize('mode',
                         ['varying', 'undef', 'freeze', 'arg', 'wide_count', 'varying_count', 'arg_count', 'constant'])
def test_unproved_or_unnecessary_stays_unchanged(tmp_path, mode):
    changed, before, after = lower(tmp_path, mode)
    assert changed == 0 and before == after


@pytest.mark.parametrize('arch,kernel', [('gfx942', True), ('gfx936', False)])
def test_requires_the_proved_target_kernel_domain(tmp_path, arch, kernel):
    changed, before, after = lower(tmp_path, arch=arch, kernel=kernel)
    assert changed == 0 and before == after
