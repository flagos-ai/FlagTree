"""Exercise the real LLVM helper with positive and unsafe-to-move fixtures."""
import re
import pytest
from triton._C.libtriton import ir, llvm

try:
    from triton._C.libtriton import hcu
except ImportError:
    pytest.skip("Requires the HCU backend", allow_module_level=True)

BASE='''
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3), i32 immarg, i32, i32, i32 immarg, i32 immarg)
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32)
declare void @llvm.amdgcn.s.waitcnt(i32 immarg)
declare void @llvm.amdgcn.s.barrier()
define void @fixture() {
entry:
  br label %loop
loop:
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) poison, ptr addrspace(3) poison, i32 8, i32 0, i32 0, i32 0, i32 0), !alias.scope !0
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) poison, ptr addrspace(3) poison, i32 8, i32 8, i32 0, i32 0, i32 0), !alias.scope !0
  %a = load <4 x i16>, ptr addrspace(3) poison, align 8, !noalias !0
  %b = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) poison, i32 0), !noalias !0
  br i1 true, label %loop, label %exit
exit:
  ret void
}
!0 = !{!1}
!1 = distinct !{!1, !2, !"async"}
!2 = distinct !{!2, !"pipeline"}
!3 = !{!4}
!4 = distinct !{!4, !2, !"different"}
'''

def load_fixture(tmp_path,text):
    # Use existing public LLVM-library import: this exercises the built helper,
    # not a Python implementation of its legality checks.
    stub=tmp_path/'root.mlir'
    stub.write_text('module { llvm.func @fixture() llvm.func @root() { llvm.call @fixture() : () -> () llvm.return } }')
    source=tmp_path/'fixture.ll';source.write_text(text)
    context=ir.context();ir.load_dialects(context);hcu.load_dialects(context)
    mlir=ir.parse_mlir_module(str(stub),context)
    llvm_context=llvm.context();module=llvm.to_module(mlir,llvm_context)
    llvm.link_extern_libs(module,[str(source)])
    fixture=next(fn for fn in module.get_functions()if fn.name=='fixture')
    fixture.set_calling_conv(hcu.CALLING_CONV_AMDGPU_KERNEL)
    return module,(context,mlir,llvm_context)

def test_ready_read_order_and_idempotence(tmp_path):
    module,keep=load_fixture(tmp_path,BASE)
    before=str(module)
    assert hcu.prefer_ready_shared_loads(module,'gfx936')==1
    after=str(module);body=after.split('define internal amdgpu_kernel',1)[1].split('exit:',1)[0]
    assert body.index('%b = call')<body.index('call void @llvm.amdgcn.sched.barrier')<body.index('call void @llvm.amdgcn.raw.ptr.buffer.load.lds')
    assert before.count('call void @llvm.amdgcn.raw.ptr.buffer.load.lds')==after.count('call void @llvm.amdgcn.raw.ptr.buffer.load.lds')==2
    assert after.count('call void @llvm.amdgcn.sched.barrier')==1
    assert hcu.prefer_ready_shared_loads(module,'gfx936')==0
    assert str(module)==after

@pytest.mark.parametrize('case', ['other_arch','missing_copy_scope','missing_read_noalias','different_scope','incomplete_scope','wait','barrier','global_store','volatile','atomic','unknown_call','inline_asm','no_native_read','single_copy','already_ordered','no_backedge'])
def test_unsafe_or_inapplicable_unchanged(tmp_path,case):
    text=BASE
    if case=='missing_copy_scope':text=text.replace(', !alias.scope !0','')
    elif case=='missing_read_noalias':text=text.replace(', !noalias !0','')
    elif case=='different_scope':text=text.replace('!noalias !0','!noalias !3')
    elif case=='incomplete_scope':text=text.replace('!0 = !{!1}','!0 = !{!1, !4}').replace('!noalias !0','!noalias !5')+'\n!5 = !{!1}\n'
    elif case in ['wait','barrier','global_store','unknown_call','inline_asm']:
        effect={'wait':'call void @llvm.amdgcn.s.waitcnt(i32 0)','barrier':'call void @llvm.amdgcn.s.barrier()','global_store':'store i32 1, ptr addrspace(1) poison','unknown_call':'call void @unknown()','inline_asm':'call void asm sideeffect "", ""()'}[case]
        text=text.replace('  %a = load','  '+effect+'\n  %a = load')
        if case=='unknown_call':text='declare void @unknown()\n'+text
    elif case=='volatile':text=text.replace('%a = load ','%a = load volatile ')
    elif case=='atomic':text=text.replace('%a = load <4 x i16>, ptr addrspace(3) poison, align 8','%a = load atomic i64, ptr addrspace(3) poison monotonic, align 8')
    elif case=='no_native_read':text=re.sub(r'  %b = call.*\n','',text)
    elif case=='single_copy':text=re.sub(r'^  call void @llvm.amdgcn.raw.ptr.buffer.load.lds.*i32 8, i32 8,.*\n','',text,flags=re.M)
    elif case=='already_ordered':
        lines=text.splitlines();copies=[line for line in lines if line.startswith('  call void @llvm.amdgcn.raw.ptr.buffer.load.lds')];lines=[line for line in lines if line not in copies];index=next(i for i,line in enumerate(lines)if line.startswith('  br i1'));text='\n'.join(lines[:index]+copies+lines[index:])
    elif case=='no_backedge':text=text.replace('br i1 true, label %loop, label %exit','br label %exit')
    module,keep=load_fixture(tmp_path,text);before=str(module)
    assert hcu.prefer_ready_shared_loads(module,'gfx942'if case=='other_arch'else'gfx936')==0
    assert str(module)==before
