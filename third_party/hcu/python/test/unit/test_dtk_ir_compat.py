from triton.backends.hcu.compiler_hcu import prepare_legacy_dtk_ir
import pytest

def test_compat_only_code_attributes_and_lifetimes():
    literal='@s = private constant [80 x i8] c"memory(write, errnomem: none) call @llvm.lifetime.start.p5(x)\\00"\n'
    code=('declare !dbg !3 void @llvm.lifetime.start.p5(ptr addrspace(5) nocapture) #1\n'
          '  call void @llvm.lifetime.start.p5(ptr addrspace(5) %x)\n'
          '  fence syncscope("workgroup-one-as") release\n'
          '  %a = load atomic i32, ptr addrspace(3) %flag syncscope("workgroup-one-as") acquire, align 4\n'
          '  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"()\n'
          'attributes #1 = { convergent memory(write, errnomem: none) }\n')
    out,err,life=prepare_legacy_dtk_ir(literal+code)
    assert literal in out and err==1 and life==2
    assert 'memory(write) }' in out
    for line in code.splitlines():
        if 'fence ' in line or ' atomic ' in line or 'sideeffect ' in line:assert line in out

def test_conservative_unspecified_effect():
    out,err,life=prepare_legacy_dtk_ir('attributes #0 = { nounwind memory(errnomem: none) }\n')
    assert err==1 and life==0 and 'memory(' not in out

def test_other_effect_requires_explicit_support():
    with pytest.raises(RuntimeError,match='unverified'):
        prepare_legacy_dtk_ir('attributes #0 = { memory(errnomem: write) }\n')

def test_original_ir_unchanged_when_no_new_constructs():
    s='define void @f() {\nret void\n}\nattributes #0 = { memory(argmem: readwrite) }\n'
    assert prepare_legacy_dtk_ir(s)==(s,0,0)


def test_attribute_string_values_stay_literal():
    s='attributes #0 = { "opaque"="memory(write, errnomem: none)" memory(write, errnomem: none) }\n'
    out,err,_=prepare_legacy_dtk_ir(s)
    assert err==1 and '"opaque"="memory(write, errnomem: none)"' in out
    assert out.endswith('memory(write) }\n')
