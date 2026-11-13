"""Finite cross-wave scheduling proof for ordinary and legacy LDS controls."""
import pytest
from triton._C import libtriton

if not hasattr(libtriton,"hcu"):
    pytest.skip("HCU compiler required",allow_module_level=True)


def loop_source(structured=False,control="linear",overlap=False):
    name="llvm.amdgcn."+("struct" if structured else "raw")+".ptr.buffer.load.lds"
    signature="ptr addrspace(8),ptr addrspace(3),i32,i32,i32,i32,i32"+(",i32" if structured else "")
    lines=['target datalayout = "e-p:64:64-p3:32:32"',"@global_smem = external addrspace(3) global [0 x i8], align 16",
           f"declare void @{name}({signature})",
           "declare i32 @llvm.amdgcn.workitem.id.x() #0",
           "declare void @llvm.amdgcn.s.waitcnt(i32)",
           "declare void @llvm.amdgcn.s.barrier()",
           "declare <4 x float> @llvm.hcu.mmac.test(<4 x float>,<4 x float>) #0",
           "define amdgpu_kernel void @ring(ptr addrspace(8) inreg %rsrc,i32 inreg %unknown) {",
           "entry:"," %tid = call i32 @llvm.amdgcn.workitem.id.x()",
           " %wave = lshr i32 %tid,6", " %waveoff = shl i32 %wave,10",
           " %laneoff_base = shl i32 %tid,4"]
    if control in ["select","select_unknown"]:
        operand="%unknown" if control=="select_unknown" else "%tid"
        lines += [f" %parity = and i32 {operand},1",
                  " %choice = icmp eq i32 %parity,0",
                  " %swap = select i1 %choice,i32 0,i32 16",
                  " %laneoff = xor i32 %laneoff_base,%swap"]
    else:lines.append(" %laneoff = add i32 %laneoff_base,0")
    if control=="legacy":
        lines += [f" %c = and i32 %wave,{3 if structured else 1}",
                  f" %hi = shl i32 %c,{16 if structured else 18}",
                  " %copyoff = or i32 %waveoff,%hi"]
    elif control=="unknown":lines += [" %copyoff = or i32 %waveoff,1048576"]
    elif control=="nonuniform":lines += [" %copyoff = add i32 %waveoff,%tid"]
    else:lines += [" %copyoff = add i32 %waveoff,0"]
    for slot in range(4):
        for side in ["a","b"]:
            base=slot*16384+(8192 if side=="a" and not overlap else 0)
            key=f"{side}{slot}"
            lines += [f" %copyindex{key} = add i32 %copyoff,{base}",
                      f" %copy{key} = getelementptr i8,ptr addrspace(3) @global_smem,i32 %copyindex{key}",
                      f" %readindex{key} = add i32 %laneoff,{base}",
                      f" %read{key} = getelementptr i8,ptr addrspace(3) @global_smem,i32 %readindex{key}"]
    def copy(slot):
        for side in ["a","b"]:
            args=f"ptr addrspace(8) %rsrc,ptr addrspace(3) %copy{side}{slot},i32 16,i32 0,i32 0,i32 0,i32 0"+(",i32 0" if structured else "")
            lines.append(f" call void @{name}({args})")
    def fence(count):
        lines.extend([f" call void @llvm.amdgcn.s.waitcnt(i32 {0x3ff0|count})",
                      " call void @llvm.amdgcn.s.waitcnt(i32 49279)",
                      " call void @llvm.amdgcn.s.barrier()"])
    def read(slot,label):
        for side in ["a","b"]:lines.append(f" %value{side}{label} = load <8 x i16>,ptr addrspace(3) %read{side}{slot},align 16")
    for slot in [0,1,2]:copy(slot)
    fence(4);read(0,"prefix");copy(3);lines.extend([" br label %body","body:"," %iv = phi i32 [0,%entry],[%next,%body]"])
    for group,slot in enumerate([1,2,3,0]):
        fence(4);read(slot,"loop"+str(group));copy((slot+3)%4)
        lines.append(f" %mma{group} = call <4 x float> @llvm.hcu.mmac.test(<4 x float> zeroinitializer,<4 x float> zeroinitializer)")
    lines.extend([" %next = add i32 %iv,1"," %more = icmp ult i32 %next,3"," br i1 %more,label %body,label %exit","exit:"])
    for slot,count in [(1,4),(2,2),(3,0)]:fence(count);read(slot,"tail"+str(slot))
    lines.extend([" ret void","}","attributes #0 = { nounwind willreturn memory(none) }"])
    return "\n".join(lines)


@pytest.mark.parametrize("binding",["prepare_hcu_post_mma_wait"])
@pytest.mark.parametrize("structured",[False,True])
@pytest.mark.parametrize("control",["linear","legacy","select"])
def test_actual_complete_wave_domains_preserve_cross_wave_proof(binding,structured,control):
    source=loop_source(structured,control)
    output,loops,reason=getattr(libtriton.llvm,binding)(source,"gfx936",256,65536)
    assert loops==1,reason
    assert output!=source and "PASS_" in reason


@pytest.mark.parametrize("kind",["unknown","nonuniform","overlap","select_unknown"])
@pytest.mark.parametrize("binding",["prepare_hcu_post_mma_wait"])
def test_unproven_placement_and_overlapping_dma_rejected(binding,kind):
    source=loop_source(control=kind,overlap=kind=="overlap")
    output,loops,reason=getattr(libtriton.llvm,binding)(source,"gfx936",256,65536)
    assert loops==0 and output==source


@pytest.mark.parametrize("binding",["prepare_hcu_post_mma_wait"])
@pytest.mark.parametrize("predicate,truth",[
    ("eq",False),("ne",True),("ugt",True),("uge",True),("ult",False),
    ("ule",False),("sgt",False),("sge",False),("slt",True),("sle",True),
])
@pytest.mark.parametrize("valid_read",[False,True])
def test_comparison_signedness_controls_real_lds_footprint(binding,predicate,truth,valid_read):
    source=loop_source(control="select")
    source=source.replace("%choice = icmp eq i32 %parity,0",
                          f"%choice = icmp {predicate} i32 -2147483648,0")
    # The correct branch reads initialized bytes. Reversing the branches reads
    # the unwritten4KiB padding, which must fail the finite publication proof.
    zero_on_true=truth==valid_read
    true_value,false_value=(0,4096) if zero_on_true else (4096,0)
    source=source.replace("%swap = select i1 %choice,i32 0,i32 16",
        f"%swap = select i1 %choice,i32 {true_value},i32 {false_value}")
    output,loops,reason=getattr(libtriton.llvm,binding)(source,"gfx936",256,65536)
    assert loops==int(valid_read),reason
    if not valid_read:assert output==source
