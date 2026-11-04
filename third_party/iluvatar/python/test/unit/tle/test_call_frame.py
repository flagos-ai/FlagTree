# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Private scalar spills must survive an out-of-line callee's VGPR saves."""

import ctypes
import json

import pytest


def _scalar_spill_call_ir(private_pressure, nested_call=False, block_threads=1024):
    count = 64
    lines = [
        'target triple = "bi-iluvatar-ilurt"',
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
        "declare i32 @llvm.bi.sl.gettime()",
        "define internal void @callee() noinline {",
        '  call void asm sideeffect "", "~{v32},~{v33}"()',
    ]
    if nested_call:
        lines.append("  call void @leaf()")
    lines.extend(["  ret void", "}"])
    if nested_call:
        lines.extend([
            "define internal void @leaf() noinline {",
            '  call void asm sideeffect "", "~{v34},~{v35}"()',
            "  ret void",
            "}",
        ])
    lines.extend([
        "define iluvatar_kernel void @entry(ptr addrspace(1) %out) #0 {",
        "  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()",
        "  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
        f"  %block = mul i32 %bid, {block_threads}",
        "  %id = add i32 %block, %tid",
        f"  %base = mul i32 %id, {count * 2}",
        "  %base64 = zext i32 %base to i64",
    ])
    # Scalar timestamps cannot be rematerialized after the call. Recording
    # them on both sides supplies an exact reference without assuming a clock.
    lines.extend(f"  %time{i} = call i32 @llvm.bi.sl.gettime()" for i in range(count))
    for i in range(count):
        lines.extend([
            f"  %idx_before{i} = add i64 %base64, {17 + i}",
            f"  %before{i} = getelementptr i32, ptr addrspace(1) %out, i64 %idx_before{i}",
            f"  store volatile i32 %time{i}, ptr addrspace(1) %before{i}, align 4",
        ])
    if private_pressure:
        # Occupy the VGPRs at this point so live SGPRs must spill to private
        # memory rather than to spare VGPR lanes. The callee still runs with
        # the same two callee-saved VGPRs and the same launch geometry.
        clobbers = ",".join(f"~{{v{i}}}" for i in range(256))
        lines.append(f'  call void asm sideeffect "", "{clobbers}"()')
    lines.append("  call void @callee()")
    for i in range(count):
        lines.extend([
            f"  %idx_after{i} = add i64 %base64, {17 + count + i}",
            f"  %after{i} = getelementptr i32, ptr addrspace(1) %out, i64 %idx_after{i}",
            f"  store volatile i32 %time{i}, ptr addrspace(1) %after{i}, align 4",
        ])
    lines.extend(["  ret void", "}", f'attributes #0 = {{ "nvvm.reqntid"="{block_threads}" }}'])
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("private_pressure", (False, True))
@pytest.mark.parametrize("nested_call", (False, True))
@pytest.mark.parametrize("ctas,block_threads", (
    pytest.param(1, 64, id="one-warp"),
    pytest.param(16, 1024, id="model-launch"),
))
def test_callee_save_preserves_scalar_spills(tmp_path, private_pressure, nested_call, ctas, block_threads):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not getattr(torch, "corex", False):
        pytest.skip("requires an Iluvatar COREX GPU")
    from triton._C.libtriton import iluvatar, llvm
    from triton.runtime.driver import driver

    llvm.init_targets()
    source = _scalar_spill_call_ir(private_pressure, nested_call, block_threads)
    (tmp_path / "scalar-spill-call.ll").write_text(source)
    binary = iluvatar.translate_llvmir_to_cubin(
        source, iluvatar.TARGET_TRIPLE, "ivcore11", "", [], False, False)
    (tmp_path / "scalar-spill-call.cubin").write_bytes(binary)
    module, function, registers, spills, max_threads = driver.active.utils.load_binary(
        "entry", binary, 0, 0)
    assert module and max_threads >= block_threads

    count, guard, sentinel = 64, 17, -123456789
    threads = ctas * block_threads
    output = torch.full((threads * 2 * count + 2 * guard,), sentinel,
                        dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    libcuda = ctypes.CDLL("libcuda.so.1")
    launch = libcuda.cuLaunchKernel
    launch.argtypes = [ctypes.c_void_p, *([ctypes.c_uint] * 7), ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    launch.restype = ctypes.c_int
    argument = ctypes.c_uint64(output.data_ptr())
    arguments = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.pointer(argument), ctypes.c_void_p))
    status = launch(ctypes.c_void_p(function), ctas, 1, 1, block_threads, 1, 1, 0,
                    ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), arguments, None)
    assert status == 0
    torch.cuda.synchronize()
    before, after = output[guard:-guard].view(threads, 2, count).unbind(1)
    mismatches = torch.nonzero(before != after)[:8].cpu().tolist()
    (tmp_path / "result.json").write_text(json.dumps({
        "private_pressure": private_pressure,
        "nested_call": nested_call,
        "physical_grid": [ctas, 1, 1],
        "physical_threads": block_threads,
        "compared_values": before.numel(),
        "prefix_guard_intact": bool(torch.all(output[:guard] == sentinel)),
        "suffix_guard_intact": bool(torch.all(output[-guard:] == sentinel)),
        "registers": registers,
        "spill_slots": spills,
        "mismatched_values": int((before != after).sum()),
        "examples": [{"position": p, "before": int(before[p[0], p[1]]),
                      "after": int(after[p[0], p[1]])} for p in mismatches],
    }, indent=2) + "\n")
    assert torch.all(output[:guard] == sentinel)
    assert torch.all(output[-guard:] == sentinel)
    assert torch.equal(before, after)
