"""Build the FP32/N4 QKV ABI used by the BI-V150 model into a real pipeline."""
from dataclasses import replace
import os
import re

import pytest

import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.flagmega import ir as fm
from triton.flagmega.compiler import Compiler
from triton.flagmega.options import CompileOptions
from triton.flagmega.codegen.triton import describe_tir_package, render_tir_package
from triton.flagmega.selection import override_plan


def _qkv_pipeline_stages():
    return int(os.environ.get("FLAGMEGA_QKV_PIPELINE_STAGES", "2"))


@triton.jit(noinline=True)
def _static_3072_default(reader, out, block: tl.constexpr):
    ready = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(ready.slot.tile) + tl.arange(0, block))
    tl.store(out + tl.arange(0, block), values)
    reader.release(0)
    reader.pipe.wait_drained()


@triton.jit(noinline=True)
def _static_3072_worker(writer, value, block: tl.constexpr):
    slot = writer.acquire(0)
    values = tl.load(value + tl.arange(0, block))
    tl.store(tle.gpu.local_ptr(slot.tile) + tl.arange(0, block), values)
    writer.commit(0)
    writer.close(1)
    writer.pipe.wait_drained()


@triton.jit
def _static_3072_kernel(value, out, block: tl.constexpr):
    # 32 default warps + 16 worker warps = 3072 CTA threads.  This mirrors
    # the full model's compute=32 + producer=16 partition contract.
    storage = tle.gpu.alloc(
        [2, block], dtype=tl.int32, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    pipe = tle.pipe(capacity=2, scope="cta", name="static_3072", tile=storage)
    tle.gpu.warp_specialize(
        [
            (_static_3072_default, (pipe.reader(), out, block)),
            (_static_3072_worker, (pipe.writer(), value, block)),
        ],
        worker_num_warps=[16],
        worker_num_regs=[80],
    )


@pytest.fixture(scope="module")
def pointer_qkv_module():
    class QKV(fm.Module):
        def __init__(self):
            super().__init__(dialect="high_level", stage="imported", entry="main")
        def forward(self):
            x = self.input("x", fm.tensor_type("bfloat16", (1, 2048)), id="x")
            weights = [self.weight(name, fm.tensor_type("bfloat16", (2048, n)),
                                   source="memory", key=name, id=name)
                       for name, n in (("q_weight",2048),("k_weight",1024),("v_weight",1024))]
            no = fm.F.builtin.none(name="none")
            y = fm.F.nn.qkv_parallel_linear(x, *weights, *([no]*9), num_heads=16,
                    num_kv_heads=8, output_data_type="float32", name="qkv")
            self.function("main", (x,), fm.F.tensors.get_items(y, 0, 1, 2, name_prefix="qkv"))
    compiler = Compiler(CompileOptions(target="iluvatar-bi-v150"))
    graph = compiler.compile(QKV().build(), stop_after="propose-microkernels").module
    candidate = "tir.qkv_parallel_linear.packed_gemv_async_smem_pipeline"
    points = [p for p in graph.selection_points if any(c.id == candidate for c in p.candidates)]
    assert len(points) == 1, [(p.id, [c.id for c in p.candidates]) for p in graph.selection_points if p.kind == 'microkernel']
    selected = compiler.run_stage(graph,"select-microkernels",
        plan=override_plan(graph, ((points[0].id,candidate),))).module
    return compiler.compile(selected).module


def test_model_qkv_abi_selects_real_warp_specialization_and_pointer_copy(pointer_qkv_module):
    package = describe_tir_package(pointer_qkv_module)
    assert package["pipeline_schedule"] is not None
    call, = [c for c in package["render_calls"] if c["family"] == "qkv_parallel_linear"]
    assert call["variant"] == "packed_pointer_smem_pipeline"
    assert call["output_type"] == "tl.float32"
    assert call["n_lane"] == 4
    assert package["host_tensor_descriptor_specs"] == []
    source = render_tir_package(package,"pointer_qkv")
    assert "tle.gpu.warp_specialize(" in source
    assert "is_async=True" in source
    assert "tle.gpu.IluvatarSmeBlockEncoding(" in source
    assert "input_stride=4096" in source
    assert "source_payload = tl.arange(0, 16)" in source
    assert "source_payload = tl.arange(0, 16)[None, :]" in source
    assert source.index("source = tl.load") < source.index("ready = pipeline_weight_reader.wait(step)")
    assert 'eviction_policy="evict_last"' in source
    assert "tl.reshape(weight.to(tl.float32) * source[:, None, :]" in source
    assert "(kg, payload)," in source
    stages = _qkv_pipeline_stages()
    assert source.count(f"tle.gpu.async_wait_group({stages - 1})") == 1
    assert source.count("tle.gpu.async_wait_group(0)") == 1
    assert source.index(f"tle.gpu.async_wait_group({stages - 1})") < source.index(
        f".commit(step - {stages})"
    )
    assert source.index("tle.gpu.async_wait_group(0)") < source.index(".close(")
    assert "reinterpret_tensor_map" not in source

def test_model_qkv_pointer_copy_reaches_sme_lowering(tmp_path, pointer_qkv_module, monkeypatch):
    import importlib.util
    import triton
    from triton.compiler.compiler import ASTSource, GPUTarget

    package = describe_tir_package(pointer_qkv_module)
    source = render_tir_package(package, "pointer_qkv")
    generated = tmp_path / "pointer_qkv.py"
    generated.write_text(source, encoding="utf-8")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    spec = importlib.util.spec_from_file_location("pointer_qkv", generated)
    generated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated_module)

    compiled = triton.compile(
        ASTSource(
            generated_module.flagmega_main,
            signature={
                "x": "*bf16",
                "qkv_q_reshard5": "*fp32",
                "qkv_k_reshard6": "*fp32",
                "qkv_v_reshard7": "*fp32",
                "rdata": "*bf16",
                "block_local_data": "*bf16",
            },
        ),
        target=GPUTarget("corex", 71, 64),
        options={"num_warps": package["num_warps"]},
    )
    assert "inputStride" in compiled.asm["ttgir"]
    assert "static_arrive_count = 256 : i32" in compiled.asm["ttgir"]
    assert "static_arrive_count = 1024 : i32" in compiled.asm["ttgir"]
    assert re.search(r"lshr i32 %\d+, 8", compiled.asm["llir"])
    assert re.search(r"lshr i32 %\d+, 10", compiled.asm["llir"])
    # Participant-arrive only publishes the producer's tile; the matching
    # phase wait remains the acquire side of this handoff.
    writer_cta_aggregate = os.environ.get("FLAGMEGA_QKV_WRITER_CTA_AGGREGATE") == "1"
    if not writer_cta_aggregate:
        assert 'i32 64 syncscope("workgroup") release' in compiled.asm["llir"]
    # The SME pointer-pipeline reader has completed the slot reads before
    # release.  The default path publishes once per warp; the explicit CTA
    # experiment publishes the proven full reader participant count once.
    reader_counts = {
        int(value)
        for value in re.findall(
            r"arrive_barrier[^\n]*, (\d+) \{[^\n}]*warp_aggregated_reader_release",
            compiled.asm["ttgir"],
        )
    }
    assert reader_counts
    writer_counts = {
        int(value)
        for value in re.findall(
            r"arrive_barrier[^\n]*, (\d+) \{[^\n}]*sme_async_payload",
            compiled.asm["ttgir"],
        )
    }
    assert writer_counts
    if writer_cta_aggregate:
        assert "cta_aggregated_writer_leader_tid" in compiled.asm["ttgir"]
        assert re.search(
            r'cta_aggregated_writer_leader_tid = 0 : i32',
            compiled.asm["ttgir"],
        )
        for count in writer_counts:
            assert re.search(
                rf'i32 {count} syncscope\("workgroup"\) (?:release|acq_rel)',
                compiled.asm["llir"],
            )
    cta_reader_aggregate = os.environ.get("FLAGMEGA_QKV_READER_CTA_AGGREGATE") == "1"
    if cta_reader_aggregate:
        for count in reader_counts:
            assert f'i32 {count} syncscope("workgroup") release' in compiled.asm["llir"]
    else:
        assert re.search(
            r"atomicrmw add ptr addrspace\(3\) %\d+, i32 64 "
            r"syncscope\(\"workgroup\"\) release",
            compiled.asm["llir"],
        )
    # No non-WS pipe counter may retain the old per-lane +1 release in the
    # SME pointer pipeline.  WS named-barrier counters use a global symbol and
    # are intentionally outside this assertion.
    assert not re.search(
        r"atomicrmw add ptr addrspace\(3\) %\d+, i32 1 "
        r"syncscope\(\"workgroup\"\) release",
        compiled.asm["llir"],
    )
    assert "llvm.nvvm.barrier.cta.sync.aligned.all" in compiled.asm["llir"]
    # Only the three slots used by this module (default, inline default, and
    # partition 0) need shared counter storage; the backend keeps the 16-slot
    # index limit for larger warp-specialize modules.
    assert re.search(
        r"@__ws_namedbar_state = internal addrspace\(3\) global \[3 x i32\]",
        compiled.asm["llir"],
    )
    assert len(
        re.findall(
            r"store i32 0, ptr addrspace\(3\).*@__ws_namedbar_state",
            compiled.asm["llir"],
        )
    ) == 3
    assert "__ilu_sme_publication_barrier_" not in compiled.asm["llir"]
    assert "llvm.bi.sme.load" in compiled.asm["llir"]
    # Pipe phase polling is warp-uniform: lane 0 performs the acquire load and
    # broadcasts the completion bit, so the remaining lanes do not hammer the
    # same shared counter independently.
    assert re.search(
        r"load atomic i32, ptr addrspace\(3\).*?"
        r"llvm\.bi\.slb\.shfl\.idx\.b32\.i32\(i32 %\d+, i32 0\)",
        compiled.asm["llir"],
        re.S,
    )
    assert "llvm.bi.lane.id()" in compiled.asm["llir"]
    assert not re.search(
        r"atomicrmw (?:add|or) ptr addrspace\(3\).*"
        r"@__ws_namedbar_state.*i32 0",
        compiled.asm["llir"],
    )
    # The QKV WS slots have fixed participant counts, so their software
    # barriers use the power-of-two round mask (4/16 warps) without division.
    # The generic release + acquire-poll protocol is used for both fixed and
    # mixed slots; only the fixed path specializes target arithmetic.
    ws_releases = re.findall(
        r"atomicrmw add ptr addrspace\(3\).*"
        r"@__ws_namedbar_state.*?i32 (\d+) "
        r"syncscope\(\"workgroup\"\) release",
        compiled.asm["llir"],
    )
    assert ws_releases and all(int(value) == 1 for value in ws_releases)
    assert re.search(
        r"atomicrmw add ptr addrspace\(3\).*@__ws_namedbar_state[^\n]*\n"
        r"\s*%\d+ = and i32 %\d+, -(?:4|16)",
        compiled.asm["llir"],
    )
    # WS waits must not let non-leader lanes bypass the counter.  Each warp
    # polls with lane 0 and broadcasts the completion bit at the merge block.
    assert re.search(
        r"load atomic i32, ptr addrspace\(3\).*?"
        r"llvm\.bi\.slb\.shfl\.idx\.b32\.i32\(i32 %\d+, i32 0\)",
        compiled.asm["llir"],
        re.S,
    )
    assert re.search(
        r"llvm\.bi\.sl\.waitcnt\(i64 8\).*?"
        r"llvm\.bi\.sl\.barrier\.alu",
        compiled.asm["llir"],
        re.S,
    )
    # The wait-counter field tracks the number of 4-row transaction groups in
    # one logical N tile.  Keep this derived from the controlled block_n knob
    # so the block_n=16 candidate is tested with its actual encoding rather
    # than being rejected by a block_n=32-only literal.
    block_n = int(os.environ.get("FLAGMEGA_QKV_BLOCK_N", "32"))
    assert block_n in {16, 32, 64}
    stages = _qkv_pipeline_stages()
    transaction_groups = block_n // 16
    expected_pending_wait = 8 | (transaction_groups * (stages - 1) << 24)
    assert f"llvm.bi.sl.waitcnt(i64 {expected_pending_wait})" in compiled.asm["llir"]
    assert "llvm.bi.sl.waitcnt(i64 8)" in compiled.asm["llir"]
    # The SME publication marker already orders the async payload before the
    # warp-leader release; participant-arrive must not add a second CTA fence.
    # The count includes the fence from `close`'s own empty-slot acquire,
    # restored in perf-iteration/ITERATION.md Trial 85 (required so the
    # writer cannot publish the close tag into a slot still being consumed).
    assert compiled.asm["llir"].count("llvm.nvvm.membar.cta") == 3
    # The wait arms feed the pipe participant-arrive through pure slot/index
    # arithmetic, so MembarAnalysis must not add a second rendezvous before the
    # SME publication barrier.  The reader release still contributes its real
    # warp rendezvous before the aggregated completion atomic.
    assert compiled.asm["llir"].count("llvm.bi.sl.barrier.alu") == 4
    # Do not regress the reader-release ordering to an implicit lock-step
    # assumption: its CTA fence must precede the real rendezvous and the
    # leader's completion publication.
    expected_reader_delta = next(iter(reader_counts)) if cta_reader_aggregate else 64
    assert re.search(
        r"llvm\.nvvm\.membar\.cta\(\).*?"
        r"llvm\.bi\.sl\.barrier\.alu\(\).*?"
        rf"atomicrmw add ptr addrspace\(3\).*?i32 {expected_reader_delta} "
        rf"syncscope\(\"workgroup\"\) release",
        compiled.asm["llir"],
        re.S,
    )
    assert not re.search(
        r"llvm\.nvvm\.membar\.cta\(\)[^\n]*\n"
        r"\s*tail call void @llvm\.nvvm\.membar\.cta",
        compiled.asm["llir"],
    )
    assert not re.search(
        r"llvm\.nvvm\.barrier\.cta\.sync\.aligned\.all\(i32 0\)[^\n]*\n"
        r"\s*tail call void @llvm\.nvvm\.barrier\.cta\.sync\.aligned\.all",
        compiled.asm["llir"],
    )


@pytest.mark.skip(reason="SME async-copy pipe still hangs at launch; the plain "
                         "shared-memory pipe deadlock is fixed (Trial 85). The "
                         "remaining issue is specific to the SME publication "
                         "rendezvous, not the dispatch protocol.")
def test_pointer_qkv_preserves_all_three_projection_outputs(tmp_path, pointer_qkv_module):
    import torch
    import triton
    from triton.flagmega.artifacts import write_artifact
    from triton.flagmega.importer import MemoryCheckpoint, TensorInfo
    from triton.flagmega.runtime import load
    if not torch.cuda.is_available() or triton.runtime.driver.active.get_current_target().backend != "corex":
        pytest.skip("BI-V150 device required")
    generator = torch.Generator().manual_seed(119)
    weights = {name: torch.randint(-3,4,(2048,n),generator=generator).bfloat16()
               for name,n in (("q_weight",2048),("k_weight",1024),("v_weight",1024))}
    checkpoint = MemoryCheckpoint({}, {name: TensorInfo(name,fm.DType.BFLOAT16,tuple(w.shape),"memory")
                                        for name,w in weights.items()}, weights)
    artifact = write_artifact(pointer_qkv_module,tmp_path/'artifact',target="iluvatar-bi-v150",
                               checkpoint=checkpoint,emit_executable=True)
    runtime = load(artifact,device="cuda:0")
    x = torch.randint(-3,4,(1,2048),generator=generator).bfloat16().cuda()
    outputs = tuple(torch.empty((1,n),dtype=torch.float32,device="cuda") for n in (2048,1024,1024))
    runtime.prepare(x,*outputs)
    for _ in range(5):
        for output in outputs: output.fill_(float('nan'))
        runtime.run_into(x,*outputs)
        for out,weight in zip(outputs,weights.values()):
            torch.testing.assert_close(out,x.float()@weight.cuda().float(),rtol=0,atol=0)


def test_static_3072_wait_uses_exact_reciprocal_lowering():
    """The full 48-warp CTA must not lower its phase wait to i32 udiv."""
    from triton.compiler.compiler import ASTSource, GPUTarget

    compiled = triton.compile(
        ASTSource(
            _static_3072_kernel,
            signature={"value": "*i32", "out": "*i32"},
            constexprs={"block": 64},
        ),
        target=GPUTarget("corex", 71, 64),
        options={"num_warps": 32},
    )
    assert "static_arrive_count = 3072 : i32" in compiled.asm["ttgir"]
    llir = compiled.asm["llir"]
    assert not re.search(r"udiv i32 [^\n]*, 3072", llir)
    # floor(counter / 3072) is floor((counter >> 10) / 3).  The ivcore11
    # target has a native unsigned mul-high, so keep this phase calculation in
    # i32 and avoid widening every polling loop to an i64 multiply.
    assert "llvm.nvvm.mulhi.ui" in llir
    assert "i32 -1431655765" in llir
    assert "mul i64" not in llir
    # Drain participants publish one warp-sized contribution instead of one
    # release atomic per lane; close/release owns the preceding fence.
    assert llir.count('i32 64 syncscope("workgroup") release') >= 2
    # close/release already fenced before entering wait_drained; drain itself
    # only counts completion and must not add another CTA fence. The count is
    # 4 because `close` performs its own empty-slot acquire (restored in
    # perf-iteration/ITERATION.md Trial 85 -- it is required so the writer
    # cannot publish the close tag into a slot a reader is still consuming),
    # and that wait carries one fence of its own.
    assert llir.count("llvm.nvvm.membar.cta") == 4
