"""Explicit MP31 collective TME issue; setup stays single-threaded."""
import re
import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _copy(desc, out):
    buf = tle.gpu.alloc((2, 32, 32), tl.float16)
    pipe = tle.pipe(capacity=2, tile=buf)
    writer = pipe.writer()
    slot = writer.acquire(0)
    tle.gpu.copy(desc, slot.tile, (32, 32), (0, 0))
    writer.commit(0)
    reader = pipe.reader()
    ready = reader.wait(0)
    values = tl.load(tle.gpu.local_ptr(ready.slot.tile))
    i = tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(out + i, values)
    reader.release(0)


@pytest.mark.parametrize("warps", [4, 8, 16])
@pytest.mark.parametrize("collective", [False, True])
def test_collective_predicate_and_single_thread_initialization(collective, warps):
    k = compile_musa(_copy, {"desc": "tensordesc<fp16[32,32]>", "out": "*fp16"},
                     options={"num_warps": warps, "tme_issue_warp": collective})
    ir = k.asm["llir"]
    tid = re.search(r"(%[\w.]+) = .*call i32 @llvm.musa.read.ptx.sreg.tid.x", ir)[1]
    assert re.search(r"icmp eq i32 " + re.escape(tid) + r", 0", ir)
    warp = re.search(r"icmp (?:samesign )?ult i32 " + re.escape(tid) + r", 32", ir)
    assert bool(warp) == collective
    assert "llvm.musa.tme.ld.tile.2d" in ir
    assert k.metadata.tme_issue_warp == collective


def test_collective_option_validation_and_cache():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).tme_issue_warp
    assert backend.parse_options({}).hash() != backend.parse_options({"tme_issue_warp": True}).hash()
    with pytest.raises(ValueError, match="tme_issue_warp must be a bool"):
        backend.parse_options({"tme_issue_warp": 1})
    with pytest.raises(ValueError, match="only on MP31"):
        backend.parse_options({"tme_issue_warp": True, "arch": "22"})


@pytest.mark.parametrize("issuer", [0, 1, 32, 33, 96])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("operation", ["init", "add", "arrive"])
def test_explicit_issue_domain(tmp_path, issuer, enabled, operation):
    """Only aligned transaction issuers expand; init and unaligned IDs do not."""
    from triton._C import libtriton
    from triton._C.libtriton import ir, passes

    attr = f"{{musa.tme.issue_thread = {issuer} : i32}}"
    operations = {
        "init": f"ttmg.init_arrival %one, %one, %zero {attr} : i32",
        "add": f"ttmg.barrier_add_trans %one, %bytes, %true {attr} : i32",
        "arrive": f"ttmg.arrive_barrier_noret %one, %true {attr} : i32",
    }
    path = tmp_path / "issuer.ttgir"
    path.write_text(f'''module attributes {{"ttg.num-ctas" = 1 : i32,
      "ttg.num-warps" = 4 : i32, ttg.target = "musa:ph1",
      "ttg.threads-per-warp" = 32 : i32, musa.tme_issue_warp = {int(enabled)} : i32}} {{
      tt.func public @probe() attributes {{musa.max_bar_id = 1 : i32}} {{
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %bytes = arith.constant 256 : i32
        %true = arith.constant true
        {operations[operation]}
        tt.return
      }}
    }}''')
    _, backend = mthreads_backend()
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    passes.convert.add_scf_to_cf(pm)
    passes.convert.add_index_to_llvmir(pm)
    libtriton.mthreads.passes.ttgpuir.add_allocate_shared_memory(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_mtgpu_to_llvm(pm, 31)
    libtriton.mthreads.passes.ttgpuir.add_to_llvmir(pm, 31)
    passes.convert.add_cf_to_llvmir(pm)
    passes.convert.add_arith_to_llvmir(pm)
    pm.run(module, "test_explicit_tme_issue_domain")
    text = module.str_nodebug()
    # Inspect pre-optimization LLVM dialect so the mask has not become an ult.
    assert ("llvm.and" in text and "-32" in text) == (
        enabled and operation != "init" and issuer % 32 == 0), text
