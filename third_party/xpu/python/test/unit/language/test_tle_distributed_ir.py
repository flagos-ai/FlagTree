from pathlib import Path

import pytest
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

from triton._C.libtriton import ir, passes, xpu
from triton.backends.compiler import GPUTarget
from triton.backends.xpu.compiler import XPUBackend
from triton.compiler.compiler import ASTSource
from triton.runtime._distributed import DistributedRtContext

MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit
def _rank(out, comm, mesh: tl.constexpr):
    tl.store(out, tle.shard_id(mesh, "device", device_dptr=comm))


@triton.jit
def _size(out, comm):
    tl.store(out, tle.n_pes(comm))


@triton.jit
def _barrier(out, comm, mesh: tl.constexpr):
    tle.distributed_barrier(mesh, device_dptr=comm, space="device", barrier_kind="sync")
    tl.store(out, 1)


@triton.jit
def _remote(out, mem):
    ptr = tle.remote(mem, space="device", dtype=tl.float32, shard_id=1, offset=0)
    tl.store(out, tl.load(ptr))


@triton.jit
def _implicit_rank(out):
    tl.store(out, tle._get_local_rank(None))


def _ir_text(module, path: Path):
    text = str(module)
    if text:
        path.write_text(text)
        return text
    module.create_location_snapshot(str(path))
    return path.read_text()


@pytest.mark.parametrize(
    "fn,signature,constexprs,op,symbol,lite",
    [
        (_rank, {"out": "*i32", "comm": "i64", "mesh": "constexpr"
                 }, {"mesh": MESH}, "get_device_id", "flagcxDevCommGetIntraRank", False),
        (_size, {"out": "*i32", "comm": "i64"}, {}, "get_num_pes", "flagcxDevCommGetIntraSize", False),
        (_barrier, {"out": "*i32", "comm": "i64", "mesh": "constexpr"
                    }, {"mesh": MESH}, "distributed_barrier", "flagcxIntraBarrierSyncS", False),
        (_remote, {"out": "*fp32", "mem": "i64"}, {}, "remote_pointer", "flagcxGetIntraPointerC", False),
        (_implicit_rank, {"out": "*i32"}, {}, "get_device_id", "flagcxDevCommGetIntraRank", True),
    ],
    ids=["rank", "size", "barrier", "remote", "implicit-rank"],
)
def test_tle_flagcx_lowers_without_device(tmp_path, monkeypatch, fn, signature, constexprs, op, symbol, lite):
    monkeypatch.setenv("FLAGTREE_LITE_DIST", "1" if lite else "0")
    monkeypatch.setattr(DistributedRtContext, "_init_count", 1 if lite else 0)

    target = GPUTarget("xpu", 3, 1)
    backend = XPUBackend(target)
    source = ASTSource(fn, signature=signature, constexprs=constexprs)
    options = backend.parse_options(source.parse_options())
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    metadata = {"target": target, **options.__dict__}

    module = backend.make_ttir(module, metadata, options)
    ttir = _ir_text(module, tmp_path / "kernel.ttir")
    assert f"triton_xpu.{op}" in ttir

    module = backend.make_ttxir(module, metadata, options)
    ttxir = _ir_text(module, tmp_path / "kernel.ttxir")
    assert f"triton_xpu.{op}" in ttxir
    if lite:
        assert 'tt.name = "device_comm_ptr"' in ttxir
        assert 'tt.name = "device_mem_ptr"' in ttxir

    pm = ir.pass_manager(context)
    passes.convert.add_scf_to_cf(pm)
    xpu.passes.ttxpuir.add_allocate_xpu_shared_memory(pm)
    xpu.passes.ttxpuir.add_convert_tritonxpu_to_llvm_pass(pm, 3, 512, False)
    pm.run(module, "lower")
    llvm_dialect = _ir_text(module, tmp_path / "kernel.llvm.mlir")
    assert f"llvm.call @{symbol}" in llvm_dialect
