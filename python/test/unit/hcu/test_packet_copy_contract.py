"""Packet copy arity is shared by the Gluon frontend and LLVM preflight."""
import pytest

from triton._C import libtriton
from triton.backends.compiler import GPUTarget

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler bindings are required", allow_module_level=True)

compiler = pytest.importorskip("triton.backends.hcu.compiler")


def packet_copy_module(index_count):
    indices = ["%zero"] * index_count
    if index_count == 3:
        indices[2] = "%one"
    indices = ", ".join(indices)
    return '''#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#packet = #ttg.mls_shared<{opIdx = 0, mlsTile = [64, 32], elemBitWidth = 16, alt2Kind = 0, version = 257, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {ttg.hcu.whole_packet_global_bounds = 1 : i32, ttg.hcu.whole_packet_v1 = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx936"} {
  tt.func public @copy(%src: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %rows = arith.constant 64 : i32
    %columns = arith.constant 256 : i32
    %stride = arith.constant 256 : i64
    %unit = arith.constant 1 : i64
    %owner = ttg.local_alloc {alignment = 16384 : i32} : () -> !ttg.memdesc<1x8192xbf16, #shared, #smem, mutable>
    %frame = ttg.memdesc_index %owner[%zero] : !ttg.memdesc<1x8192xbf16, #shared, #smem, mutable> -> !ttg.memdesc<8192xbf16, #shared, #smem, mutable>
    %slice = ttg.memdesc_subslice %frame[4096] : !ttg.memdesc<8192xbf16, #shared, #smem, mutable> -> !ttg.memdesc<2048xbf16, #shared, #smem, mutable, 8192>
    %view = ttg.memdesc_reinterpret %slice : !ttg.memdesc<2048xbf16, #shared, #smem, mutable, 8192> -> !ttg.memdesc<64x32xbf16, #packet, #smem, mutable>
    %copy = hcug.matrix_load_to_local %src, %view, shape = [%rows, %columns], strides = [%stride, %unit], tensorShape = [64, 32], indices = [INDICES] {hcu.packet.source_basis = array<i32: 1, 2, 4, 8, 16, 32, 256, 512, 1024, 64, 128, 0>} : <bf16> -> <64x32xbf16, #packet, #smem, mutable>
    %commit = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %commit {num = 0 : i32}
    ttg.local_dealloc %owner : !ttg.memdesc<1x8192xbf16, #shared, #smem, mutable>
    tt.return
  }
}
'''.replace("INDICES", indices)


def lower_packet_copy(tmp_path, index_count, columns=256):
    source = tmp_path / f"packet-copy-{index_count}.mlir"
    text = packet_copy_module(index_count)
    text = text.replace("%columns = arith.constant 256 : i32",
                        f"%columns = arith.constant {columns} : i32")
    text = text.replace("%stride = arith.constant 256 : i64",
                        f"%stride = arith.constant {columns} : i64")
    source.write_text(text)
    context = libtriton.ir.context()
    libtriton.ir.load_dialects(context)
    compiler.HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
    module = libtriton.ir.parse_mlir_module(str(source), context)
    pm = libtriton.ir.pass_manager(context)
    libtriton.hcu.passes.ttgpuir.add_update_async_wait_count(pm, "gfx936")
    libtriton.hcu.passes.ttgpuir.add_allocate_shared_memory(pm)
    libtriton.hcu.passes.ttgpuir.add_to_llvmir(pm, "gfx936", False)
    pm.run(module, "packet_copy_contract")
    return str(module)


def test_packet_copy_three_indices(tmp_path):
    assert "llvm.func" in lower_packet_copy(tmp_path, 3)


@pytest.mark.parametrize("columns", [272, 768, 1280, 3072, 4096, 6144, 7936, 8176])
def test_packet_copy_structured_stride_field(tmp_path, columns):
    assert "llvm.func" in lower_packet_copy(tmp_path, 3, columns)


@pytest.mark.parametrize("columns", [0, 16, 255, 257, 8192, 8193, 1 << 31])
def test_packet_copy_rejects_invalid_structured_stride(tmp_path, columns, capfd):
    with pytest.raises(RuntimeError):
        lower_packet_copy(tmp_path, 3, columns)
    assert "wholepacket-v1" in capfd.readouterr().err


@pytest.mark.parametrize("index_count", [1, 2, 4])
def test_packet_copy_rejects_other_arities(tmp_path, index_count, capfd):
    with pytest.raises(RuntimeError):
        lower_packet_copy(tmp_path, index_count)
    assert "packet copy requires rank-two shape/strides and row, column, enable indices" in capfd.readouterr().err
