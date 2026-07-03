import triton
import triton.experimental.tle as tle
import triton.language as tl

from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, tle as tle_ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    tle_ir.load_dialects(context)
    tle_ir.load_tile_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    if not module.verify():
        raise RuntimeError(f"{kernel.__name__}: module.verify() failed")
    return str(module)


@triton.jit
def bind_buffer(XBLOCK: tl.constexpr, N: tl.constexpr):
    # buffer is allocated outside the for loop and converted to a tensor inside the loop: entering the loop body triggers scope cloning,
    # After cloning, the buffer must remain a buffer and not a tensor (buffer_type._unflatten_ir)
    # tle.dsa.ascend.UB is triton.language.extra.extension.cann.core.ascend_address_space.UB
    buffer1 = tle.dsa.alloc(shape=[XBLOCK], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    acc = tl.zeros([XBLOCK], dtype=tl.float32)
    for i in tl.range(0, N):
        acc = acc + tle.dsa.to_tensor(buffer1, writable=True)
    return acc


@triton.jit
def copy_buffer(x):
    offsets = tl.arange(0, 32)
    buffer1 = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(tl.load(x + offsets), buffer1, [32])
    tle.dsa.to_tensor(buffer1, writable=False)


@triton.jit
def subview_buffer():
    buffer1 = tle.dsa.alloc(shape=[64, 32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    sub = tle.dsa.subview(buffer1, [0, 0], [32, 32], [1, 1])
    tle.dsa.to_tensor(sub, writable=False)


@triton.jit
def to_buffer_tensor(x):
    offsets = tl.arange(0, 32)
    values = tl.load(x + offsets)
    buffer1 = tle.dsa.to_buffer(values, tle.dsa.ascend.UB)
    tle.dsa.copy(buffer1, x + offsets, [32])


@triton.jit
def add_buffer(x, y, out):
    offsets = tl.arange(0, 32)
    a_ub = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(x + offsets, a_ub, [32])
    tle.dsa.copy(y + offsets, b_ub, [32])
    tle.dsa.add(a_ub, b_ub, c_ub)
    tle.dsa.copy(c_ub, out + offsets, [32])


@triton.jit
def binary_buffer(x, y, out, OP_ID: tl.constexpr):
    offsets = tl.arange(0, 32)
    a_ub = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    b_ub = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    c_ub = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.copy(x + offsets, a_ub, [32])
    tle.dsa.copy(y + offsets, b_ub, [32])
    if OP_ID == 0:
        tle.dsa.add(a_ub, b_ub, c_ub)
    elif OP_ID == 1:
        tle.dsa.sub(a_ub, b_ub, c_ub)
    elif OP_ID == 2:
        tle.dsa.mul(a_ub, b_ub, c_ub)
    elif OP_ID == 3:
        tle.dsa.div(a_ub, b_ub, c_ub)
    elif OP_ID == 4:
        tle.dsa.max(a_ub, b_ub, c_ub)
    elif OP_ID == 5:
        tle.dsa.min(a_ub, b_ub, c_ub)
    tle.dsa.copy(c_ub, out + offsets, [32])


@triton.jit
def extract_slice_tensor(x, out):
    offsets = tl.arange(0, 32)
    values = tl.load(x + offsets)
    one = tle.dsa.extract_slice(values, (0,), (1,), (1,))
    tl.store(out + tl.arange(0, 1), one)


@triton.jit
def insert_slice_tensor(x, out):
    offsets = tl.arange(0, 32)
    values = tl.load(x + offsets)
    one = tl.load(x + tl.arange(0, 1))
    merged = tle.dsa.insert_slice(values, one, (0,), (1,), (1,))
    tl.store(out + offsets, merged)


@triton.jit
def extract_element_tensor(x, out):
    offsets = tl.arange(0, 32)
    values = tl.load(x + offsets)
    first = tle.dsa.extract_element(values, (0,))
    tl.store(out, first)


@triton.jit
def top_level_tile_dsl(x):
    offsets = tl.arange(0, 32)
    buf = tle.dsa.tile_alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    tle.dsa.tile_copy(tl.load(x + offsets), buf, [32])
    tle.dsa.tile_to_tensor(buf, writable=False)
    tle.dsa.tile_pipe_barrier(1)


@triton.jit
def tle_scope_region(x):
    offsets = tl.arange(0, 32)
    with tle.scope(core_mode="cube"):
        buf = tle.dsa.alloc(shape=[32], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
        tle.dsa.copy(tl.load(x + offsets), buf, [32])
        tle.dsa.to_tensor(buf, writable=False)


@triton.jit
def subview_constexpr_buffer(SIZE: tl.constexpr):
    buf = tle.dsa.alloc(shape=[64, 64], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    sub = tle.dsa.subview(buf, [0, 0], [SIZE, SIZE], [1, 1])
    tle.dsa.to_tensor(sub, writable=False)


@triton.jit
def method_subview_multibuffer():
    slot_id = tl.program_id(0)
    buf = tle.dsa.alloc(shape=[2, 64, 128], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    slot = buf.subview([slot_id, 0, 0], [1, 64, 128], [1, 1, 1])
    slot.to_tensor(writable=False)


@triton.jit
def method_subview_after_to_buffer(x):
    offsets = tl.arange(0, 32)
    values = tl.load(x + offsets)
    buf = tle.dsa.to_buffer(values, tle.dsa.ascend.UB)
    slot = buf.subview([tl.program_id(0)], [1], [1])
    slot.to_tensor(writable=False)


@triton.jit
def cube_launch_wait(out):
    a = tle.dsa.tile_alloc(shape=[16, 16], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.L1)
    b = tle.dsa.tile_alloc(shape=[16, 16], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.L1)
    acc = tle.dsa.tile_alloc(shape=[16, 16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.L0C)
    stage_a = tle.dsa.tile_alloc(shape=[16, 16], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.L0A)
    stage_b = tle.dsa.tile_alloc(shape=[16, 16], dtype=tl.float16, mem_addr_space=tle.dsa.ascend.L0B)
    tle.dsa.tile_cube_launch(a, b, acc, stage_a, stage_b, out, transpose_b=True, init=True, mma="smoke")
    tle.dsa.tile_cube_wait()


@triton.jit
def tile_sync_flags():
    tle.dsa.tile_set_flag(3, 1, 0)
    tle.dsa.tile_wait_flag(3, 1, 0)


@triton.jit
def gm_offset_copy(x, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    stride = tl.full((), BLOCK, tl.int64)
    base = tle.dsa.tile_gm_offset(x, [pid], [stride])
    values = tl.load(base + offsets)
    tl.store(out + offsets, values)


def assert_no_legacy_dsa_memory_ir(mlir):
    assert "memref.alloc" not in mlir
    assert "bufferization.to_tensor" not in mlir
    assert "bufferization.to_memref" not in mlir
    assert "#hivm.address_space" not in mlir


def assert_public_dsa_uses_tileir(mlir):
    assert "tile.alloc" in mlir
    assert "tile.to_tensor" in mlir
    assert_no_legacy_dsa_memory_ir(mlir)


def test_bind_buffer_tileir():
    mlir = compile_kernel(bind_buffer, {}, {})
    assert_public_dsa_uses_tileir(mlir)


def test_copy_buffer_tileir():
    mlir = compile_kernel(copy_buffer, {"x": "*fp32"}, {})
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.copy" in mlir
    assert "!tile.buf" in mlir
    assert "tle.dsa_copy" not in mlir


def test_subview_buffer_tileir():
    mlir = compile_kernel(subview_buffer, {}, {})
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.subview" in mlir
    assert "tle.dsa_subview" not in mlir


def test_to_buffer_tileir():
    mlir = compile_kernel(to_buffer_tensor, {"x": "*fp32"}, {})
    assert "tile.alloc" in mlir
    assert "tile.store_tensor" in mlir
    assert "tile.copy" in mlir
    assert "!tile.buf" in mlir
    assert_no_legacy_dsa_memory_ir(mlir)
    assert "tle.dsa_to_buffer" not in mlir


def test_add_buffer_tileir():
    mlir = compile_kernel(add_buffer, {"x": "*fp32", "y": "*fp32", "out": "*fp32"}, {})
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.store_tensor" in mlir
    assert "arith.addf" in mlir
    assert "tle.dsa_add" not in mlir


def test_binary_buffer_tileir_ops():
    op_cases = [
        (0, "add", "arith.addf"),
        (1, "sub", "arith.subf"),
        (2, "mul", "arith.mulf"),
        (3, "div", "arith.divf"),
        (4, "max", "arith.maxnumf"),
        (5, "min", "arith.minnumf"),
    ]
    signature = {"x": "*fp32", "y": "*fp32", "out": "*fp32"}
    for op_id, op_name, arith_op in op_cases:
        mlir = compile_kernel(binary_buffer, signature, {"OP_ID": op_id})
        assert_public_dsa_uses_tileir(mlir)
        assert "tile.store_tensor" in mlir
        assert arith_op in mlir
        assert f"tle.dsa_{op_name}" not in mlir


def test_tensor_slice_ops_verify():
    signature = {"x": "*fp32", "out": "*fp32"}
    extract_mlir = compile_kernel(extract_slice_tensor, signature, {})
    insert_mlir = compile_kernel(insert_slice_tensor, signature, {})
    element_mlir = compile_kernel(extract_element_tensor, signature, {})
    assert "tensor.extract_slice" in extract_mlir
    assert "tensor.insert_slice" in insert_mlir
    assert "tensor.extract" in element_mlir
    assert_no_legacy_dsa_memory_ir(extract_mlir)
    assert_no_legacy_dsa_memory_ir(insert_mlir)
    assert_no_legacy_dsa_memory_ir(element_mlir)


def test_top_level_tile_dsl_exports():
    mlir = compile_kernel(top_level_tile_dsl, {"x": "*fp32"}, {})
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.copy" in mlir
    assert "tile.pipe_barrier" in mlir
    assert "tle.dsa_copy" not in mlir


def test_tle_scope_emits_scope_op():
    mlir = compile_kernel(tle_scope_region, {"x": "*fp32"}, {})
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.copy" in mlir
    assert "scope.scope" in mlir
    assert "tcore_type = #hivm.tcore_type<CUBE>" in mlir


def test_subview_constexpr_sizes_stay_ranked():
    mlir = compile_kernel(subview_constexpr_buffer, {}, {"SIZE": 32})
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.subview" in mlir
    assert "[[32, 32]]" in mlir
    assert "tensor<32x32xf32>" in mlir
    assert "tensor<32x32x32x32xf32>" not in mlir


def test_buffer_method_subview_multibuffer_preserves_slot_dim():
    mlir = compile_kernel(method_subview_multibuffer, {}, {})
    compact_mlir = mlir.replace(" ", "")
    assert_public_dsa_uses_tileir(mlir)
    assert "tile.subview" in mlir
    assert "[[1, 64, 128]]" in mlir
    assert "<[1,64,128],f32,ub>" in compact_mlir
    assert "tensor<1x64x128xf32>" in mlir
    assert "tle.dsa_subview" not in mlir


def test_buffer_method_subview_after_to_buffer_tensor():
    mlir = compile_kernel(method_subview_after_to_buffer, {"x": "*fp32"}, {})
    assert "tile.alloc" in mlir
    assert "tile.store_tensor" in mlir
    assert "tile.subview" in mlir
    assert "tensor<1xf32>" in mlir
    assert_no_legacy_dsa_memory_ir(mlir)


def test_tile_cube_launch_wait_smoke():
    mlir = compile_kernel(cube_launch_wait, {"out": "*fp32"}, {})
    assert "tile.cube_launch" in mlir
    assert "tile.cube_wait" in mlir
    assert "mma = \"smoke\"" in mlir
    assert_no_legacy_dsa_memory_ir(mlir)


def test_tile_sync_flags_smoke():
    mlir = compile_kernel(tile_sync_flags, {}, {})
    assert "tile.set_flag" in mlir
    assert "tile.wait_flag" in mlir
    assert_no_legacy_dsa_memory_ir(mlir)


def test_tile_gm_offset_smoke():
    mlir = compile_kernel(gm_offset_copy, {"x": "*fp32", "out": "*fp32"}, {"BLOCK": 32})
    assert "tile.gm_offset" in mlir
    assert "arith.index_cast" in mlir
    assert_no_legacy_dsa_memory_ir(mlir)


if __name__ == "__main__":
    print("=" * 60)
    mlir = compile_kernel(bind_buffer, {}, {"XBLOCK": 128, "N": 4})
    print(mlir)
    bind_mlir = compile_kernel(bind_buffer, {}, {})
    copy_mlir = compile_kernel(copy_buffer, {"x": "*fp32"}, {})
    subview_mlir = compile_kernel(subview_buffer, {}, {})
    to_buffer_mlir = compile_kernel(to_buffer_tensor, {"x": "*fp32"}, {})
    add_mlir = compile_kernel(add_buffer, {"x": "*fp32", "y": "*fp32", "out": "*fp32"}, {})
    extract_slice_mlir = compile_kernel(extract_slice_tensor, {"x": "*fp32", "out": "*fp32"}, {})
    insert_slice_mlir = compile_kernel(insert_slice_tensor, {"x": "*fp32", "out": "*fp32"}, {})
    extract_element_mlir = compile_kernel(extract_element_tensor, {"x": "*fp32", "out": "*fp32"}, {})
    top_level_tile_mlir = compile_kernel(top_level_tile_dsl, {"x": "*fp32"}, {})
    subview_constexpr_mlir = compile_kernel(subview_constexpr_buffer, {}, {"SIZE": 32})
    method_subview_mlir = compile_kernel(method_subview_multibuffer, {}, {})
    to_buffer_method_subview_mlir = compile_kernel(method_subview_after_to_buffer, {"x": "*fp32"}, {})
    cube_launch_mlir = compile_kernel(cube_launch_wait, {"out": "*fp32"}, {})
    sync_flags_mlir = compile_kernel(tile_sync_flags, {}, {})
    gm_offset_mlir = compile_kernel(gm_offset_copy, {"x": "*fp32", "out": "*fp32"}, {"BLOCK": 32})
    assert_public_dsa_uses_tileir(bind_mlir)
    assert_public_dsa_uses_tileir(copy_mlir)
    assert_public_dsa_uses_tileir(subview_mlir)
    assert "tile.copy" in copy_mlir
    assert "!tile.buf" in copy_mlir
    assert "tle.dsa_copy" not in copy_mlir
    assert "tile.subview" in subview_mlir
    assert "tle.dsa_subview" not in subview_mlir
    assert "tile.store_tensor" in to_buffer_mlir
    assert_no_legacy_dsa_memory_ir(to_buffer_mlir)
    assert "tile.store_tensor" in add_mlir
    assert "tle.dsa_add" not in add_mlir
    assert "tensor.extract_slice" in extract_slice_mlir
    assert "tensor.insert_slice" in insert_slice_mlir
    assert "tensor.extract" in extract_element_mlir
    assert "tile.pipe_barrier" in top_level_tile_mlir
    assert "tensor<32x32xf32>" in subview_constexpr_mlir
    assert "tensor<32x32x32x32xf32>" not in subview_constexpr_mlir
    assert "tile.subview" in method_subview_mlir
    assert "tensor<1x64x128xf32>" in method_subview_mlir
    assert "tile.store_tensor" in to_buffer_method_subview_mlir
    assert "tile.subview" in to_buffer_method_subview_mlir
    assert "tile.cube_launch" in cube_launch_mlir
    assert "tile.cube_wait" in cube_launch_mlir
    assert "tile.set_flag" in sync_flags_mlir
    assert "tile.wait_flag" in sync_flags_mlir
    assert "tile.gm_offset" in gm_offset_mlir
    print(bind_mlir)
    print(copy_mlir)
    print(subview_mlir)
    print(to_buffer_mlir)
    print(add_mlir)
    print(extract_slice_mlir)
    print(insert_slice_mlir)
    print(extract_element_mlir)
    print(top_level_tile_mlir)
    print(subview_constexpr_mlir)
    print(method_subview_mlir)
    print(to_buffer_method_subview_mlir)
    print(cube_launch_mlir)
    print(sync_flags_mlir)
    print(gm_offset_mlir)
