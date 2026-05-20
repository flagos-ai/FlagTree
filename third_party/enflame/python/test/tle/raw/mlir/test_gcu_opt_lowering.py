"""
Test: use_gcu_opt=True path - verify gcu-compiler-opt can lower EDSL output.

Uses gcu.ptr2memref + vector.maskedload/maskedstore (native GCU IR path).
"""
from typing_extensions import Literal as L

from mlir import ir
from mlir.dialects import arith, gpu
from triton.experimental.tle.raw import dialect, Input
from triton.experimental.tle.raw.tops.gcu_dialect import gcu


@dialect(name="tops_mlir", use_gcu_opt=True)
def edsl_simple(
    output: Input[L["!gcu.ptr<f32>"]],
    x: Input[L["!gcu.ptr<f32>"]],
    y: Input[L["!gcu.ptr<f32>"]],
    n_elements: Input[L["i32"]],
):
    i32 = ir.IntegerType.get_signless(32)
    idx_ty = ir.IndexType.get()
    f32 = ir.F32Type.get()
    vec128xf32 = ir.VectorType.get([128], f32)
    vec128xi1 = ir.VectorType.get([128], ir.IntegerType.get_signless(1))
    dyn = ir.ShapedType.get_dynamic_size()
    memref_f32 = ir.MemRefType.get([dyn], f32)

    zeros = arith.constant(
        vec128xf32,
        ir.DenseElementsAttr.get_splat(vec128xf32, ir.FloatAttr.get(f32, 0.0)))
    c128 = arith.constant(i32, ir.IntegerAttr.get(i32, 128))

    block_id = gpu.block_id(gpu.Dimension.x)
    block_id_i32 = arith.index_cast(i32, block_id)
    global_offset = arith.muli(block_id_i32, c128)

    # gcu.ptr2memref: !gcu.ptr<f32> → memref<?xf32>
    x_memref = gcu.ptr2memref(x, memref_f32)
    y_memref = gcu.ptr2memref(y, memref_f32)
    out_memref = gcu.ptr2memref(output, memref_f32)

    offset_idx = arith.index_cast(idx_ty, global_offset)
    mask_full = gcu.constant_mask([128], vec128xi1)
    x_vec = gcu.maskedload(x_memref, offset_idx, mask_full, zeros, vec128xf32)
    y_vec = gcu.maskedload(y_memref, offset_idx, mask_full, zeros, vec128xf32)
    out_vec = arith.addf(x_vec, y_vec)
    gcu.maskedstore(out_memref, offset_idx, mask_full, out_vec)


if __name__ == "__main__":
    print("=== Testing use_gcu_opt=True (gcu-compiler-opt lowering) ===")
    print()

    print("--- Step 1: EDSL MLIR module (before lowering) ---")
    print(edsl_simple.mlir_module)
    print()

    print("--- Step 2: Calling gcu-compiler-opt for LLVM lowering ---")
    try:
        llvm_ir = edsl_simple.make_llvm()
        print("SUCCESS! LLVM IR output (first 60 lines):")
        lines = llvm_ir.splitlines()
        for line in lines[:60]:
            print(line)
        if len(lines) > 60:
            print(f"... ({len(lines) - 60} more lines)")
        print()
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        raise
