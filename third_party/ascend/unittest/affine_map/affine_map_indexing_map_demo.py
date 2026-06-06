#!/usr/bin/env python3

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


def main():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)

        builder = ascend_ir.ascendnpu_ir_builder(ctx)

        d0 = ascend_ir.affine_expr.get_dim(0)
        d1 = ascend_ir.affine_expr.get_dim(1)
        c8 = ascend_ir.affine_expr.get_constant(8)

        # Example indexing maps: transpose and a tiled/reduced projection.
        map_in0 = ascend_ir.affine_map.get(2, 0, [d1, d0])
        map_in1 = ascend_ir.affine_map.get(2, 0, [d0, d1])
        map_out = ascend_ir.affine_map.get(2, 0, [d0.floordiv(c8), d1.mod(c8)])

        indexing_map_attr = builder.get_affine_map_array_attr([map_in0, map_in1, map_out])
        print("indexing_map attr:", indexing_map_attr)

        ub_space = builder.get_target_attribute(ascend_ir.AddressSpace.UB)
        f32 = builder.get_float_ty()
        memref_ty = builder.get_buffer_ty_with_affine_map([16, 32], f32, map_in0, ub_space)
        print("buffer type with map_in0:", memref_ty)


if __name__ == "__main__":
    main()
