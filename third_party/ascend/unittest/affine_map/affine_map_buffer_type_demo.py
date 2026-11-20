#!/usr/bin/env python3

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


def main():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)

        builder = ascend_ir.ascendnpu_ir_builder(ctx)
        f32 = builder.get_float_ty()
        ub_space = builder.get_target_attribute(ascend_ir.AddressSpace.UB)

        # Build a memref type using an explicit affine map layout.
        transpose_map = ascend_ir.affine_map.get(2, 0, [1, 0])
        memref_ty = builder.get_buffer_ty_with_affine_map([8, 16], f32, transpose_map, ub_space)
        map_attr = builder.get_affine_map_attr(transpose_map)

        print("affine map:", transpose_map)
        print("affine map attr:", map_attr)
        print("memref type:", memref_ty)


if __name__ == "__main__":
    main()
