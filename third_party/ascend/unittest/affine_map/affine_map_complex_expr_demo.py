#!/usr/bin/env python3

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


def main():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)

        d0 = ascend_ir.affine_expr.get_dim(0)
        d1 = ascend_ir.affine_expr.get_dim(1)
        s0 = ascend_ir.affine_expr.get_symbol(0)
        c3 = ascend_ir.affine_expr.get_constant(3)
        c4 = ascend_ir.affine_expr.get_constant(4)

        # Complex expressions with symbols and integer arithmetic.
        tiled_row = (d0 + s0).floordiv(c4)
        tiled_col = (d1 + c3).ceildiv(c4)
        inner = (d0 + d1).mod(c4)

        map_a = ascend_ir.affine_map.get(2, 1, [tiled_row, tiled_col, inner])
        map_b = ascend_ir.affine_map.get(2, 0, [d1, d0])
        map_comp = map_a.compose(map_b)

        print("map_a:", map_a)
        print("map_b:", map_b)
        print("map_a composed with map_b:", map_comp)
        print("map_a results:", [str(r) for r in map_a.get_results()])
        print("map_a submap [0, 2]:", map_a.get_sub_map([0, 2]))
        print("map_a metadata:", map_a.to_dict())


if __name__ == "__main__":
    main()
