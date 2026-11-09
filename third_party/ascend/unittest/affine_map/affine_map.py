#!/usr/bin/env python3

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


def main():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)

        d0 = ascend_ir.affine_expr.get_dim(0)
        d1 = ascend_ir.affine_expr.get_dim(1)
        c2 = ascend_ir.affine_expr.get_constant(2)

        expr = (d0 + c2) * d1
        print("expr:", expr)
        print("expr pure affine:", expr.is_pure_affine())
        print("expr hashable:", hash(expr))

        m0 = ascend_ir.affine_map.get_identity(2)
        m1 = ascend_ir.affine_map.get(2, 0, [d1, d0])
        m2 = ascend_ir.affine_map.get(2, 0, [d0 + d1, d1])
        m3 = ascend_ir.affine_map.get_constant(7)
        minor = ascend_ir.affine_map.get_minor_identity(3, 2)

        print("m0:", m0)
        print("m1:", m1)
        print("m2:", m2)
        print("m1 inverse:", m1.inverse_permutation())
        print("m2 submap[1]:", m2.get_sub_map([1]))
        print("m2 compose m1:", m2.compose(m1))
        print("m1 as dict:", m1.to_dict())
        print("m3 constant:", m3, "value=", m3.get_constant_result())
        print("minor identity:", minor)
        print("m2 results:", [str(x) for x in m2.get_results()])


if __name__ == "__main__":
    main()
