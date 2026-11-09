#!/usr/bin/env python3

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


def main():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)

        identity_map = ascend_ir.affine_map.get_identity(2)
        transpose_map = ascend_ir.affine_map.get(2, 0, [1, 0])

        print("identity map:", identity_map)
        print("  dims:", identity_map.get_num_dims())
        print("  symbols:", identity_map.get_num_symbols())
        print("  results:", identity_map.get_num_results())
        print("  is_identity:", identity_map.is_identity())
        print("  is_permutation:", identity_map.is_permutation())

        print("transpose map:", transpose_map)
        print("  is_identity:", transpose_map.is_identity())
        print("  is_permutation:", transpose_map.is_permutation())
        print("  as python object:", transpose_map.to_dict())


if __name__ == "__main__":
    main()
