# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
import triton.language.extra.cann.extension as al
import pytest


def test_extension_reexports_affine_bindings():
    assert al.affine_map is ascend_ir.affine_map
    assert al.affine_expr is ascend_ir.affine_expr
    assert al.affine_constant_expr is ascend_ir.affine_constant_expr
    assert al.affine_dim_expr is ascend_ir.affine_dim_expr
    assert al.affine_symbol_expr is ascend_ir.affine_symbol_expr
    assert al.affine_binary_op_expr is ascend_ir.affine_binary_op_expr
    assert al.AffineMap is al.affine_map
    assert al.AffineExpr is al.affine_expr
    assert al.AffineConstantExpr is al.affine_constant_expr
    assert al.AffineDimExpr is al.affine_dim_expr
    assert al.AffineSymbolExpr is al.affine_symbol_expr
    assert al.AffineBinaryOpExpr is al.affine_binary_op_expr


def test_make_affine_map():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)

        d0 = ascend_ir.affine_expr.get_dim(0)
        d1 = ascend_ir.affine_expr.get_dim(1)
        c2 = ascend_ir.affine_expr.get_constant(2)

        expr = (d0 + c2) * d1
        assert "d0" in str(expr) and "d1" in str(expr)
        assert not expr.is_pure_affine()
        assert hash(expr) == hash(expr)
        assert d0 == ascend_ir.affine_expr.get_dim(0)
        assert c2 == ascend_ir.affine_expr.get_constant(2)
        assert isinstance(c2, ascend_ir.affine_expr)
        assert isinstance(d0, ascend_ir.affine_expr)

        identity_map = ascend_ir.affine_map.get_identity(2)
        transpose_map = ascend_ir.affine_map.get(2, 0, [1, 0])
        transpose_map_by_expr = ascend_ir.affine_map.get(2, 0, [d1, d0])
        sum_map = ascend_ir.affine_map.get(2, 0, [d0 + d1, d1])
        const_map = ascend_ir.affine_map.get_constant(7)
        minor_identity_map = ascend_ir.affine_map.get_minor_identity(3, 2)

        assert identity_map.is_identity()
        assert identity_map.is_permutation()
        assert identity_map.get_num_dims() == 2
        assert identity_map.get_num_symbols() == 0
        assert identity_map.get_num_results() == 2
        assert str(identity_map) == "(d0, d1) -> (d0, d1)"

        assert not transpose_map.is_identity()
        assert transpose_map.is_permutation()
        assert str(transpose_map) == "(d0, d1) -> (d1, d0)"
        assert str(transpose_map_by_expr) == "(d0, d1) -> (d1, d0)"
        assert str(sum_map) == "(d0, d1) -> (d0 + d1, d1)"
        assert transpose_map.to_dict() == {
            "num_dims": 2,
            "num_symbols": 0,
            "results": [1, 0],
        }
        assert str(sum_map.get_sub_map([1])) == "(d0, d1) -> (d1)"
        assert str(sum_map.compose(transpose_map)) == "(d0, d1) -> (d0 + d1, d0)"
        assert str(transpose_map.inverse_permutation()) == "(d0, d1) -> (d1, d0)"
        assert transpose_map == transpose_map_by_expr
        assert hash(transpose_map) == hash(transpose_map)
        assert [str(r) for r in sum_map.get_results()] == ["d0 + d1", "d1"]
        assert const_map.is_single_constant()
        assert const_map.get_constant_result() == 7
        assert str(minor_identity_map) == "(d0, d1, d2) -> (d1, d2)"


def test_build_buffer_type_with_affine_map():
    with ir.context() as ctx:
        ir.load_dialects(ctx)
        ascend_ir.load_dialects(ctx)
        builder = ascend_ir.ascendnpu_ir_builder(ctx)

        transpose_map = ascend_ir.affine_map.get(2, 0, [1, 0])
        ub_attr = builder.get_target_attribute(ascend_ir.AddressSpace.UB)

        buffer_ty = builder.get_buffer_ty_with_affine_map([8, 16], builder.get_float_ty(), transpose_map, ub_attr)

        assert "memref<8x16xf32" in str(buffer_ty)
        assert "affine_map<(d0, d1) -> (d1, d0)>" in str(buffer_ty)
        assert "ub" in str(buffer_ty)


if __name__ == '__main__':
    test_build_buffer_type_with_affine_map()
    test_extension_reexports_affine_bindings()
    test_make_affine_map()
