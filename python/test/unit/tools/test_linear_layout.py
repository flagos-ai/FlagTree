# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from triton.tools import LinearLayout


def test_identity_1d():
    layout = LinearLayout.identity_1d(8, "idx", "idx")
    for value in range(8):
        assert layout.apply({"idx": value})["idx"] == value
    assert layout.is_surjective()


def test_zeros_1d():
    layout = LinearLayout.zeros_1d(8, "idx", "zero")
    for value in range(8):
        assert layout.apply({"idx": value})["zero"] == 0
    assert layout.is_surjective()

    widened = LinearLayout.zeros_1d(8, "idx", "zero", outDimSize=4)
    assert not widened.is_surjective()
    assert {widened.apply({"idx": value})["zero"] for value in range(8)} == {0}


def test_identity_2d():
    layout = LinearLayout.from_bases(
        [
            ("in0", [[0, 1], [0, 2]]),
            ("in1", [[1, 0], [2, 0]]),
        ],
        ["out0", "out1"],
    )
    for row in range(4):
        for col in range(4):
            result = layout.apply({"in0": col, "in1": row})
            assert result == {"out0": row, "out1": col}


def test_operator_mul_identity():
    layout = LinearLayout.identity_1d(4, "idx", "out") * LinearLayout.identity_1d(8, "idx", "out")
    for value in range(8):
        assert layout.apply({"idx": value})["out"] == value


def test_operator_mul_disjoint_dims():
    layout = LinearLayout.identity_1d(8, "i0", "o0") * LinearLayout.identity_1d(4, "i1", "o1")
    for i0 in range(8):
        for i1 in range(4):
            result = layout.apply({"i0": i0, "i1": i1})
            assert result == {"o0": i0, "o1": i1}


def test_compose():
    reg = LinearLayout.identity_1d(8, "reg", "tensor")
    shared = LinearLayout.identity_1d(8, "tensor", "tensor")
    composed = reg.compose(shared)
    for idx in range(8):
        assert composed.apply({"reg": idx})["tensor"] == idx


def test_invert():
    base = LinearLayout.identity_1d(8, "inp", "out")
    inverted = base.invert()
    for value in range(8):
        out = base.apply({"inp": value})["out"]
        recovered = inverted.apply({"out": out})["inp"]
        assert recovered == value


def test_invert_and_compose():
    base = LinearLayout.identity_1d(8, "inp", "mid")
    other = LinearLayout.identity_1d(8, "out", "mid")
    inverted = base.invert_and_compose(other)
    for value in range(8):
        assert inverted.apply({"inp": value})["out"] == value


def test_get_matrix_view_identity():
    layout = LinearLayout.identity_1d(4, "idx", "idx")
    assert layout.get_matrix_view() == [
        [1, 0],
        [0, 1],
    ]


def test_get_matrix_view_strided():
    layout = LinearLayout.strided_1d(4, 2, "idx", "out")
    assert layout.get_matrix_view() == [
        [0, 0],
        [1, 0],
        [0, 1],
    ]


def test_get_matrix_view_from_bases():
    layout = LinearLayout.from_bases(
        [
            ("in0", [[1, 0], [2, 0]]),
            ("in1", [[0, 1], [0, 2]]),
        ],
        ["out0", "out1"],
    )
    assert layout.get_matrix_view() == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
