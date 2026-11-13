# Copyright 2025-     FlagOS Contributors
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

def validate_block_shape(shape):
    # Imported lazily because the shared _utils module dispatches here.
    from triton._utils import TRITON_MAX_TENSOR_NUMEL

    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(
                f"Shape element {i} must have type `constexpr[int]`, "
                f"got `constexpr[{type(d)}]`"
            )
        if d < 0:
            raise ValueError(f"Shape element {i} must be non-negative")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(
            f"numel ({numel}) exceeds triton maximum tensor numel "
            f"({TRITON_MAX_TENSOR_NUMEL})"
        )
    return numel
