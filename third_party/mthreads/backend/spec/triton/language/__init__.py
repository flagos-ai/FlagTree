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


def language_extend_globals(globals_dict):
    # NOTE: Must use absolute path import.
    from triton.language.standard import squeeze, unsqueeze  # Triton 3.7
    from triton.language.core import to_tensor  # Triton 3.7
    from triton.language.core import _experimental_descriptor_load, _experimental_descriptor_store
    globals_dict["squeeze"] = squeeze  # Triton 3.7
    globals_dict["unsqueeze"] = unsqueeze  # Triton 3.7
    globals_dict["to_tensor"] = to_tensor  # Triton 3.7
    globals_dict["_experimental_descriptor_load"] = _experimental_descriptor_load
    globals_dict["_experimental_descriptor_store"] = _experimental_descriptor_store
