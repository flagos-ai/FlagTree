# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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


import pytest
from triton.backends.ascend.utils import downgrade_llir, _downgrade_mem_attrs, _downgrade_stacksaverestore_intrinsics

@pytest.mark.parametrize("new_attr,legacy_attrs", [
    ("memory(none)" , ["readnone"]),
    ("memory(read)" , ["readonly"]),
    ("memory(write)" , ["writeonly"]),
    ("memory(readwrite)" , []),
    ("memory(argmem: read)" , ["readonly", "argmemonly"]),
    ("memory(argmem: read, inaccessiblemem: write)" , ["inaccessiblemem_or_argmemonly"]),
    ("memory(read, argmem: readwrite)" , []),
    ("memory(readwrite, argmem: none)" , []),
])
def test_mem_attrs(new_attr, legacy_attrs):
    assert _downgrade_mem_attrs(new_attr).strip().split() == legacy_attrs

@pytest.mark.parametrize("new_intr,legacy_intr", [
    ("declare ptr @llvm.stacksave.p0()" , "declare ptr @llvm.stacksave()"),
    ("declare ptr addrspace(5) @llvm.stacksave.p5()" , "declare ptr addrspace(5) @llvm.stacksave()"),
    ("declare void @llvm.stackrestore.p0(ptr %ptr)" , "declare void @llvm.stackrestore(ptr %ptr)"),
    ("declare void @llvm.stackrestore.p5(ptr addrspace(5) %ptr)" , "declare void @llvm.stackrestore(ptr addrspace(5) %ptr)"),
    ("%53 = call ptr @llvm.stacksave.p0()" , "%53 = call ptr @llvm.stacksave()"),
    ("call void @llvm.stackrestore.p0(ptr %53)" , "call void @llvm.stackrestore(ptr %53)"),
])
def test_stacksaverestore_intrinsics(new_intr, legacy_intr):
    assert _downgrade_stacksaverestore_intrinsics(new_intr).strip() == legacy_intr
