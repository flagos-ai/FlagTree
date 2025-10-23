# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
