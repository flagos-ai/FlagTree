"""LLVM test-module loading; retain the returned contexts with the module."""
import pytest
from triton._C import libtriton

if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU compiler required", allow_module_level=True)

from triton._C.libtriton import ir, llvm, hcu


def load_fixture(tmp_path, text):
    # Use existing public LLVM-library import: this exercises the built helper,
    # not a Python implementation of its legality checks.
    stub = tmp_path / 'root.mlir'
    stub.write_text('module { llvm.func @fixture() llvm.func @root() { llvm.call @fixture() : () -> () llvm.return } }')
    source = tmp_path / 'fixture.ll'
    source.write_text(text)
    context = ir.context()
    ir.load_dialects(context)
    hcu.load_dialects(context)
    mlir = ir.parse_mlir_module(str(stub), context)
    llvm_context = llvm.context()
    module = llvm.to_module(mlir, llvm_context)
    llvm.link_extern_libs(module, [str(source)])
    fixture = next(fn for fn in module.get_functions() if fn.name == 'fixture')
    fixture.set_calling_conv(hcu.CALLING_CONV_AMDGPU_KERNEL)
    return module, (context, mlir, llvm_context)
