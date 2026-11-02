"""Compatibility imports used by consumers such as PyTorch Inductor."""


def test_compiler_exports_runtime_triton_key():
    from triton.compiler.compiler import triton_key
    from triton.runtime.cache import triton_key as runtime_triton_key

    assert callable(triton_key)
    assert triton_key is runtime_triton_key
