"""Unused empty tied identities may become volatile read-only dependencies."""
import pytest
import triton
import triton.language as tl
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _probe(X, O, BLOCK: tl.constexpr, USED: tl.constexpr, REAL: tl.constexpr):
    i = tl.arange(0, BLOCK)
    x = tl.load(X + i)
    asm: tl.constexpr = "FOP.MOV $0, $1" if REAL else ""
    y = tl.inline_asm_elementwise(asm, constraints="=R,0", args=[x],
                                 dtype=tl.float32, is_pure=False, pack=1)
    if USED:
        tl.store(O + i, y)
    else:
        tl.store(O + i, x)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("used", [False, True])
def test_only_unused_identity_result_is_removed(enabled, used):
    k = compile_musa(_probe,
        {"X": "*fp32", "O": "*fp32", "BLOCK": "constexpr",
         "USED": "constexpr", "REAL": "constexpr"},
        {"BLOCK": 1024, "USED": used, "REAL": False},
        options={"num_warps": 4, "lower_dead_tied_register_fences": enabled})
    expected = enabled and not used
    assert bool(k.metadata.readonly_register_fence_count) == expected
    text = k.asm["llir"]
    assert ('call void asm sideeffect "", "R"' in text) == expected
    assert "store" in text and "load" in text
    assert "sideeffect" in text


def test_nonempty_register_instruction_is_retained():
    k = compile_musa(_probe,
        {"X": "*fp32", "O": "*fp32", "BLOCK": "constexpr",
         "USED": "constexpr", "REAL": "constexpr"},
        {"BLOCK": 1024, "USED": False, "REAL": True},
        options={"num_warps": 4, "lower_dead_tied_register_fences": True})
    assert k.metadata.readonly_register_fence_count == 0
    assert 'asm sideeffect "FOP.MOV $0, $1", "=R,0"' in k.asm["llir"]


def test_option_validation_and_cache():
    _, backend = mthreads_backend()
    assert not backend.parse_options({}).lower_dead_tied_register_fences
    assert backend.parse_options({}).hash() != backend.parse_options(
        {"lower_dead_tied_register_fences": True}).hash()
    for invalid in [1, "true", None]:
        with pytest.raises(ValueError, match="lower_dead_tied_register_fences must be a bool"):
            backend.parse_options({"lower_dead_tied_register_fences": invalid})


@pytest.mark.parametrize("constraints,side_effects,dtype,used,expected", [
    ("=R,0", True, "f32", False, 1),
    ("=R,0", True, "f32", True, 0),
    ("=R,0,~{memory}", True, "f32", False, 0),
    ("=R,R", True, "f32", False, 0),
    ("=R,0", False, "f32", False, 0),
    ("=R,0", True, "i32", False, 0),
])
def test_exact_llvm_fence_contract(tmp_path, constraints, side_effects, dtype, used, expected):
    """Test the helper before target codegen, including memory-clobber negatives."""
    from triton._C import libtriton
    from triton._C.libtriton import ir, llvm

    context = ir.context()
    ir.load_dialects(context)
    _, backend = mthreads_backend()
    backend.load_dialects(context)
    path = tmp_path / "fence.mlir"
    effects = "has_side_effects" if side_effects else ""
    result = f" -> {dtype}" if used else ""
    ret = f"llvm.return %out : {dtype}" if used else "llvm.return"
    path.write_text(f'''module {{
      llvm.func @probe(%arg: {dtype}){result} {{
        %out = llvm.inline_asm {effects} "", "{constraints}" %arg : ({dtype}) -> {dtype}
        {ret}
      }}
    }}''')
    module = ir.parse_mlir_module(str(path), context)
    llvm_context = llvm.context()
    llvm_module = llvm.to_module(module, llvm_context)
    before = str(llvm_module)
    assert libtriton.mthreads.lower_dead_tied_register_fences(llvm_module) == expected
    after = str(llvm_module)
    if expected:
        assert 'call void asm sideeffect "", "R"' in after
        assert "=R,0" not in after
        assert libtriton.mthreads.lower_dead_tied_register_fences(llvm_module) == 0
    else:
        assert after == before
