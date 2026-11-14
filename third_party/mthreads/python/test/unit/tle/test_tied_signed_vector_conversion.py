"""MP31 real destructive S32->F32 burst conversion, default off."""
import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir, llvm, mthreads
from test_tle_utils import compile_musa, mthreads_backend, require_mthreads_libtriton

require_mthreads_libtriton()


@triton.jit
def _cast(x, out, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(out + i, tl.load(x + i).to(tl.float32))


@pytest.mark.parametrize("enabled,width,itype", [
    (False, 4, "i32"),
    (True, 4, "i32"),
    (True, 1, "i32"),
    (True, 4, "u32"),
    (True, 4, "i16"),
])
def test_exact_signed_four_lane_conversion_only(enabled, width, itype):
    _, backend = mthreads_backend()
    if not hasattr(backend.parse_options({}), "fp32_vector_width"):
        pytest.skip("optional integration requires companion MP31 vector math lowering")
    k = compile_musa(_cast, {"x": "*" + itype, "out": "*fp32", "BLOCK": "constexpr"}, {"BLOCK": 1024},
                     options={"num_warps": 4, "fp32_vector_width": width, "tie_signed_vector_conversions": enabled})
    active = enabled and width == 4 and itype == "i32"
    assert bool(k.metadata.tied_signed_vector_conversion_count) == active
    text = k.asm["llir"]
    assert ('"CVT.RTE.BST4 $0.f32, $1.s32", "=R,0"' in text) == active
    if active:
        assert "bitcast <4 x i32>" in text
        assert "sitofp <4 x i32>" not in text
    assert " fast " not in text and " reassoc " not in text
    assert k.module is None and k.function is None


def test_default_option_and_cache_separation():
    _, backend = mthreads_backend()
    a = backend.parse_options({})
    b = backend.parse_options({"tie_signed_vector_conversions": True})
    assert not a.tie_signed_vector_conversions and a.hash() != b.hash()
    for value in [1, "true", None]:
        with pytest.raises(ValueError, match="must be a bool"):
            backend.parse_options({"tie_signed_vector_conversions": value})


def test_independent_fragment_chunks_keep_all_conversions(tmp_path):
    # Exercise eight independent contiguous chunk positions, with every output
    # live. No operation is deleted or reassociated by the allocation policy.
    functions = []
    for chunk in range(8):
        indices = ", ".join(str(4 * chunk + i) for i in range(4))
        functions.append(f'''llvm.func @chunk{chunk}(%raw: vector<32xi32>) -> vector<4xf32> {{
          %s = llvm.shufflevector %raw, %raw [{indices}] : vector<32xi32>
          %v = llvm.sitofp %s : vector<4xi32> to vector<4xf32>
          llvm.return %v : vector<4xf32>
        }}''')
    p = tmp_path / "fragment.mlir"
    p.write_text("module {" + "\n".join(functions) + "}")
    ctx = ir.context()
    ir.load_dialects(ctx)
    _, backend = mthreads_backend()
    backend.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(p), ctx)
    llvm_ctx = llvm.context()
    module = llvm.to_module(mod, llvm_ctx)
    count = mthreads.tie_signed_vector_conversions(module)
    assert count == 8
    text = str(module)
    assert text.count('asm "CVT.RTE.BST4') == 8
    assert text.count("sitofp <4 x i32>") == 8 - 8


@pytest.mark.parametrize("width,bits,unsigned,strict,expected", [
    (4, 32, False, False, 1),
    (2, 32, False, False, 0),
    (4, 16, False, False, 0),
    (4, 32, True, False, 0),
    (4, 32, False, True, 0),
])
def test_standalone_signed_conversion_contract(tmp_path, width, bits, unsigned, strict, expected):
    """Direct LLVM coverage independent of companion frontend vectorization."""
    conversion = "uitofp" if unsigned else "sitofp"
    attributes = 'attributes {passthrough = ["strictfp"]}' if strict else ''
    path = tmp_path / "signed-contract.mlir"
    path.write_text(f"""module {{
      llvm.func @probe(%x: vector<{width}xi{bits}>) -> vector<{width}xf32> {attributes} {{
        %y = llvm.{conversion} %x : vector<{width}xi{bits}> to vector<{width}xf32>
        llvm.return %y : vector<{width}xf32>
      }}
    }}""")
    context = ir.context()
    ir.load_dialects(context)
    _, backend = mthreads_backend()
    backend.load_dialects(context)
    parsed = ir.parse_mlir_module(str(path), context)
    llvm_context = llvm.context()
    module = llvm.to_module(parsed, llvm_context)
    before = str(module)
    assert mthreads.tie_signed_vector_conversions(module) == expected
    after = str(module)
    if expected:
        assert 'CVT.RTE.BST4' in after and '"=R,0"' in after
        assert 'sitofp <4 x i32>' not in after
    else:
        assert before == after
