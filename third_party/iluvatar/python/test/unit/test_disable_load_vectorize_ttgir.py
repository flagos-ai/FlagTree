"""`disable_load_vectorize` driven through `triton.compile(options=...)`.

Complements test_disable_load_vectorize.py, which goes through launch kwargs.
Here the option is passed to `triton.compile` against hand-written TTGIR, which
is the path a tool or custom pipeline would use, and which pins the lowering
behavior without depending on argument specialization.

The TTGIR sets `sizePerThread = [4]` and `tt.divisibility` explicitly, so the
baseline genuinely vectorizes and the assertions are not vacuous.
"""

import re

import triton
from triton.compiler.compiler import GPUTarget

TTGIR = """
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32,
                   ttg.target = "cuda:71", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @load_store(%in: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                             %out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %inp = tt.splat %in : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ip = tt.addptr %inp, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %val = tt.load %ip : tensor<1024x!tt.ptr<f32>, #blocked>
    %outp = tt.splat %out : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %op = tt.addptr %outp, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    tt.store %op, %val : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
"""

# Global accesses lower to llvm.bi.{load,store}.kop.<vecty> intrinsics.
_LOAD_RE = re.compile(r"@llvm\.bi\.load\.kop\.v(\d+)[a-z]\d+")
_STORE_RE = re.compile(r"@llvm\.bi\.store\.kop\.v(\d+)[a-z]\d+")


def _compile(tmp_path, disable):
    src = tmp_path / "load_store.ttgir"
    src.write_text(TTGIR)
    return triton.compile(str(src), target=GPUTarget("corex", 71, 64),
                          options={"disable_load_vectorize": disable})


def _vector_loads(llir):
    """Element counts of vectorized (>1) global load intrinsics."""
    return [int(n) for n in _LOAD_RE.findall(llir) if int(n) > 1]


def _vector_stores(llir):
    return [int(n) for n in _STORE_RE.findall(llir) if int(n) > 1]


def test_default_vectorizes(tmp_path):
    """Baseline: sizePerThread=4 on f32 gives 128-bit loads."""
    llir = _compile(tmp_path, False).asm["llir"]
    assert _vector_loads(llir), llir


def test_option_disables_load_vectorization(tmp_path):
    llir = _compile(tmp_path, True).asm["llir"]
    assert not _vector_loads(llir), _vector_loads(llir)


def test_option_leaves_stores_vectorized(tmp_path):
    """Scoped to loads: the store in the same function keeps its width."""
    base = _vector_stores(_compile(tmp_path, False).asm["llir"])
    narrowed = _vector_stores(_compile(tmp_path, True).asm["llir"])
    assert base, "expected vectorized stores in the baseline"
    assert base == narrowed, f"store widths changed: {base} -> {narrowed}"


def test_incoming_ir_is_untouched(tmp_path):
    """The option must leave the TTGIR it was handed completely unchanged.

    Source locations are normalized away: the compile cache is keyed on source
    text plus options, so an artifact can come from whichever test first
    compiled that combination, carrying that test's tmp_path inside `loc(...)`.
    Everything that is not a file path must match exactly.
    """

    def without_locations(text):
        return re.sub(r'loc\("[^"]*"', 'loc("<path>"', text)

    base = _compile(tmp_path, False)
    narrowed = _compile(tmp_path, True)
    assert without_locations(base.asm["ttgir"]) == without_locations(narrowed.asm["ttgir"])
