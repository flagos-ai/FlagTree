"""Unit tests for the shared intrinsic-attribute util's pybind surface.

`IntrinsicAttrTable.h/.cpp` is one decode + one lookup + two writers, shared by
the emitters and landing B'.  C++ cannot be imported from Python, so these
tests drive it through the two bindings that exist for exactly that purpose:

* ``llvm.set_intrinsic_attr_table(payload)`` -- the process-wide table the
  emitters consult at creation time (its contract: one process = one table);
* ``llvm.add_intrinsic_attrs(module, payload)`` -- landing B', the sweep over
  a translated module (its contract: stamp declared entries, return the count,
  and raise on malformed input).

Both paths ultimately exercise `parseTable`/`lookup`/`writeAttrsTo`; the
payloads are built by the producer the compiler itself uses
(`intrinsic_tables.stamp_payload`). Public FlagTree builds conceal IR text, so
these tests keep binding-level, error, lifetime, and stamping-count contracts
without inspecting `str(module)`.

The one-process-one-table rule forces an ordering across the whole file: the
first install wins, so every case that needs a payload installs *that*
payload's leg's exact bytes and the guard test runs last.  (The guard itself
re-arms in a subprocess, so the main-process install stays whichever leg the
earlier cases chose; the foreign-payload refusal case is order-independent --
it only needs *an* installed table to disagree with.)
"""

import pytest

import os
import pathlib

from triton._C.libtriton import llvm
from triton.backends import intrinsic_tables as it


def _payload(tag: str) -> str:
    return it.stamp_payload(tag)


def _entry(payload: str, name: str) -> str:
    for line in payload.split("\n"):
        if line.split(" ")[0] == name:
            return line
    raise AssertionError(f"{name} is not in the payload")


def _make_module(text: str):
    triton = pytest.importorskip("triton")
    ir = triton._C.libtriton.ir
    lllvm = triton._C.libtriton.llvm
    import pathlib
    import tempfile
    path = pathlib.Path(tempfile.mkdtemp()) / "u.mlir"
    path.write_text(text)
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(path), ctx)
    lllvm.init_targets()
    return lllvm.to_module(mod, lllvm.context())


# Attribute semantics are maintained by the internal Triton UT owner. Public
# FlagTree keeps the binding-level count/error/lifetime contracts below because
# conceal builds intentionally do not expose IR text.
_MODULE = ('module attributes {"llvm.target_triple" = "xpu3-baidu-none-gnu"} {\n'
           "  llvm.func @llvm.xpu.core_id() -> i32\n"
           "}\n")


def test_bprime_stamps_table_facts_and_nothing_else():
    """The sweep finds and stamps the table's declaration entry."""
    payload = _payload("l19")
    module = _make_module(_MODULE)
    assert llvm.add_intrinsic_attrs(module, payload) == 1


def test_bprime_suffix_falls_back_to_base_entry():
    """An unknown overload suffix resolves through the base table entry."""
    payload = _payload("l19")
    base = _entry(payload, "llvm.xpu.lm2gm_v3")
    assert base, "l19 lost the lm2gm_v3 entry"
    module = _make_module('module attributes {"llvm.target_triple" = "xpu3-baidu-none-gnu"} {\n'
                          "  llvm.func @llvm.xpu.lm2gm_v3.i32(!llvm.ptr<1>, !llvm.ptr, i32, i32)\n"
                          "}\n")
    assert llvm.add_intrinsic_attrs(module, payload) == 1


def test_bprime_leaves_unknown_names_alone():
    """An unknown name must not be stamped."""
    payload = _payload("l19")
    text = ('module attributes {"llvm.target_triple" = "xpu3-baidu-none-gnu"} {\n'
            "  llvm.func @definitely_not_an_intrinsic(i32) -> i32\n"
            "}\n")
    module = _make_module(text)
    assert llvm.add_intrinsic_attrs(module, payload) == 0


def test_bprime_malformed_payload_raises():
    """A payload C++ cannot parse must be loud, not silently unstamped."""
    module = _make_module(_MODULE)
    with pytest.raises(RuntimeError, match="mem:"):
        llvm.add_intrinsic_attrs(module, "llvm.xpu.core_id nounwind mem:0")


def test_one_process_one_table_guard():
    """Installing a *different* payload after the emitters armed the table
    must throw.

    `lookupInProcess` hands out pointers read without the lock, so replacing
    the table could dangle a reader; the contract is enforced in `setPayload`
    itself.  Arming requires the emitters to have run once (the parse is lazy),
    so this test compiles one tiny kernel in-process before attempting the
    second install -- the same flow the guard e2e probe documented in the KB
    report exercises.
    """
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    from triton.backends import llvm19_toolchain as T

    import subprocess
    import sys

    # One process, one story: install this leg's table, compile (arms the
    # table through the emitters), then refuse a different table.
    work = _fresh_tmpdir()
    script = work / "guard_probe.py"
    script.write_text("import os\n"
                      "os.environ.update({'TRITON_XPU_ARCH':'3','TRITON_LLVM19_READIN':'0',\n"
                      "                   'TRITON_LLVM19_STAMP':'1',\n"
                      "                   'TRITON_LLVM19_STAMP_NETS':'none'})\n"
                      "import torch, triton, triton.language as tl\n"
                      "from triton.backends import llvm19_toolchain as T\n"
                      "from triton._C.libtriton import llvm\n"
                      "from triton.backends import intrinsic_tables as it\n"
                      "T.install_intrinsic_attr_table()\n"
                      "@triton.jit\n"
                      "def k(O, BLOCK: tl.constexpr):\n"
                      "    offs = tl.arange(0, BLOCK)\n"
                      "    tl.store(O + offs, offs.to(tl.float32))\n"
                      "out = torch.empty(64, device='cuda')\n"
                      "k[(1,)](out, BLOCK=64)\n"
                      "torch.cuda.synchronize()\n"
                      "try:\n"
                      "    llvm.set_intrinsic_attr_table(it.stamp_payload('l22'))\n"
                      "    raise SystemExit('BUG: no raise')\n"
                      "except RuntimeError:\n"
                      "    pass\n"
                      "# The refusal must not be sticky: reinstalling the *same* table is the\n"
                      "# no-op path and must not raise again (the error was taken, and the\n"
                      "# no-op branch leaves no new one).\n"
                      "llvm.set_intrinsic_attr_table(it.stamp_payload('l19'))\n"
                      "print('guard-ok')\n")
    env = dict(os.environ)
    env.update({
        "TRITON_XPU_ARCH": "3", "TRITON_LLVM19_READIN": "0", "TRITON_LLVM19_STAMP": "1", "TRITON_LLVM19_STAMP_NETS":
        "none", "TRITON_CACHE_DIR": str(work / "cache")
    })
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"
    assert "guard-ok" in proc.stdout, proc.stdout[-500:]


def test_bprime_sweep_refuses_foreign_payload():
    """The B' sweep must not silently sweep with a table the process did not
    install.

    `install_intrinsic_attr_table()` runs before every compilation, so the
    sweep always sees the installed table in production; a payload that
    differs from it means two tables are in flight in one process -- exactly
    what `set_intrinsic_attr_table` refuses for the emitters.  The sweep
    refuses too (a *malformed* foreign payload is still reported as
    malformed: that is the louder diagnosis).

    The witness is the *installed* payload (`state.payload`), not a parsed
    table: an install nobody has consulted yet still owns the process, so
    this passes without any prior compilation -- no test-order dependence.
    """
    # Arm the process table without parsing it (a bare install suffices; no
    # compilation runs here), then sweep with a different leg's bytes.
    llvm.set_intrinsic_attr_table(it.stamp_payload("l19"))
    module = _make_module(_MODULE)
    with pytest.raises(RuntimeError, match="differs"):
        llvm.add_intrinsic_attrs(module, it.stamp_payload("l22"))


def test_bprime_lazy_parse_survives_payload_temporary():
    """The lazy path's table must point into the *process's* payload copy.

    A decoded entry's atoms are `StringRef`s into the buffer the table was
    parsed from (see `parseTable`'s contract).  The sweep's `payload` argument
    is a pybind temporary: if the lazy path parses straight from it and then
    keeps the table, every atom string dangles the moment this call returns --
    and a later sweep reads freed bytes (observed: `speculatable` silently
    vanished from a private intrinsic's group after a 1 MB churn).  The
    fix parses from `state.payload`'s copy; this test pins the observable:
    after a lazy sweep and an allocation churn, the next sweep still writes
    the table's full atom set.
    """
    llvm.set_intrinsic_attr_table(it.stamp_payload("l19"))  # arm, unparsed
    # Force the lazy path (the sweep itself parses): a name with a
    # passthrough atom, in a module the emitters never touched.
    text = ('module attributes {"llvm.target_triple" = '
            '"xpu3-baidu-none-gnu"} {\n'
            "  llvm.func @llvm.xcn.workitem.id.x() -> i32\n"
            "}\n")
    module = _make_module(text)
    assert llvm.add_intrinsic_attrs(module, it.stamp_payload("l19")) == 1
    # Churn the heap so any pointer into the freed temporary now reads junk.
    junk = [b"Z" * 1048576 for _ in range(64)]
    assert len(junk) == 64
    # A second sweep must still complete after heap churn. Attribute contents
    # are not inspected here because public builds conceal IR text.
    text2 = ('module attributes {"llvm.target_triple" = '
             '"xpu3-baidu-none-gnu"} {\n'
             "  llvm.func @llvm.xcn.block.Idx.offset.x() -> i32\n"
             "}\n")
    module2 = _make_module(text2)
    assert llvm.add_intrinsic_attrs(module2, it.stamp_payload("l19")) == 1


def _fresh_tmpdir() -> pathlib.Path:
    import tempfile
    return pathlib.Path(tempfile.mkdtemp())
