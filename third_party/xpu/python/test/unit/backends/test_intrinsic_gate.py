"""The intrinsic name/attribute gates (names gate … fault gate).

Every gate here answers one question about *generated* IR and the two tables the
plan calls the norm, and none of them needs anything precomputed by hand:

* the tables are the ones the wheel ships (`triton.backends._tables/*.json`,
  decoded from the staged LLVM 19 / XTDK LLVM 22 `IntrinsicImpl.inc` at build
  time, with the source sha256 as provenance); both ship, and the one the corpus
  gates judge a module against is the leg's own (`llvm19_toolchain.stamp_tag()`),
  because that is the table the compiler in this process aligned it to;
* the ledger is `triton.backends.intrinsic_bridge_registry.json`;
* the corpus is whatever cache the caller points at with ``TRITON_GATE_CACHE``
  (the smoke stage passes its own cold cache).

The corpus gates skip loudly when no cache is given.  A missing *table* is never
a skip, though: the wheel has to carry both, and one that does not is a failure
here instead of a file full of nothing-to-check.  `attrs gate` judges the text
*after* the LLVM 19 read-in round and `synth gate` the module the
compiler itself emitted, before it (landing B): together they cover the
two consumers a declaration has to satisfy -- one that re-parses the text, and
one that never does.

The checks themselves live in the shipped module, as pure text/table functions,
so the last section can hand them deliberately broken modules and prove they go
red -- a gate that only ever sees clean input does not demonstrate anything.
"""

import hashlib
import json
import os
import pathlib
import re

import pytest

from triton.backends import intrinsic_tables as it
from triton.backends import llvm19_toolchain

CACHE_ENV = "TRITON_GATE_CACHE"
LIMIT_ENV = "TRITON_GATE_LIMIT"


def _load_shipped(tag):
    """(table, error) for one shipped table; the error names *that* table."""
    try:
        return it.load_shipped(tag), None
    except Exception as e:  # noqa: BLE001 - name the missing table; see _require_tables
        return None, f"{tag}: {e}"


T19, _E19 = _load_shipped("l19")
T22, _E22 = _load_shipped("l22")
TABLES_ERROR = "; ".join(e for e in (_E19, _E22) if e) or None

# Which leg this install is (handoff §7 ruling ①: "each leg asserts its own
# table").  Both tables ship, but only one is the norm for the modules the
# compiler *in this process* produces -- `stamp_tag()` is the selector landing B
# stamps with, so the corpus gates judge a module against the table its compiler
# aligned it to instead of assuming one leg.  The default leg's in-process
# verifier/optimizer is the public LLVM 22, which has no table for the private
# names at all (l19); the fallback leg runs XTDK's LLVM 22, whose reader resolves
# them against its own table (l22) -- R13 P1②.
LEG = llvm19_toolchain.stamp_tag()
TABLE = {"l19": T19, "l22": T22}.get(LEG)

# The full vendor ledger no longer ships with this tree, so it cannot be
# loaded here.  The gate tests below judge the *gate*, not the table -- the only
# name they need is the one their own fixtures use.  `llvm.xcn.ballot` must stay
# OUT of this set: `test_fault_gate_bare_overloaded_name_is_red` asserts that a
# bare overloaded name IS flagged.
GATE_NAMES = frozenset({"llvm.xcn.s.barrier"})

# The corpus gates (every test that walks `_modules()`) scan real `.llir` files
# and therefore need the FULL name set, which this tree no longer carries.
# They skip explicitly rather than judging against GATE_NAMES.
_FULL_LEDGER_AVAILABLE = False

_LLVM19_PKG = "xtdk-llvm19"
_LLVM22_PKG = "xtdk"


def _require_tables():
    """Both tables (and a usable leg), or a failure -- never a skip.

    A skip makes every gate a no-op while pytest still exits 0, and that is
    not hypothetical: a build whose XTDK package was not staged ships no
    `l22.json` (`setup.py` notes it and carries on), which used to turn this file
    into `17 skipped / rc=0` -- a gate that judged nothing (R3 §11, R14 F1).
    """
    if T19 is None or T22 is None:
        pytest.fail(
            f"the wheel ships no intrinsic table for {TABLES_ERROR}; build it "
            "with the toolchain staged (setup.py decodes them from "
            "_deps/*/IntrinsicImpl.inc)", pytrace=False)
    if TABLE is None:
        pytest.fail(f"stamp_tag() says this leg is {LEG!r}, which is not one of "
                    f"the shipped tags (l19 / l22)", pytrace=False)


def _cache_dir() -> pathlib.Path:
    raw = os.environ.get(CACHE_ENV, "")
    if not raw:
        pytest.skip(f"{CACHE_ENV} is not set: the corpus gates need a cold-cache "
                    f"directory (the smoke stage passes its own)")
    cache = pathlib.Path(raw)
    if not cache.is_dir():
        pytest.skip(f"{CACHE_ENV}={raw} is not a directory")
    return cache


def _limit() -> int:
    """How many modules the corpus gates may read: `TRITON_GATE_LIMIT`, 0 = all.

    Unset (or unparsable) means 0 -- the whole cache, which is what the gate
    actually ran on every recorded corpus; set it to a positive N to sample the
    first N modules by path while iterating on a gate (R3 F6: the docstrings
    below used to claim "default 200 modules").
    """
    try:
        return int(os.environ.get(LIMIT_ENV, "0"))
    except ValueError:
        return 0


def _modules():
    """(path, text) of every cached module, excluding a nested leg directory."""
    cache = _cache_dir()
    limit = _limit()
    out = []
    for p in sorted(cache.rglob("*.llir")):
        if "xtdk-leg" in p.parts:
            continue
        try:
            out.append((p, p.read_text(errors="ignore")))
        except OSError:
            continue
        if limit and len(out) >= limit:
            break
    if not out:
        pytest.skip(f"no .llir under {cache}")
    return out


def _module_meta(llir: pathlib.Path):
    """{arch, backend, is_sdnn} for a cached module, or None."""
    for cand in (llir.with_suffix(".json"), llir.parent / "kernel.json"):
        if not cand.is_file():
            continue
        try:
            meta = json.loads(cand.read_text())
        except Exception:  # noqa: BLE001 - treat broken metadata as absent; provenance gate covers it
            continue
        target = meta.get("target") or {}
        arch = meta.get("arch", target.get("arch"))
        if arch is None and not meta.get("is_sdnn"):
            continue
        return {"arch": arch, "backend": target.get("backend"), "is_sdnn": bool(meta.get("is_sdnn"))}
    return None


# ---------------------------------------------------------------------------
# names gate / mangling gate / matrix gate
# ---------------------------------------------------------------------------


def test_names_gate_every_emitted_name_resolves():
    """names gate: every private name in the corpus resolves in the backend table.

    An unresolvable name is not an intrinsic to llc: it becomes an ordinary
    external call, so the kernel fails to link or binds to a wrong signature.
    The table here is the LLVM 19 one on *both* legs, by the same argument as the
    product's own name guard (`llvm19_toolchain.check_intrinsic_names`): the
    staged LLVM 19 `llc` is what resolves the name in the end, whatever table the
    frontend used to create the call.
    """
    if not _FULL_LEDGER_AVAILABLE:
        pytest.skip("corpus gate needs the full vendor ledger, which this tree no longer ships")
    _require_tables()
    bad = {}
    for path, text in _modules():
        found = it.unresolved_names(T19, text, GATE_NAMES.__contains__)
        if found:
            bad[str(path)] = found
    assert not bad, ("modules name private intrinsics the LLVM 19 table cannot resolve:\n  " +
                     "\n  ".join(f"{k}: {v}" for k, v in list(bad.items())[:10]))


def test_mangling_gate_bare_overloaded_names_are_registered():
    """mangling gate: a bare overloaded name is only allowed as a recorded exception.

    Both consumers rewrite a bare name to its canonical suffixed spelling while
    parsing, so a bare name in generated IR means the emitter relies on that
    rewrite instead of spelling the name it resolves to.  Judged against the
    LLVM 19 table like `names gate`: the rewrite that has to happen is `llc`'s.
    """
    if not _FULL_LEDGER_AVAILABLE:
        pytest.skip("corpus gate needs the full vendor ledger, which this tree no longer ships")
    _require_tables()
    bad = {}
    for path, text in _modules():
        found = it.bare_overloaded_names(T19, text, GATE_NAMES.__contains__)
        if found:
            bad[str(path)] = found
    assert not bad, ("bare names of overloaded intrinsics (spell the canonical suffix, or "
                     "record the exception in the registry):\n  " +
                     "\n  ".join(f"{k}: {v}" for k, v in list(bad.items())[:10]))


def test_matrix_gate_namespace_matches_the_backend():
    """matrix gate: the name space has to match the (backend, arch, is_sdnn) row.

    The alternate cluster family belongs to the alternate cluster backends, the
    xpu3 family to the xpu3 backend; an alternate backend build may emit the xpu3
    family only in its SDNN mode.  A wrong
    name space is a name the consumer's own table does not know at all.
    """
    _require_tables()
    bad = []
    for path, text in _modules():
        meta = _module_meta(path)
        if meta is None:
            continue
        names = set(it.ir_names(text))
        if meta["is_sdnn"] and meta["arch"] == "xpu5":
            continue  # alternate SDNN: both families legal
        if meta["arch"] == 3 or meta["arch"] == "xpu3":
            wrong = sorted(n for n in names if n.startswith("llvm.xcn."))
        else:
            wrong = sorted(n for n in names if n.startswith("llvm.xpu."))
        if wrong:
            bad.append(f"{path}: arch={meta['arch']} is_sdnn={meta['is_sdnn']} "
                       f"-> {wrong[:4]}")
    assert not bad, "name space does not match the module's target:\n  " + \
        "\n  ".join(bad[:10])


# ---------------------------------------------------------------------------
# calls gate
# ---------------------------------------------------------------------------


def test_calls_gate_convergent_is_visible_where_the_table_requires_it():
    """calls gate: the call-site half of the convergent rule.

    The declare side is covered by the audit test; this gate reads the
    call sites, which is what the in-process optimizer (the one that decides
    whether a call may be duplicated into a branch) sees.  That optimizer is
    this leg's, so the expectation is this leg's table.
    """
    _require_tables()
    bad = {}
    for path, text in _modules():
        found = it.convergent_call_violations(TABLE, text)
        if found:
            bad[str(path)] = found
    assert not bad, ("convergent is not visible where the table requires it (or is stamped "
                     "where it is not allowed):\n  " + "\n  ".join(f"{k}: {v}" for k, v in list(bad.items())[:10]))


# ---------------------------------------------------------------------------
# registry gate / xdiff gate / provenance gate
# ---------------------------------------------------------------------------


# The `registry` ledger-audit gate was removed from the public tree: it reads the
# vendor ledger (`intrinsic_bridge_registry.json`), which no longer ships here.
# It lives on in the internal tree's copy of this file, which still has the ledger.
def _payload_map(tag: str) -> dict:
    """{name: the payload's attribute fields} for one tag, as a consumer sees it."""
    out = {}
    for line in it.stamp_payload(tag).split("\n"):
        name, _, rest = line.partition(" ")
        out[name] = rest
    return out


def _payload_memory(rest: str | None) -> str | None:
    """The `mem:a-b-c` triple of one payload line's fields, or None."""
    found = re.search(r"\bmem:(\d+-\d+-\d+)", rest or "")
    return found.group(1) if found else None


_PRIVATE_PREFIXES = ("llvm.xcn.", "llvm.xpu.")


def _private_namespace_names() -> list[str]:
    """Every name of the private families *either* table knows (the union).

    The union, not the intersection (W6b: the intersection left the names exactly
    one table knows in a gap -- the guard of the leg that *has* them passes them,
    and the completeness side of this gate did not ask for them at all).  Such a
    name is the asymmetry this file exists to keep visible: on the leg whose
    consumer does not know it, the same declaration is a plain external call with
    no table attributes, or an undefined symbol at link time.  The union is split
    below by what can be stated about a name -- names both tables know are judged
    value-by-value (xdiff rows), names one table knows are pinned as a set
    (`single_table`).
    """
    return sorted(name for name in set(T19.names) | set(T22.names) if name.startswith(_PRIVATE_PREFIXES))


def _single_table_sides() -> dict:
    """{tag: the private names only that tag's table knows}, both directions."""
    return {
        tag: [name
              for name in _private_namespace_names()
              if table.has(name) and not other.has(name)]
        for tag, table, other in (("l19", T19, T22), ("l22", T22, T19))
    }


def _sha256_lines(lines) -> str:
    """One digest over an ordered list of lines -- the set pins' only degrees."""
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# The `xdiff` ledger-audit gate was removed from the public tree: it reads the
# vendor ledger (`intrinsic_bridge_registry.json`), which no longer ships here.
# It lives on in the internal tree's copy of this file, which still has the ledger.
def test_tables_gate_are_shipped():
    """tables gate: the wheel carries *both* tables, and this leg's is usable.

    `_require_tables()` now fails every other gate when a table is missing, so
    this is the one that says what "both" means: the two files are in the
    package, they are two different tables (a build that wrote one table twice
    would satisfy a naive existence check), and they are the pair the build
    decodes.  A clean-room build that never staged the XTDK package ships no
    `l22.json` at all and used to make this whole file skip (R14 F1); which of
    the two *this* leg judges against is `stamp_tag()`, and `_require_tables`
    rejects a tag that is not one of them.
    """
    _require_tables()
    missing = [tag for tag in ("l19", "l22") if not it.shipped_path(tag).is_file()]
    assert not missing, (f"the wheel is missing {missing} under {it.SHIPPED_DIR}; a leg that "
                         f"asserts its own table needs both of them shipped")
    counts = {"l19": len(T19.names), "l22": len(T22.names)}
    assert counts == {"l19": 16240, "l22":
                      17455}, (f"the pair the build decodes is 16,240 / 17,455 names (same counts as "
                               f"provenance gate); this wheel has {counts}")
    assert T19.source_sha256 != T22.source_sha256, (
        f"both tables cite one source ({T19.source}): one of them was written "
        f"under the wrong name")


def test_provenance_gate_shipped_tables():
    """provenance gate: the tables in use are the ones the build decoded.

    The counts are the ones the audit recorded; a different number means the
    package was built against another `.inc` (or the JSON is stale), either of
    which invalidates every other gate silently.
    """
    _require_tables()
    assert len(T19.names) == 16240, f"l19 name count {len(T19.names)} != 16240"
    assert len(T22.names) == 17455, f"l22 name count {len(T22.names)} != 17455"
    assert len(T19.family("llvm.xcn.")) == 875
    assert len(T19.family("llvm.xpu.")) == 1428
    assert len(T22.family("llvm.xcn.")) == 403
    assert len(T22.family("llvm.xpu.")) == 1418
    assert T19.source_sha256 and T22.source_sha256, "no source provenance"
    assert _LLVM19_PKG in T19.source and _LLVM22_PKG in T22.source, (
        f"tables cite unexpected sources: {T19.source} / {T22.source}")


# ---------------------------------------------------------------------------
# attrs gate / synth gate: waiting for the LLVM 19 read-in pass
# ---------------------------------------------------------------------------


def _repo_root() -> pathlib.Path:
    """The checkout this gate runs from (…/python/test/unit/backends)."""
    return pathlib.Path(__file__).resolve().parents[4]


def _emitted_name_literals() -> set[str]:
    """Private intrinsic name literals the emitters in this repo mention.

    The attribute gate judges these (`scope = emitters`, the same scope the
    audit tool uses).  Device-library declares (`ocml.bc`, `libdevice-*.bc`) reach
    a kernel's module with the library's own attributes, which the tables need
    not agree with -- they are not our emissions and not ours to pin.
    """
    root = _repo_root()
    dirs = ["lib", "include", "third_party/xpu/lib", "third_party/xpu/include"]
    pattern = re.compile(r'"(llvm\.(?:xcn|xpu)\.[A-Za-z0-9._]*)"')
    names: set[str] = set()
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            if f.suffix not in (".cpp", ".h", ".cc", ".td"):
                continue
            try:
                names.update(pattern.findall(f.read_text(errors="ignore")))
            except OSError:
                continue
    return names


def _is_ours(name: str, literals: set[str]) -> bool:
    return any(lit == name or name.startswith(lit) or lit.startswith(name) for lit in literals)


def _ir_text_is_concealed() -> bool:
    """True when this build serialises modules as empty text (TRITON_CONCEAL_IR)."""
    triton = pytest.importorskip("triton")
    ir = triton._C.libtriton.ir
    lllvm = triton._C.libtriton.llvm
    import tempfile
    path = pathlib.Path(tempfile.mkdtemp()) / "conceal_probe.mlir"
    path.write_text('module attributes {"llvm.target_triple" = "xpu3-baidu-none-gnu"} {\n'
                    "  llvm.func @probe(i32) -> i32\n"
                    "}\n")
    ctx = ir.context()
    ir.load_dialects(ctx)
    mod = ir.parse_mlir_module(str(path), ctx)
    lllvm.init_targets()
    return not str(lllvm.to_module(mod, lllvm.context()))


# The `attrs` ledger-audit gate was removed from the public tree: it reads the
# vendor ledger (`intrinsic_bridge_registry.json`), which no longer ships here.
# It lives on in the internal tree's copy of this file, which still has the ledger.
def test_emit_gate_kernel_declaration_carries_the_table(tmp_path):
    """emit gate: a declaration the emitters created carries the table at birth.

    The emitters (`createIntrinsicCallByName`, `createCallIntrinsic`, the two
    cluster declaration helpers) create their declaration through the gate,
    which reads the in-process table *while it creates*.  That is the fact path
    the plan wants -- nothing "patches up" afterwards -- so the gate judges it on
    a real kernel, not on a synthetic module: compile the CSE-gap kernel (four
    `core_id` uses, the xpu3 STATUS-SUMMARY item 18 shape) with the B' sweep off
    (`TRITON_LLVM19_STAMP_NETS=none`), and the pre-O3 dump it leaves behind is
    *only* what the emitters wrote.  Every private declaration in it has to
    carry this leg's table set; with the sweep on, the same assertion must hold
    (the sweep is a no-op for a name the emitter already covered).
    """
    _require_tables()
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    import triton.language as tl  # noqa: F811 - the kernel below needs it bound

    net_env = os.environ.get("TRITON_LLVM19_STAMP_NETS", "all")
    assert net_env in ("all", "none"), (f"run this gate with TRITON_LLVM19_STAMP_NETS=all or none "
                                        f"(got {net_env!r}); other values no longer exist")

    @triton.jit
    def _emit_gate_kernel(O, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        v = tl.sum(offs.to(tl.float32)) + pid
        tl.store(O + offs, v * 0.0 + offs.to(tl.float32))

    cache = tmp_path / "cache"
    dump = tmp_path / "dump"
    dump.mkdir()
    env = dict(os.environ)
    env.update({
        "TRITON_LLVM19_STAMP_NETS": "none", "TRITON_LLVM19_STAMP": "1", "TRITON_LLVM19_READIN": "0", "TRITON_CACHE_DIR":
        str(cache), "TRITON_DUMP_PRE_O3_LLIR": str(dump) + os.sep
    })
    script = tmp_path / "emit_gate_kernel.py"
    script.write_text("import torch, triton, triton.language as tl\n"
                      "@triton.jit\n"
                      "def _emit_gate_kernel(O, n, BLOCK: tl.constexpr):\n"
                      "    pid = tl.program_id(0)\n"
                      "    offs = tl.arange(0, BLOCK)\n"
                      "    v = tl.sum(offs.to(tl.float32)) + pid\n"
                      "    tl.store(O + offs, v * 0.0 + offs.to(tl.float32))\n"
                      "out = torch.empty(256, device='cuda')\n"
                      "_emit_gate_kernel[(1,)](out, 256, BLOCK=256)\n")
    import subprocess, sys
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"kernel failed:\n{proc.stderr[-2000:]}"

    dumps = sorted(dump.glob("*.llir"))
    assert dumps, "no pre-O3 dump was produced; is TRITON_DUMP_PRE_O3_LLIR set?"
    text = dumps[0].read_text(errors="replace")
    if not text.strip() and _ir_text_is_concealed():
        # An empty dump on a concealed build is the channel being closed by design
        # (TRITON_CONCEAL_IR), not a broken emitter; the assertions below still face
        # every non-empty dump, and an empty dump on a build that does not conceal
        # IR text still falls through and fails below.
        pytest.skip("pre-O3 dump is empty: IR text is concealed by design on this "
                    "build; the emitter-path observation is unavailable and the "
                    "proof stays unverified/blocked")

    problems = []
    groups = "\n".join(ln for ln in text.split("\n") if ln.startswith("attributes "))
    # Every private declaration in the dump that this leg's table lists.  A name
    # the table does not list is not ours to pin (device-library declarations),
    # and a name in the table that the module happens not to use is not a
    # failure either -- what has to hold is that whatever the emitters did
    # create came out with the facts, on whichever backend this leg runs.
    known = _table_names(LEG)
    emitted = sorted({
        m.group(1)
        for ln in text.split("\n") if ln.startswith("declare") for m in [_PRIVATE_DECL_RE.search(ln)] if m
        if m.group(1) in known
    })
    assert emitted, ("the translated module carries no private declaration the "
                     f"{LEG} table lists -- the kernel stopped exercising the "
                     "emitters")
    for name in emitted:
        decl = next((ln for ln in text.split("\n") if ln.startswith("declare") and f"@{name}(" in ln), None)
        if decl is None:
            problems.append(f"{name}: no declaration in the translated module")
            continue
        # The emitters write the atoms as an attribute group (`#N`); the group
        # text is where the tokens live, not the declare line.
        group_ref = decl.split("#")[-1].strip()
        group = next((ln for ln in groups.split("\n") if ln.startswith(f"attributes #{group_ref} = ")), None)
        if group is None:
            problems.append(f"{name}: declaration carries no attribute group")
            continue
        for token in _payload_tokens(name, LEG):
            if token.startswith("mem:"):
                spelling = "memory"
            else:
                spelling = _STAMP_EMIT_NATIVE.get(token, token)
            if spelling not in group:
                problems.append(f"{name}: table token {token!r} missing "
                                f"({spelling!r} not in {group.strip()[:90]})")
    assert not problems, ("the emitters did not stamp the declarations at "
                          "creation:\n  " + "\n  ".join(problems))


# The private declarations a translated module can carry, and the payload
# spelling -> LLVM IR text mapping for the atoms the emitters write.
#
# The gate derives the names from the dump instead of naming them: which
# private family a kernel emits is a property of the backend
# (`xpu3` emits `core_id`/`load_param`, the alternate backends emit
# `s.barrier`/`workitem.id.*`/`ds.bpermute`), so a hard-coded list quietly
# stopped testing anything on the leg it did not describe.
_PRIVATE_DECL_RE = re.compile(r"@(llvm\.(?:xcn|xpu)\.[A-Za-z0-9._]+)\(")
_STAMP_EMIT_NATIVE = {"nounwind": "nounwind"}


def _table_names(tag: str) -> set[str]:
    """The names this leg's shipped table has an entry for."""
    return {line.split(" ")[0] for line in it.stamp_payload(tag).split("\n") if line.strip()}


# The `synth` ledger-audit gate was removed from the public tree: it reads the
# vendor ledger (`intrinsic_bridge_registry.json`), which no longer ships here.
# It lives on in the internal tree's copy of this file, which still has the ledger.
def test_synth_gate_stamp_payload_is_the_table():
    """synth gate (ii): the payload the emitters and landing B' read is the table.

    The payload is built at runtime from the shipped table and read by C++ that
    cannot consult Python, so it is the one place where the two can drift without
    anything else noticing: a name left out of it is a declaration neither the
    emitters nor the B' sweep stamps, and a token the classifier does not know
    is an attribute the module carries while every gate stays green.  Both
    tables are checked --
    the LLVM 22 one packs four memory locations and has to normalise onto the
    three the LLVM 19 reader has.
    """
    _require_tables()
    problems: list[str] = []
    for tag, table in (("l19", T19), ("l22", T22)):
        covered: set[str] = set()
        for line in it.stamp_payload(tag).split("\n"):
            fields = line.split(" ")
            name, tokens = fields[0], fields[1:]
            covered.add(name)
            want = set(table.attributes(name))
            if not want:
                problems.append(f"{tag}/{name}: stamped although the table "
                                f"declares nothing for it")
                continue
            memory_tokens = [t for t in tokens if t.startswith("mem:")]
            declared = it.classify_attributes(" ".join(t for t in tokens if not t.startswith("mem:")))
            expected = {a for a in want if not a.startswith("memory(")}
            if declared != expected:
                problems.append(f"{tag}/{name}: payload spells {sorted(declared)}, "
                                f"the table says {sorted(expected)}")
            wants_memory = any(a.startswith("memory(") for a in want)
            if wants_memory != (len(memory_tokens) == 1):
                problems.append(f"{tag}/{name}: table memory {sorted(a for a in want if a.startswith('memory('))}, "
                                f"payload {memory_tokens}")
                continue
            if not memory_tokens:
                continue
            triple = memory_tokens[0][len("mem:"):].split("-")
            if len(triple) != 3 or any(not part.isdigit() or int(part) > 3 for part in triple):
                problems.append(f"{tag}/{name}: not a ModRefInfo triple: "
                                f"{memory_tokens[0]}")
            elif ("memory(none)" in want) != (triple == ["0", "0", "0"]):
                problems.append(f"{tag}/{name}: table says {sorted(want)}, the "
                                f"payload's locations are {triple}")
        unstamped = sorted(n for n in table.names if table.attributes(n))
        absent = [n for n in unstamped if n not in covered]
        if absent:
            problems.append(f"{tag}: {len(absent)} name(s) with a table attribute "
                            f"are not in the payload, e.g. {absent[:5]}")
        unknown = sorted(covered - set(table.names))
        if unknown:
            problems.append(f"{tag}: payload carries names the table lacks, "
                            f"e.g. {unknown[:5]}")
    assert not problems, (f"{len(problems)} payload/table disagreements:\n  " + "\n  ".join(problems[:12]))


def _translation_time_names() -> set[str]:
    """Names the LLVM-dialect -> LLVM IR *translation* creates, not the pass.

    The LLVMXPU/LLVMSDNN ops issue their `llvm.xpu.*` calls from their `.td`
    llvmBuilder (`createIntrinsicCallByName`), which runs while the module is
    translated to LLVM IR -- after every pass has run -- so no MLIR pass can
    stamp those declarations.  The read-in round is what covers them, and
    `attrs gate` is the gate that judges it.  Scanning the dialect definitions
    keeps this list current by construction.

    Measured on the recorded corpora: 28 names are derived; the default leg's
    `smoke_cache_w5` declares 8 of them and the stamp does leave 7 unstamped
    (`core_id`, `csr_set_sync_group`, `gm2lm_v3`, `gm2sm_v3`, `lm2gm_v3`,
    `load_param`, `mfence.v2` -- 5,236 declare instances), which is exactly the
    residual `synth gate (i)` is not allowed to blame anybody for; the pre-stamp
    corpus has 49 further names / 19,003 instances in scope, which keep this gate
    red.  The eighth, `llvm.xpu.log2f`, is the over-approximation: a `.td`
    literal whose call the pipeline creates *before* the pass (see
    `_STAMPED_DESPITE_EXCLUDED`).

    Re-measured 2026-09-18 on the 662-module corpus that includes
    `third_party/xpu/test/tle`: 16 derived names appear, 4 of them in *both*
    flavours (`core_id`, `lm2gm_v3`, `load_param`, `mfence.v2` -- which is what
    lets the width check run instead of skipping), and the stamp reaches 9 that
    no module leaves unstamped -- the TLE vector/DMA family, recorded below.
    """
    pattern = re.compile(r'"(llvm\.(?:xcn|xpu)\.[A-Za-z0-9._]*)"')
    names: set[str] = set()
    dialect = _repo_root() / "third_party/xpu/include/Dialect"
    for sub in ("LLVMXPU/IR", "LLVMSDNN/IR"):
        for td in sorted((dialect / sub).glob("*.td")):
            names.update(pattern.findall(td.read_text(errors="ignore")))
    return names


# Names the derivation above excludes although the stamp does reach them: the
# set is built from `.td` *mentions*, and a mention is not an emission.  Kept
# explicit so that `synth gate (i)` can tell the recorded over-approximation from
# a new one: a `.td` literal added (or renamed) into this category lands here
# only after someone says why, which is the difference between a derivable
# exclusion set and a widening one that quietly stops checking names.
#
# `log2f` was the pre-existing one (mentioned, never emitted).  The three below
# are reached *since the emitters stamp at creation*: the translation-time half of
# the gate (`intrinsic_attr_table::getOrCreateLLVMFunction`, which is what
# `createIntrinsicCallByName` calls instead of its own `Function::Create`) reads the
# table at creation, and landing B' sweeps the translated module, so a declaration
# the dialect's `llvmBuilder` creates during the translation -- which the MLIR pass
# structurally cannot reach -- now carries the table's facts in the emitted module.
# The derivation is about the *pass*,
# so these stay excluded there and are recorded here instead.
#
# The nine below are the same phenomenon, witnessed only once the TLE corpus
# entered the gate's scope (2026-09-18): the Gluon/TLE vector + DMA family
# (`vload_mz`, `vstore_mh`, `vscatter_mh`, `vmerge_{h,l}_hf`, `vshuffle2_hf`,
# `vvor_s_mh`) plus `gm2sm_v3` / `svsrlp_s` are declared *only* with the table's
# attributes in `third_party/xpu/test/tle`'s modules -- 662-module corpus, not
# one module leaves them unstamped -- where before the widening they simply never
# appeared, so the width check had nothing to judge.  Note `gm2sm_v3` shows the
# path dependence the check is about: the non-TLE corpus leaves it unstamped
# (see the measurement in `_translation_time_names`), the TLE one carries it.
_STAMPED_DESPITE_EXCLUDED = {
    "llvm.xpu.log2f",
    "llvm.xpu.csr_set_sync_group",
    "llvm.xpu.gm2lm_v3",
    "llvm.xpu.vvor_f_mh_rn",
    # TLE/Gluon vector + DMA declarations (2026-09-18, TLE corpus).
    "llvm.xpu.gm2sm_v3",
    "llvm.xpu.svsrlp_s",
    "llvm.xpu.vload_mz",
    "llvm.xpu.vmerge_h_hf",
    "llvm.xpu.vmerge_l_hf",
    "llvm.xpu.vscatter_mh",
    "llvm.xpu.vshuffle2_hf",
    "llvm.xpu.vstore_mh",
    "llvm.xpu.vvor_s_mh",
}


def _payload_tokens(name: str, tag: str) -> list[str]:
    for line in it.stamp_payload(tag).split("\n"):
        fields = line.split(" ")
        if fields[0] == name:
            return fields[1:]
    raise AssertionError(f"{name} is not in the {tag} stamp payload")


# ---------------------------------------------------------------------------
# fault gate: the gates must go red on a corrupted module
# ---------------------------------------------------------------------------


def _module_with_barrier() -> str:
    return ('target triple = "xcn-xcn-xcnpkg"\n'
            "declare i32 @llvm.xcn.s.barrier() #0\n"
            "define void @k(i1 %m) {\n"
            "  %a = tail call i32 @llvm.xcn.s.barrier() #0\n"
            "  ret void\n"
            "}\n"
            "attributes #0 = { convergent nounwind willreturn }\n")


def test_fault_gate_unknown_name_is_red():
    _require_tables()
    text = ('declare i32 @llvm.xcn.not_a_real_intrinsic(i32)\n'
            'define void @k() {\n  ret void\n}\n')
    assert it.unresolved_names(T19, text, GATE_NAMES.__contains__) == ["llvm.xcn.not_a_real_intrinsic"
                                                                       ], "an unknown name slipped through names gate"
    # and the shipped guard -- the thing that runs on the compile path --
    # raises on the same text
    from triton.backends import llvm19_toolchain
    with pytest.raises(RuntimeError, match="do not resolve"):
        llvm19_toolchain.check_intrinsic_names(text, "jupiter")


def test_fault_gate_free_form_suffix_is_red():
    _require_tables()
    text = 'declare i32 @llvm.xcn.ballot.bogus(i1)\n'
    assert it.unresolved_names(T19, text, GATE_NAMES.__contains__) == ["llvm.xcn.ballot.bogus"
                                                                       ], "a free-form suffix passed the name check"


def test_fault_gate_missing_convergent_is_red():
    _require_tables()
    text = ('declare i32 @llvm.xcn.s.barrier()\n'
            'define void @k() {\n'
            '  %a = tail call i32 @llvm.xcn.s.barrier()\n'
            '  ret void\n}\n')
    found = it.convergent_call_violations(T19, text)
    assert found and "convergent" in found[0], "a missing convergent stamp passed"


def test_fault_gate_bare_overloaded_name_is_red():
    _require_tables()
    text = 'declare i32 @llvm.xcn.ballot(i1)\n'
    assert it.bare_overloaded_names(T19, text) == ["llvm.xcn.ballot"], ("a bare overloaded name passed mangling gate")


def test_fault_gate_missing_declare_attribute_is_red():
    _require_tables()
    text = ('declare i32 @llvm.xcn.s.barrier() #0\n'
            'attributes #0 = { nounwind willreturn }\n')  # no `convergent`
    missing, extra = it.attribute_gaps(T19, text)
    assert missing.get("llvm.xcn.s.barrier") == [
        "Convergent"
    ], (f"a declare missing a table attribute passed attrs gate: {missing}")
    # and a declare that carries something the table does not declare
    text = ('declare i32 @llvm.xcn.s.barrier() #0\n'
            'attributes #0 = { convergent nounwind willreturn mustprogress }\n')
    missing, extra = it.attribute_gaps(T19, text)
    assert extra.get("llvm.xcn.s.barrier") == ["MustProgress"], (f"an over-stamped declare passed attrs gate: {extra}")


def test_fault_gate_injection_negative_controls():
    """The gates must not fire on the canonical spelling (no false red)."""
    _require_tables()
    text = _module_with_barrier()
    assert it.unresolved_names(T19, text, GATE_NAMES.__contains__) == []
    assert it.bare_overloaded_names(T19, text, GATE_NAMES.__contains__) == []
    assert it.convergent_call_violations(T19, text) == []
