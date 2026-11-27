"""Guard: the staged compiler-rt archives must land under the names the
backends resolve at compile time.

The compiler-rt staging in `setup.py` copies the resource-dir archives into
`third_party/<backend>/backend/lib/` (and, from there, into the wheel), and the
backend Python resolves them by *literal* name:

  * the backend's compiler-rt accessor                 -> lib/libclang_rt.builtins-<arch>.a
  * the backend's printf-runtime accessor              -> lib/libclang_rt.xcnprintf.a

(one archive per arch, because llc/llvm-mc stamp a per-arch `e_flags` that lld
refuses to mix -- see the staging note in setup.py).

Nothing used to check those names.  A commit that staged the printf runtime
under its resource-dir name (`libclang_rt.xcnprintf-**xcn**.a`) kept every
count-based check green -- the lib dir still held exactly four archives -- while
the consumer's bundled candidate became permanently absent: the lookup then
succeeded only on hosts whose `CUDA_PATH` carries a matching runtime (the CI and
simulator hosts do, which is why the smoke stages stayed green), and raised for
every kernel of the backend where it does not (the printf-runtime accessor is
called unconditionally in the bin stage, not only for printf kernels).  The
names below are therefore asserted, not counted.

Run with:  pytest python/test/unit/backends/test_backend_staging.py
"""

from pathlib import Path

import pytest

triton = pytest.importorskip("triton")

import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BACKENDS = ("xcn", "jupiter")
# the arch list setup.py stages, plus the single arch-independent printf
# runtime: the two formats that reach the two path accessors.
_ARCHES = ("xcn", "xpu5", "xpu6")
_EXPECTED = {f"libclang_rt.builtins-{arch}.a" for arch in _ARCHES} | {"libclang_rt.xcnprintf.a"}


@pytest.mark.parametrize("backend", _BACKENDS)
def test_staged_lib_holds_exactly_the_resolved_names(backend):
    """The staged directory (what setup.py will package) has no stray archive.

    Set equality on purpose: the pre-arch-suffixed `libclang_rt.builtins.a` this
    staging replaced is a *wrong variant* for two of the three archs, and a
    leftover copy in the build tree is how it kept reaching wheels.
    """
    lib = _REPO_ROOT / "third_party" / backend / "backend" / "lib"
    if not lib.is_dir():
        pytest.skip(f"{lib} is not staged in this tree (run setup.py's staging)")
    assert {p.name for p in lib.glob("libclang_rt.*.a")} == _EXPECTED


@pytest.mark.parametrize("backend", _BACKENDS)
def test_bundled_printf_runtime_resolves_without_cuda_path(backend, monkeypatch):
    """The installed backend must find its own printf runtime.

    This is the deployment case: no `CUDA_PATH` runtime to glob and no
    printf-runtime path override, so the bundled candidate is the only one
    left.  A missing/wrongly named archive raises here for every arch instead of
    failing later, and only for the kernels that actually call printf.
    """
    module = pytest.importorskip(f"triton.backends.{backend}.compiler")
    lib = Path(module.__file__).parent / "lib"
    if not lib.is_dir():
        pytest.skip(f"{lib} is not staged in this installation")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("TRITON_XCN_PRINTF_LIB", raising=False)

    for arch in _ARCHES:
        assert module.HOUYIBackend.path_to_xcn_crt(arch) == lib / f"libclang_rt.builtins-{arch}.a"
        assert module.HOUYIBackend.path_to_xcn_printf_lib(arch) == lib / "libclang_rt.xcnprintf.a"


# ---------------------------------------------------------------------------
# The staged LLVM 19 runtime: staged once, and every tool resolving *it*
# ---------------------------------------------------------------------------

_LLVM19_SHLIBS = ("libLLVM.so.19.1", "libclang-cpp.so.19.1")
_SHARED_PARTS = ("third_party", "_xtdk_llvm19", "lib")
_TOOL_DIR_TEMPLATES = (
    "third_party/{backend}/backend/llvm19/bin",
    "third_party/xpu/backend/{arch}/bin",
    "third_party/{backend}/backend/bin",
)


def _scan_roots():
    roots = [_REPO_ROOT / "third_party"]
    roots += list(_REPO_ROOT.glob("build/lib*/triton/backends"))
    return [r for r in roots if r.is_dir()]


def test_the_llvm19_runtime_is_staged_once_not_per_backend():
    """No libLLVM/libclang-cpp outside `_xtdk_llvm19/lib`.

    They used to be copied into six places -- each backend's `llvm19/lib` plus
    the three `xpu/backend/xpu{3,4,5}/lib` -- byte-identical copies that were
    383 MB of the wheel.  A copy reappearing anywhere else is what this catches;
    the derived `build/lib*` copy that gets packaged is legitimate.
    """
    found = {}
    for root in _scan_roots():
        for name in _LLVM19_SHLIBS:
            found.setdefault(name, []).extend(p for p in root.rglob(name) if p.is_file())
    for name, paths in found.items():
        stray = [p for p in paths if "_xtdk_llvm19" not in p.parts]
        assert not stray, (f"{name} staged outside the shared dir: " + ", ".join(str(p) for p in stray))
    staged = _REPO_ROOT.joinpath(*_SHARED_PARTS)
    if not staged.is_dir():
        pytest.skip(f"{staged} is not staged in this tree (run setup.py's staging)")
    for name in _LLVM19_SHLIBS:
        assert (staged / name).is_file(), f"shared runtime incomplete: {staged / name}"


def _elf_tools():
    dirs = []
    for tpl in _TOOL_DIR_TEMPLATES:
        if "{arch}" in tpl:
            dirs += [_REPO_ROOT / tpl.format(arch=a) for a in ("xpu3", "xpu4", "xpu5")]
        else:
            dirs += [_REPO_ROOT / tpl.format(backend=b) for b in ("xpu", )]
    for d in dirs:
        if not d.is_dir():
            continue
        backend = "xpu"
        for tool in sorted(d.iterdir()):
            if not tool.is_file():
                continue
            with open(tool, "rb") as fh:
                if fh.read(4) == b"\x7fELF":
                    yield backend, tool


def _needed(path):
    out = subprocess.run(["readelf", "-d", str(path)], capture_output=True, text=True).stdout
    return set(re.findall(r"Shared library: \[([^\]]+)\]", out))


def _resolved(tool, env):
    e = dict(env)
    e["LD_TRACE_LOADED_OBJECTS"] = "1"
    out = subprocess.run([str(tool)], capture_output=True, text=True, env=e).stdout
    # "libLLVM.so.19.1 => not found" must not come back as a path: the caller
    # asserts on the resolved location, and a relative string would slip through.
    return {k: v for k, v in re.findall(r"^\s*(\S+)\s+=>\s+(\S+)", out, re.M) if v != "not"}


_ENV_KINDS = ("bare", "ambient", "injected")


def _tool_env(backend, kind):
    """Environment to run a staged tool in.

    `bare` -- no `LD_LIBRARY_PATH` at all: only the tool's own RPATH can
    resolve, which is what the patchelf step in setup.py buys (and what a
    clean-room host without an ambient libLLVM looks like).
    `ambient` -- the caller's environment as-is: what a call site that forgets
    `llvm19_env` hands the tool.  The XPU runtime puts an *older* libLLVM on
    `LD_LIBRARY_PATH`, so the RPATH has to win here too.
    `injected` -- `llvm19_toolchain.llvm19_env`, the fallback for hosts without
    patchelf.
    """
    from triton.backends import llvm19_toolchain

    if kind == "injected":
        return llvm19_toolchain.llvm19_env(backend)
    env = dict(os.environ)
    if kind == "bare":
        env.pop("LD_LIBRARY_PATH", None)
    return env


@pytest.mark.parametrize("env_kind", _ENV_KINDS)
def test_staged_llvm19_tools_resolve_the_staged_libllvm(env_kind):
    """Every staged ELF tool loads the *staged* libLLVM, never an ambient one.

    This is not academic: `<backend>/bin/ld.lld` (what `path_to_xtdk_lld`
    returns) had no staged copy on its RUNPATH and resolved
    `/usr/local/xcuda/lib64/libLLVM.so.19.1` -- the same soname from a different
    build, which silently leaves the private intrinsics unlowered and only
    shows up later as an `undefined symbol` at kernel link time.
    """
    checked = 0
    for backend, tool in _elf_tools():
        want = _needed(tool) & set(_LLVM19_SHLIBS)
        if not want:
            continue
        resolved = _resolved(tool, _tool_env(backend, env_kind))
        for lib in sorted(want):
            path = resolved.get(lib)
            assert path is not None, f"{tool}: {lib} did not resolve ({env_kind} env)"
            assert str(_REPO_ROOT) in os.path.realpath(path), (
                f"{tool} resolves {lib} to {path} with a {env_kind} env, outside "
                f"the staged tree -- an ambient libLLVM of the same soname would be used")
        checked += 1
    if not checked:
        pytest.skip("no staged LLVM 19 tool binaries in this tree")


# ---------------------------------------------------------------------------
# The *installed* layout: what every wheel user gets
# ---------------------------------------------------------------------------
#
# `build_py` re-homes `third_party/<b>/backend/<...>` to `triton/backends/<b>/<...>`,
# so the same tool sits **one level shallower** once packaged.  An
# `$ORIGIN`-relative RPATH computed from the source-tree depth therefore names a
# directory that does not exist in any install -- the loader falls through to
# `LD_LIBRARY_PATH`, i.e. to whichever ambient libLLVM happens to be there, or to
# nothing at all.  Scanning `_REPO_ROOT` (above) cannot see that: its depth is
# the source one.  These two tests check the packaged depth instead -- the first
# without needing an install, the second against the package this interpreter
# actually imported.

_INSTALLED_TOOL_DIRS = (
    "backends/{backend}/llvm19/bin",
    "backends/xpu/{arch}/bin",
    "backends/{backend}/bin",
)
_PKG_ROOT = Path(triton.__file__).resolve().parent


def _installed_counterpart(source_dir: Path) -> Path:
    """Where `source_dir` lives in the installed layout (see setup.py::_installed_layout_path)."""
    parts = source_dir.relative_to(_REPO_ROOT / "third_party").parts
    if len(parts) > 1 and parts[1] == "backend":
        parts = parts[:1] + parts[2:]
    return _PKG_ROOT / "backends" / Path(*parts)


def _rpath_entries(tool: Path):
    out = subprocess.run(["readelf", "-d", str(tool)], capture_output=True, text=True).stdout
    m = re.search(r"\((?:RPATH|RUNPATH)\)\s+Library (?:rpath|runpath): \[([^\]]*)\]", out)
    return [e for e in m.group(1).split(":") if e] if m else []


def _entries_reaching(tool_dir: Path, tool: Path):
    """RPATH entries that, from `tool_dir`, land on the staged shared runtime."""
    hits = []
    for entry in _rpath_entries(tool):
        if not entry.startswith("$ORIGIN/"):
            continue
        target = tool_dir / entry[len("$ORIGIN/"):]
        if all((target / name).is_file() for name in _LLVM19_SHLIBS):
            hits.append(entry)
    return hits


def test_staged_tools_carry_an_rpath_that_works_in_the_installed_layout():
    """A tool whose RPATH only covers the source depth is dead in every wheel.

    The check is per *directory* (the depth is a property of the directory, not
    of the tool): from the source directory the entry must reach the staged
    runtime, and from the installed counterpart it must too.  `$ORIGIN` follows
    the path a tool was started with, and both spellings do occur -- backends
    start some through the package symlink -- so both have to resolve.  Entries
    naming a missing directory are simply skipped by the loader, which is why
    carrying both is safe.
    """
    checked = 0
    for backend, tool in _elf_tools():
        if not (_needed(tool) & set(_LLVM19_SHLIBS)):
            continue
        source_dir, installed_dir = tool.parent, _installed_counterpart(tool.parent)
        assert _entries_reaching(source_dir,
                                 tool), (f"{tool}: no RPATH entry reaches the staged runtime from {source_dir} "
                                         f"(entries: {_rpath_entries(tool)})")
        assert _entries_reaching(installed_dir,
                                 tool), (f"{tool}: no RPATH entry reaches the staged runtime from the installed "
                                         f"layout {installed_dir} (entries: {_rpath_entries(tool)}) -- a wheel "
                                         f"built this way loads an ambient libLLVM or fails to load the tool")
        checked += 1
    if not checked:
        pytest.skip("no staged LLVM 19 tool binaries in this tree")


def _claimed_installed_tools():
    """(backend, tool) for the staged tools of the *imported* triton package."""
    for tpl in _INSTALLED_TOOL_DIRS:
        if "{arch}" in tpl:
            dirs = [_PKG_ROOT / tpl.format(arch=a) for a in ("xpu3", "xpu4", "xpu5")]
        else:
            dirs = [_PKG_ROOT / tpl.format(backend=b) for b in ("xpu", )]
        for d in dirs:
            if not d.is_dir():
                continue
            backend = "xpu"
            for tool in sorted(d.iterdir()):
                if not tool.is_file():
                    continue
                with open(tool, "rb") as fh:
                    if fh.read(4) == b"\x7fELF":
                        yield backend, tool


@pytest.mark.parametrize("env_kind", _ENV_KINDS)
def test_installed_tools_resolve_the_staged_libllvm(env_kind):
    """The imported package -- the wheel, in every install -- resolves the staged runtime.

    Unlike the source-tree scans above, this needs the layout that is actually
    shipped: `python/triton/backends/<b>/<...>`, one level shallower than the
    staging directories.
    """
    checked = 0
    for backend, tool in _claimed_installed_tools():
        want = _needed(tool) & set(_LLVM19_SHLIBS)
        if not want:
            continue
        resolved = _resolved(tool, _tool_env(backend, env_kind))
        for lib in sorted(want):
            path = resolved.get(lib)
            assert path is not None, f"{tool}: {lib} did not resolve ({env_kind} env)"
            assert (str(_PKG_ROOT) in os.path.realpath(path) or str(_REPO_ROOT)
                    in os.path.realpath(path)), (f"{tool} resolves {lib} to {path} with a {env_kind} env, outside the "
                                                 f"staged package tree")
        checked += 1
    if not checked:
        pytest.skip("no staged LLVM 19 tools in the imported package "
                    f"({_PKG_ROOT}); run the gates stage against an installed wheel")
