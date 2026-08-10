# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Generic .pyi stub generation for compiled pybind11 _C extension modules.

The compiled ``triton._C`` extension modules (``libtriton.so``, ``libproton.so``, ...)
exist only as binary build artifacts; the source tree has no ``.py`` for them, so
LSPs/type-checkers cannot resolve the C++-exposed surface. We regenerate annotation
stubs from the freshly built, importable module into the build tree (mirroring the
existing ``linear_layout.pyi`` pattern) so the toolchain can recognize it.

Runs as a post-build step keyed off the setuptools ``build_ext`` object. It is
intentionally generic and data-driven so it can be reused for any module that
exports a pybind11 surface (see :data:`DEFAULT_MODULES`).
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile

# Compiled pybind11 modules exposed to Python whose C++ surface has no
# source-level stub. Generate *leaf* submodules (e.g. ``triton._C.libtriton.tle``)
# rather than whole parents: stubgen flattens a parent's submodules into phantom
# top-level files that do not match the real nested import paths, while a leaf
# yields the faithful ``.../tle/{attr,ir,llvm,...}.pyi`` tree. Extend this tuple
# as new modules appear (per-backend main.cc/Proton.cpp variants, enflame, ...).
DEFAULT_MODULES = (
    "triton._C.libtriton.tle",  # attr enums (SignalOpKind, ...) + passes/llvm/raw_*
    "triton._C.libtriton.ir",  # TritonOpBuilder incl. TLE create_* extensions
    "triton._C.libproton",
)


def _trace(*a):
    print("[stubgen] " + " ".join(str(x) for x in a), flush=True)


_SHIM_ROOT = os.path.join(tempfile.gettempdir(), "triton-stubgen-shim")


def _make_shim(build_lib):
    """Build a minimal fake ``triton`` package that loads the compiled .so file
    directly. Importing the real ``triton/__init__.py`` from the build tree can
    hang outside the installed environment; a namespace shim sidesteps that and
    works for any backend-gated module."""
    shim = _SHIM_ROOT
    _c = os.path.join(shim, "triton", "_C")
    os.makedirs(_c, exist_ok=True)
    for pkg in ("triton/__init__.py", "triton/_C/__init__.py"):
        f = os.path.join(shim, *pkg.split("/"))
        if not os.path.exists(f):
            open(f, "w").close()
    src_c = os.path.join(build_lib, "triton", "_C")
    if os.path.isdir(src_c):
        for name in os.listdir(src_c):
            if name.endswith(".so"):
                ln = os.path.join(_c, name)
                src = os.path.abspath(os.path.join(src_c, name))
                # Refresh stale links from previous builds (the old target may
                # have been deleted or moved).
                if os.path.lexists(ln):
                    if os.path.exists(ln) and os.path.realpath(ln) == src:
                        continue
                    os.unlink(ln)
                os.symlink(src, ln)
    return shim


def _shim_env(build_lib, env):
    env = env.copy()
    shim = _make_shim(build_lib)
    env["PYTHONPATH"] = shim + os.pathsep + env.get("PYTHONPATH", "")
    # The compiled modules may link sibling .so files (e.g. libtriton.so links
    # libproton.so) that live beside them in triton/_C, so expose that dir to
    # the dynamic loader.
    _c_dir = os.path.join(build_lib, "triton", "_C")
    env["LD_LIBRARY_PATH"] = _c_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def _module_is_importable(module_name, build_lib, env):
    env = _shim_env(build_lib, env)
    try:
        probe = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            env=env,
            capture_output=True,
            timeout=30,
        )
        return probe.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[stubgen] {module_name}: import probe timed out; skipping")
        return False


def _emit_stubs_for(module_name, build_lib, env):
    staging = tempfile.mkdtemp(prefix="triton-stubgen-")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pybind11_stubgen",
                module_name,
                "-o",
                staging,
                "--ignore-all-errors",
            ],
            env=env,
        )
        rel = module_name.replace(".", os.sep)
        src = os.path.join(staging, rel)
        dst = os.path.join(build_lib, rel)
        if os.path.isdir(src):
            # Merge (not replace) so pre-seeded package stubs like
            # linear_layout.pyi are preserved; dirs_exist_ok only overwrites
            # the files stubgen emits.
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src + ".pyi"):
            # Leaf submodule: pybind11-stubgen emits a single <rel>.pyi file.
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src + ".pyi", dst + ".pyi")
        else:
            print(f"[stubgen] {module_name}: no stub tree emitted; skipping")
            return False
        print(f"[stubgen] wrote stubs for {module_name} -> {dst}")
        return True
    except FileNotFoundError as e:
        print(f"[stubgen] {module_name}: cannot run stub generator: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[stubgen] {module_name}: pybind11-stubgen failed: {e}")
        return False
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def generate_pybind_stubs(build_lib, modules=None):
    """Generate .pyi stubs for compiled _C extension modules into ``build_lib``.

    Args:
        build_lib: path to the setuptools ``build_lib`` dir (where the ``.so``
            and the ``triton`` package tree have been assembled).
        modules: iterable of module names (e.g. ``"triton._C.libtriton.tle"``).
            Defaults to :data:`DEFAULT_MODULES`.

    A module is skipped when ``pybind11-stubgen`` is not installed, when the
    ``TRITON_SKIP_STUBS`` env var is set, or when the module is not importable
    (e.g. backend-gated modules absent on non-TLE builds).
    """
    _trace(
        f"generate_pybind_stubs: build_lib={build_lib!r} skip_env={os.getenv('TRITON_SKIP_STUBS')!r} mods={modules or DEFAULT_MODULES!r}"
    )
    if os.getenv("TRITON_SKIP_STUBS"):
        _trace("  -> skipped: TRITON_SKIP_STUBS set")
        return
    if not build_lib:
        _trace("  -> skipped: build_lib empty")
        return
    try:
        import pybind11_stubgen  # noqa: F401  # presence probe
    except ImportError:
        _trace("pybind11-stubgen not installed; skipping")
        return

    env = _shim_env(build_lib, dict(os.environ))
    for module in modules or DEFAULT_MODULES:
        try:
            _trace(f"probe {module} ...")
            if _module_is_importable(module, build_lib, env):
                _trace("  -> importable, emitting")
                _emit_stubs_for(module, build_lib, env)
            else:
                _trace("  -> NOT importable; skipping")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"[stubgen] {module}: error: {e}")

    # pyright resolves members of compiled-submodule stubs only when the
    # namespace dirs carry __init__.pyi (a sibling .so otherwise makes the
    # submodule members Unknown).
    for pkg in ("triton/_C", "triton/_C/libtriton"):
        pkg_init = os.path.join(build_lib, *pkg.split(os.sep), "__init__.pyi")
        if not os.path.exists(pkg_init):
            os.makedirs(os.path.dirname(pkg_init), exist_ok=True)
            open(pkg_init, "w").close()
    _trace("wrote namespace __init__.pyi stubs under triton/_C")


_SEMANTIC_SOURCE = os.path.join("python", "triton", "experimental", "tle", "language", "gpu", "semantic.py")


def _scan_tle_semantic_methods(source_path):
    """Return the public FunctionDef nodes of ``class TLESemantic`` in
    ``source_path`` (dunders skipped; future public methods are picked up
    automatically)."""
    with open(source_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TLESemantic":
            return [
                item for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_")
            ]
    return []


def _function_signature(node):
    """Reconstruct ``def name(self, ...) -> ret: ...`` from a FunctionDef,
    keeping annotations verbatim (defaults omitted, body ``...``)."""
    args = []
    for a in node.args.args:
        if a.arg == "self":
            continue
        ann = ast.unparse(a.annotation) if a.annotation else None
        args.append(f"{a.arg}: {ann}" if ann else a.arg)
    if node.args.vararg:
        ann = ast.unparse(node.args.vararg.annotation) if node.args.vararg.annotation else None
        args.append("*" + node.args.vararg.arg + (f": {ann}" if ann else ""))
    for a in node.args.kwonlyargs:
        ann = ast.unparse(a.annotation) if a.annotation else None
        args.append(f"{a.arg}: {ann}" if ann else a.arg)
    if node.args.kwarg:
        ann = ast.unparse(node.args.kwarg.annotation) if node.args.kwarg.annotation else None
        args.append("**" + node.args.kwarg.arg + (f": {ann}" if ann else ""))
    ret = ast.unparse(node.returns) if node.returns else None
    sig = f"def {node.name}(self, " + ", ".join(args) + ")"
    if ret:
        sig += f" -> {ret}"
    return sig + ": ..."


def _annotation_names(node):
    """All Name ids referenced in a function's annotations."""
    names = set()
    for a in node.args.args:
        if a.annotation:
            names.update(n.id for n in ast.walk(a.annotation) if isinstance(n, ast.Name))
    for a in node.args.kwonlyargs:
        if a.annotation:
            names.update(n.id for n in ast.walk(a.annotation) if isinstance(n, ast.Name))
    if node.returns:
        names.update(n.id for n in ast.walk(node.returns) if isinstance(n, ast.Name))
    return names


def generate_semantic_stub(build_lib, repo_root=None):
    """Generate ``triton/experimental/tle/language/gpu/semantic.pyi``.

    The runtime ``TLESemantic`` (gpu/semantic.py) is a standalone class, but
    the builtins' ``_semantic`` parameter also dereferences the TritonSemantic
    surface (builder, to_tensor, ...). Rather than fabricating a module under
    ``triton.language``, emit a shadow stub next to the runtime module so the
    TLE semantic type lives in its own package; pyright prefers the .pyi over
    the .py, the runtime is unaffected. The public methods are scanned
    programmatically from gpu/semantic.py so the signatures never drift.
    Gated on the compiled TLE module being importable from the build tree.
    """
    _trace(f"generate_semantic_stub: build_lib={build_lib!r}")
    if os.getenv("TRITON_SKIP_STUBS"):
        _trace("  -> skipped: TRITON_SKIP_STUBS set")
        return
    if not build_lib:
        _trace("  -> skipped: build_lib empty")
        return
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_path = os.path.join(repo_root, *_SEMANTIC_SOURCE.split(os.sep))
    if not os.path.isfile(source_path):
        _trace(f"  -> skipped: {source_path} not found")
        return
    # Gate on the compiled TLE module (the C++ bindings exist only on TLE builds).
    env = _shim_env(build_lib, dict(os.environ))
    if not _module_is_importable("triton._C.libtriton.tle", build_lib, env):
        _trace("  -> skipped: triton._C.libtriton.tle not importable (non-TLE build)")
        return

    methods = _scan_tle_semantic_methods(source_path)
    if not methods:
        _trace("  -> skipped: no public methods found on TLESemantic")
        return

    import typing
    used_typing = sorted(n for n in set().union(*(_annotation_names(m) for m in methods)) if hasattr(typing, n))
    lines = [
        "# generated by python/setup_tools/stubgen.py - do not edit",
        "from triton.language.semantic import TritonSemantic",
        "import triton.language as tl",
        "import triton.experimental.tle.language as tle",
    ]
    if used_typing:
        lines.append("from typing import " + ", ".join(used_typing))
    lines += [
        "",
        "class TLESemanticError(Exception):",
        "    ...",
        "",
        "class TLESemantic(TritonSemantic):",
    ]
    for m in methods:
        lines.append("    " + _function_signature(m))
    lines.append("")

    out = os.path.join(build_lib, "triton", "experimental", "tle", "language", "gpu", "semantic.pyi")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _trace(f"wrote {out} ({len(methods)} methods)")

    # Remove the predecessor stub (fabricated triton/language/_semantic.pyi)
    # so stale build trees/wheels do not ship a dead module.
    legacy = os.path.join(build_lib, "triton", "language", "_semantic.pyi")
    if os.path.exists(legacy):
        os.remove(legacy)
        _trace(f"removed legacy {legacy}")


def auto_generate_stubs_from_install_extension(build_ext):
    """Hook wired around ``helper.install_extension``; ``build_ext`` carries ``build_lib``."""
    try:
        build_lib = getattr(build_ext, "build_lib", None)
    except Exception:
        build_lib = None
    generate_pybind_stubs(build_lib)
    generate_semantic_stub(build_lib)
