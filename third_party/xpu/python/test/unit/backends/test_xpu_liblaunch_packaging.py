"""Regression: packaging must stage the liblaunch.a with try_launch_table.

FlagTree #1380: ensure_xpu_launch_static_lib used to copy
device/xpu3/liblaunch.a (from the xpu-sdnn-objects vendor cache), which has
no try_launch_table call site. After strip-all-ir, kernels then silently
no-op. The in-tree device/liblaunch.a carries the undefined ref; packaging
must prefer it, and the staged product must still show that symbol under nm.

CPU-only: stages into a temp backend dir and runs `nm --undefined-only`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "python" / "setup_tools" / "utils" / "xpu.py").is_file():
            return parent
    raise RuntimeError(f"cannot locate FlagTree root from {here}")


_REPO_ROOT = _find_repo_root()


def _load_xpu_utils():
    """Load setup_tools.utils.xpu without importing setup_tools.__init__.

    ``setup_tools`` pulls ``distutils`` via setup_helper; that module is gone on
    Python >= 3.12. This packaging check only needs the liblaunch helpers.
    """
    import importlib.util

    path = _REPO_ROOT / "python" / "setup_tools" / "utils" / "xpu.py"
    spec = importlib.util.spec_from_file_location("flagtree_xpu_utils_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_xpu = _load_xpu_utils()
ensure_xpu_launch_static_lib = _xpu.ensure_xpu_launch_static_lib
liblaunch_has_try_launch_table = _xpu.liblaunch_has_try_launch_table
resolve_xpu_launch_static_lib = _xpu.resolve_xpu_launch_static_lib

_GOOD = _REPO_ROOT / "third_party" / "xpu" / "device" / "liblaunch.a"


def _nm_undefined_has_try_launch_table(archive: Path) -> bool:
    out = subprocess.run(
        ["nm", "-C", "--undefined-only", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    return "try_launch_table" in (out.stdout or "")


@pytest.fixture(scope="module")
def good_archive():
    if not _GOOD.is_file():
        pytest.skip(f"in-tree {_GOOD} missing")
    if not liblaunch_has_try_launch_table(_GOOD):
        pytest.fail(f"in-tree {_GOOD} unexpectedly lacks try_launch_table")
    return _GOOD


def test_in_tree_device_liblaunch_has_try_launch_table(good_archive):
    assert _nm_undefined_has_try_launch_table(good_archive)


def test_resolve_prefers_device_over_xpu3_without_symbol(tmp_path, good_archive):
    """Simulate SDNN overlay: bad archive under device/xpu3/, good under device/."""
    src_dir = tmp_path / "third_party" / "xpu"
    device = src_dir / "device"
    xpu3 = device / "xpu3"
    xpu3.mkdir(parents=True)
    shutil.copy2(good_archive, device / "liblaunch.a")
    # Minimal archive with no try_launch_table (empty ar).
    bad = xpu3 / "liblaunch.a"
    bad.write_bytes(b"!<arch>\n")
    assert not liblaunch_has_try_launch_table(bad)

    chosen = resolve_xpu_launch_static_lib(src_dir)
    assert chosen == device / "liblaunch.a"
    assert liblaunch_has_try_launch_table(chosen)


def test_ensure_stages_archive_with_try_launch_table(tmp_path, good_archive):
    """ensure_xpu_launch_static_lib must stage a product nm still accepts."""
    src_dir = tmp_path / "src"
    backend_dir = tmp_path / "backend"
    device = src_dir / "device"
    xpu3 = device / "xpu3"
    xpu3.mkdir(parents=True)
    shutil.copy2(good_archive, device / "liblaunch.a")
    # Vendor-cache lookalike that packaging must NOT pick.
    (xpu3 / "liblaunch.a").write_bytes(b"!<arch>\n")

    backend = SimpleNamespace(src_dir=str(src_dir), backend_dir=str(backend_dir))
    ensure_xpu_launch_static_lib(backend)

    staged = backend_dir / "xpu3" / "lib" / "liblaunch.a"
    assert staged.is_file()
    assert staged.stat().st_size == good_archive.stat().st_size
    assert _nm_undefined_has_try_launch_table(staged)
    assert liblaunch_has_try_launch_table(staged)


def test_ensure_refuses_silent_bad_only_with_warning(tmp_path, capsys):
    """If only the no-call-site archive exists, still copy but warn (#1380)."""
    src_dir = tmp_path / "src"
    backend_dir = tmp_path / "backend"
    xpu3 = src_dir / "device" / "xpu3"
    xpu3.mkdir(parents=True)
    bad = xpu3 / "liblaunch.a"
    bad.write_bytes(b"!<arch>\n")

    backend = SimpleNamespace(src_dir=str(src_dir), backend_dir=str(backend_dir))
    ensure_xpu_launch_static_lib(backend)
    err = capsys.readouterr().err
    assert "try_launch_table" in err
    staged = backend_dir / "xpu3" / "lib" / "liblaunch.a"
    assert staged.is_file()
    assert not liblaunch_has_try_launch_table(staged)
