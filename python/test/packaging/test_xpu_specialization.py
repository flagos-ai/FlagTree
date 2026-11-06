"""Host-only checks: no backend downloads, libtriton, or GPU are required."""

import ast
import importlib
import importlib.machinery
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import types
import zipfile

import pytest
from setuptools import Distribution, find_packages
from setuptools.command.build_py import build_py

REPO = Path(__file__).resolve().parents[3]


def load_file(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def package(name, path, monkeypatch):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_definitions(path, names, namespace):
    # setup_helper has intentional SDK downloads at module scope. Test its
    # actual discovery definitions without executing build initialization.
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


def write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def xpu(monkeypatch):
    return load_file("xpu_setup_test", REPO / "python/setup_tools/utils/xpu.py", monkeypatch)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
            "python/triton/__init__.py",
            "python/triton/experimental/__init__.py",
            "python/triton/experimental/tle/__init__.py",
            "python/triton/experimental/tle/language/__init__.py",
            "python/triton/experimental/tle/language/main_only/__init__.py",
            "python/triton/spec/__init__.py",
            "third_party/xpu/spec/triton/__init__.py",
            "third_party/xpu/tle/__init__.py",
            "third_party/xpu/tle/language/__init__.py",
            "third_party/xpu/tle/language/gpu/__init__.py",
    ):
        write(tmp_path / name)
    write(tmp_path / "python/triton/experimental/tle/language/include/flagcx.h")
    return tmp_path


def discovery(xpu, backend):
    return load_definitions(
        REPO / "python/setup_tools/setup_helper.py",
        {"SpecPackageHelper", "get_spec_packages", "get_backend_python_packages"},
        {
            "os": os,
            "Path": Path,
            "find_packages": find_packages,
            "flagtree_backend": backend,
            "get_hook_instance": lambda name: getattr(xpu, name, None) if backend == "xpu" else None,
        },
    )


def test_discover_without_source_symlink(layout, xpu):
    helper = discovery(xpu, "xpu")
    packages = set(helper.get_spec_packages())
    assert "triton.spec.xpu.triton" in packages
    assert "triton.experimental.tle.language.gpu" in packages
    assert "triton.experimental.tle.language.main_only" not in packages
    assert "triton.experimental.tle.language.include" not in packages
    assert not (layout / "python/triton/spec/xpu").exists()
    assert not (layout / "third_party/xpu/python/triton").exists()


def test_other_backend_keeps_main_tle(layout, xpu):
    helper = discovery(xpu, "nvidia")
    packages = set(helper.get_spec_packages())
    assert "triton.experimental.tle.language.main_only" in packages
    assert "triton.experimental.tle.language.include" in packages
    assert "triton.spec.xpu.triton" not in packages
    assert list(helper.get_backend_python_packages()) == []


def test_stage_and_build_tle_with_binary_leaves(layout, xpu):
    cache = layout / "cache"
    binaries = (
        "triton/backends.abi3.so",
        "triton/experimental/tle.abi3.so",
        "triton/experimental/tle/language.abi3.so",
        "triton/experimental/tle/language/gpu.abi3.so",
        "triton/experimental/tle/language/gpu/core.abi3.so",
    )
    write(layout / "python/triton/backends/__init__.py")
    for name in binaries:
        write(cache / "python" / name, "test binary")
    write(cache / "python/triton/backends/_tables/l19.json", "{}")
    # Simulate a previous build that staged TLE into the wrong main package.
    old_binary = layout / "python/triton/experimental/tle/language/gpu/core.abi3.so"
    write(old_binary, "old binary")
    xpu.install_python_impl(cache, layout)
    xpu.install_python_impl(cache, layout)
    assert not old_binary.exists()
    assert not (layout / "python/triton/backends.abi3.so").exists()
    assert not (layout / "python/triton/experimental/tle.abi3.so").exists()
    tle = layout / xpu.XPU_TLE_ROOT
    assert (tle / "language/gpu/core.abi3.so").read_text() == "test binary"
    assert not (tle / "language/gpu.abi3.so").exists()
    assert (layout / "python/triton/backends/_tables/l19.json").is_file()

    helper = discovery(xpu, "xpu")
    setup = load_definitions(REPO / "setup.py", {"get_package_dirs"}, {
        "os": os,
        "helper": helper,
        "find_packages": find_packages,
        "FLAGPRISM_SETUP": types.SimpleNamespace(package_dirs=lambda: ()),
        "backends": [],
        "check_env_flag": lambda *args: False,
    })
    dirs = dict(setup.get_package_dirs())
    packages = [
        name for name in helper.get_spec_packages() if name not in
        {"triton._C", "triton._C.libtriton", "triton.tools.triton_to_gluon_translater", "triton.language.extra.xpu"}
    ]
    dist = Distribution({
        "name": "xpu-packaging-test",
        "packages": packages,
        "package_dir": dirs,
        "package_data": xpu.get_package_data([]),
    })
    dist.script_name = "setup.py"
    command = build_py(dist)
    command.build_lib = str(layout / "build")
    command.ensure_finalized()
    command.run()
    built_tle = layout / "build/triton/experimental/tle"
    assert (built_tle / "language/gpu/__init__.py").is_file()
    assert (built_tle / "language/gpu/core.abi3.so").is_file()
    assert not (built_tle / "language/main_only").exists()
    found = importlib.machinery.PathFinder.find_spec("core", [str(built_tle / "language/gpu")])
    assert found is not None
    assert Path(found.origin) == built_tle / "language/gpu/core.abi3.so"

    # Check wheel contents as well as build_py's staging tree.
    write(layout / "setup.py")
    dist.command_obj["build_py"] = command
    build = dist.get_command_obj("build")
    build.ensure_finalized()
    build.build_lib = command.build_lib
    wheel = dist.get_command_obj("bdist_wheel")
    wheel.dist_dir = str(layout / "dist")
    wheel.bdist_dir = str(layout / "wheel-build")
    wheel.skip_build = True
    wheel.ensure_finalized()
    wheel.run()
    with zipfile.ZipFile(next((layout / "dist").glob("*.whl"))) as archive:
        names = set(archive.namelist())
    assert "triton/experimental/tle/language/gpu/core.abi3.so" in names
    assert "triton/experimental/tle/language/gpu/__init__.py" in names
    assert "triton/spec/xpu/triton/__init__.py" in names
    assert not any("main_only" in name for name in names)


@pytest.fixture
def isolated_triton(monkeypatch):
    # Preserve any real triton modules if the test runs in an installed CI.
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name)
    root = package("triton", REPO / "python/triton", monkeypatch)
    package("triton.spec", REPO / "python/triton/spec", monkeypatch)
    package("triton.spec.xpu", REPO / "third_party/xpu/spec", monkeypatch)
    package("triton._C", REPO / "python/triton/_C", monkeypatch)
    lib = types.ModuleType("triton._C.libtriton")
    lib.getenv = lambda key, default=None: os.environ.get(key, default)
    lib.getenv_bool = lambda key, default=False: (os.environ[key].lower() in {"1", "true", "on", "yes"}
                                                  if key in os.environ else default)
    monkeypatch.setitem(sys.modules, lib.__name__, lib)
    backend = types.ModuleType("triton._flagtree_backend")
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    # Ensure all newly imported fixture modules are removed on teardown too.
    before = set(sys.modules)
    yield root, backend
    for name in set(sys.modules) - before:
        if name.startswith("triton."):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("backend_name", ["xpu", "nvidia", ""])
def test_shared_utils_and_knobs(isolated_triton, monkeypatch, backend_name):
    _root, backend = isolated_triton
    backend.FLAGTREE_BACKEND = backend_name
    monkeypatch.delenv("TRITON_STORE_BINARY_ONLY", raising=False)
    monkeypatch.setenv("TRITON_DISABLE_LAUNCH_FAST_PATH", "1")
    utils = importlib.import_module("triton._utils")
    knobs = importlib.import_module("triton.knobs")
    if backend_name == "xpu":
        assert utils.validate_block_shape([1536, 12]) == 18432
        assert utils.validate_block_shape([2**30]) == 2**30
        assert knobs.compilation.store_binary_only is True
        assert knobs.runtime.disable_launch_fast_path is True
    else:
        with pytest.raises(ValueError, match="power of 2"):
            utils.validate_block_shape([1536])
        assert knobs.compilation.store_binary_only is False
        assert not hasattr(knobs.runtime, "disable_launch_fast_path")
    with knobs.compilation.scope():
        knobs.compilation.store_binary_only = False
        assert knobs.compilation.store_binary_only is False
    assert knobs.compilation.store_binary_only is (backend_name == "xpu")
    monkeypatch.setenv("TRITON_STORE_BINARY_ONLY", "0")
    assert knobs.compilation.store_binary_only is False
    knobs.compilation.reset()
    assert knobs.compilation.store_binary_only is False


@pytest.mark.parametrize("backend_name,target,expected", [
    ("xpu", "cuda", True),
    ("xpu", "xpu", True),
    ("nvidia", "cuda", False),
    ("xpu", None, False),
])
def test_xpu_detection_is_backend_scoped(backend_name, target, expected):
    namespace = {
        "os": os,
        "FLAGTREE_BACKEND": backend_name,
        "get_current_target": lambda: types.SimpleNamespace(backend=target) if target else None,
    }
    module = load_definitions(REPO / "python/triton/_internal_testing.py", {"is_xpu"}, namespace)
    assert module.is_xpu() is expected


def test_xpu_benchmark_uses_active_runtime(isolated_triton, monkeypatch):
    root, backend = isolated_triton
    backend.FLAGTREE_BACKEND = "xpu"
    root.language = package("triton.language", REPO / "python/triton/language", monkeypatch)
    root.runtime = package("triton.runtime", REPO / "python/triton/runtime", monkeypatch)
    backends = types.ModuleType("triton.backends")
    backends.backends = {}
    monkeypatch.setitem(sys.modules, "triton.backends", backends)

    class Event:

        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

        def elapsed_time(self, other):
            return 0

    interface = types.SimpleNamespace(Event=Event, synchronize=lambda: None)
    root.runtime.driver = types.SimpleNamespace(active=types.SimpleNamespace(
        get_device_interface=lambda: interface,
        get_empty_cache_for_benchmark=lambda: None,
        clear_cache=lambda cache: None,
    ))
    testing = importlib.import_module("triton.testing")
    assert testing.do_bench(lambda: None, warmup=1, rep=1) == 0
    assert "n_retries" in inspect.signature(testing.do_bench_cudagraph).parameters


def test_no_root_path_override():
    for name in ("__init__.py", "experimental/__init__.py", "language/extra/__init__.py"):
        tree = ast.parse((REPO / "python/triton" / name).read_text())
        assert not any(isinstance(node, ast.Name) and node.id == "spec_path" for node in ast.walk(tree))
