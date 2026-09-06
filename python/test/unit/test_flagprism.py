# FlagPrism: test the host integration contract without requiring a device.
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from python.setup_tools import setup_helper  # noqa: E402
from flagtree import _flagprism  # noqa: E402


def _load_build_helper():
    path = (Path(__file__).resolve().parents[3] / "third_party" / "FlagPrism" / "python" / "flagprism_build.py")
    if not path.is_file():
        pytest.skip("FlagPrism sources are not available")
    spec = importlib.util.spec_from_file_location("_test_flagprism_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_helper():
    return _load_build_helper()


@pytest.fixture
def flagprism_setup_factory(monkeypatch, tmp_path):
    helper_path = tmp_path / "third_party" / "FlagPrism" / "python" / "flagprism_build.py"
    helper_path.parent.mkdir(parents=True)
    helper_path.touch()

    downloads = []
    build_config = SimpleNamespace(
        cmake_args=lambda build_lib: ["-DTRITON_BUILD_FLAGPRISM=ON"],
        packages=lambda: ("flagtree.debugger", "flagtree.profiler"),
        package_dirs=lambda: (),
        console_scripts=lambda: [],
    )
    monkeypatch.setattr(
        setup_helper,
        "download_flagtree_third_party",
        lambda *args, **kwargs: downloads.append((args, kwargs)),
    )
    monkeypatch.setattr(
        setup_helper.runpy,
        "run_path",
        lambda *args, **kwargs: {
            "create_build_config": lambda project_root: build_config,
        },
    )

    def create(backend):
        monkeypatch.setattr(setup_helper.configs, "flagtree_backend", backend)
        return setup_helper.FlagPrismSetup(tmp_path, lambda build_ext: ["dependency"])

    return create, downloads


@pytest.fixture(autouse=True)
def isolated_components():
    previous = dict(_flagprism._components)
    _flagprism._components.clear()
    try:
        yield
    finally:
        _flagprism._components.clear()
        _flagprism._components.update(previous)


def _component(name, **attributes):
    values = {
        "name": name,
        "api_version": _flagprism.HOST_API_VERSION,
    }
    values.update(attributes)
    return SimpleNamespace(**values)


def test_public_component_modules_use_flagtree_namespace():
    assert _flagprism._COMPONENT_MODULES == {
        "debugger": "flagtree.debugger",
        "profiler": "flagtree.profiler",
    }


def test_language_extensions_are_owned_only_by_flagtree():
    import flagtree.language as ftl
    import triton.language as tl

    assert ftl.debug_collect_start.__triton_builtin__ is True
    assert ftl.debug_collect_end.__triton_builtin__ is True
    assert not hasattr(tl, "debug_collect_start")
    assert not hasattr(tl, "debug_collect_end")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("triton._flagprism")


def test_language_extensions_forward_to_registered_component():
    import flagtree.language as ftl

    calls = []
    component = _component(
        "debugger",
        debug_collect_start=lambda semantic, level, addr_level: calls.append(("start", semantic, level, addr_level)),
        debug_collect_end=lambda semantic: calls.append(("end", semantic)),
    )
    _flagprism.register_component("debugger", component)

    semantic = object()
    ftl.debug_collect_start(level=2, addr_level=1, _semantic=semantic)
    ftl.debug_collect_end(_semantic=semantic)

    assert calls == [
        ("start", semantic, 2, 1),
        ("end", semantic),
    ]


@pytest.mark.parametrize(
    "component",
    ("debugger", "profiler"),
)
def test_missing_component_has_build_instruction(monkeypatch, component):

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(_flagprism, "import_module", missing)
    with pytest.raises(_flagprism.ComponentNotInstalledError) as error:
        _flagprism.load_component(component)
    assert "TRITON_BUILD_FLAGPRISM=ON" in str(error.value)


@pytest.mark.parametrize(
    ("value", "enabled"),
    ((None, True), ("ON", True), ("OFF", False)),
)
def test_build_helper_uses_unified_switch(build_helper, monkeypatch, tmp_path, value, enabled):
    if value is None:
        monkeypatch.delenv("TRITON_BUILD_FLAGPRISM", raising=False)
    else:
        monkeypatch.setenv("TRITON_BUILD_FLAGPRISM", value)

    config = build_helper.FlagPrismBuildConfig.from_environment(tmp_path)
    assert config.enabled is enabled


@pytest.mark.parametrize(
    ("backend", "enabled"),
    (
        (None, False),
        ("ascend", True),
        ("iluvatar", True),
        ("enflame", False),
        ("tsingmicro", False),
        ("cambricon", False),
        ("aipu", False),
        ("xpu", False),
        # FlagPrism: retain the former unsupported expectation for reference.
        # ("mthreads", False),
        # FlagPrism: mthreads now enables the profiler/debugger by default.
        ("mthreads", True),
    ),
)
def test_flagprism_is_enabled_by_default_for_supported_backends(flagprism_setup_factory, monkeypatch, backend, enabled):
    create, downloads = flagprism_setup_factory
    monkeypatch.delenv("TRITON_BUILD_FLAGPRISM", raising=False)
    monkeypatch.delenv("TRITON_BUILD_PROTON", raising=False)

    policy = create(backend)

    assert policy.enabled is enabled
    assert not downloads
    assert policy.cmake_args("build") == ["-DTRITON_BUILD_FLAGPRISM=" + ("ON" if enabled else "OFF")]
    if enabled:
        assert setup_helper.os.environ["TRITON_BUILD_PROTON"] == "OFF"
    else:
        assert "TRITON_BUILD_PROTON" not in setup_helper.os.environ
        assert policy.dependency_cmake_args(object()) == []
        assert policy.packages() == ()
        assert policy.package_dirs() == ()
        assert policy.console_scripts() == []


def test_ascend_can_explicitly_disable_flagprism_without_changing_proton(flagprism_setup_factory, monkeypatch):
    create, downloads = flagprism_setup_factory
    monkeypatch.setenv("TRITON_BUILD_FLAGPRISM", "OFF")
    monkeypatch.setenv("TRITON_BUILD_PROTON", "ON")

    policy = create("ascend")

    assert not policy.enabled
    assert not downloads
    assert setup_helper.os.environ["TRITON_BUILD_PROTON"] == "ON"


def test_non_ascend_explicit_flagprism_is_rejected_before_side_effects(flagprism_setup_factory, monkeypatch):
    create, downloads = flagprism_setup_factory
    monkeypatch.setenv("TRITON_BUILD_FLAGPRISM", "ON")
    monkeypatch.setenv("TRITON_BUILD_PROTON", "ON")

    # FlagPrism: retain the former diagnostic assertion for reference.
    # with pytest.raises(RuntimeError, match="ascend or iluvatar"):
    #     create("enflame")
    # FlagPrism: include mthreads in the supported-backend diagnostic.
    with pytest.raises(RuntimeError, match="ascend, iluvatar, or mthreads"):
        create("enflame")

    assert not downloads
    assert setup_helper.os.environ["TRITON_BUILD_PROTON"] == "ON"


def test_ascend_rejects_flagprism_and_proton_together(flagprism_setup_factory, monkeypatch):
    create, downloads = flagprism_setup_factory
    monkeypatch.delenv("TRITON_BUILD_FLAGPRISM", raising=False)
    monkeypatch.setenv("TRITON_BUILD_PROTON", "ON")

    with pytest.raises(RuntimeError, match="cannot both be enabled"):
        create("ascend")

    assert not downloads
    assert setup_helper.os.environ["TRITON_BUILD_PROTON"] == "ON"


def test_reused_build_tree_drops_legacy_triton_gateway(flagprism_setup_factory, tmp_path):
    create, _ = flagprism_setup_factory
    policy = create(None)
    build_lib = tmp_path / "build-lib"
    legacy_module = build_lib / "triton" / "_flagprism.py"
    legacy_cache = (build_lib / "triton" / "__pycache__" / "_flagprism.cpython-311.pyc")
    for path in (legacy_module, legacy_cache):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    policy.prepare_build_tree(str(build_lib))

    assert not legacy_module.exists()
    assert not legacy_cache.exists()


def test_known_component_is_loaded_once(monkeypatch):
    component = _component("debugger")
    module = SimpleNamespace(component=component)
    loads = []

    def import_component(name):
        loads.append(name)
        return module

    monkeypatch.setattr(_flagprism, "import_module", import_component)
    assert _flagprism.load_component("debugger") is component
    assert _flagprism.load_component("debugger") is component
    assert loads == ["flagtree.debugger"]


def test_component_api_mismatch_is_rejected():
    component = _component("debugger", api_version=999)
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
        _flagprism.register_component("debugger", component)


def test_legacy_callback_api_is_rejected():
    component = _component("debugger", api_version=1)
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
        _flagprism.register_component("debugger", component)


def test_component_api_minor_newer_than_host_is_rejected():
    component = _component("debugger", api_version=(2, 1))
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="API mismatch"):
        _flagprism.register_component("debugger", component)


def test_component_compatibility_uses_capabilities_not_core_series():
    component = _component(
        "debugger",
        api_version=(2, 0),
        core_version_series="0.0",
        required_capabilities={"compiler.events.v1"},
    )
    assert _flagprism.register_component("debugger", component) is component


def test_missing_host_capability_is_rejected():
    component = _component("debugger", required_capabilities={"runtime.future_adapter.v1"})
    with pytest.raises(
            _flagprism.ComponentCompatibilityError,
            match="runtime.future_adapter.v1",
    ):
        _flagprism.register_component("debugger", component)


def test_optional_hooks_are_noops_until_a_component_registers():
    events = []
    component = _component(
        "debugger",
        apply_compile_options=lambda options: options.update(instrumentation_mode="debug"),
        on_compiler_event=events.append,
        on_statement_event=events.append,
    )
    options = {}
    _flagprism.apply_compile_options(options)
    assert options == {}

    _flagprism.register_component("debugger", component)
    _flagprism.apply_compile_options(options)
    metadata = {
        "key": "value",
        "target": SimpleNamespace(backend="cuda"),
    }
    _flagprism.emit_compiler_event(
        phase="post_override",
        ir_kind="ttir",
        module="module",
        metadata=metadata,
    )
    node = ast.parse("result = value").body[0]
    generator = SimpleNamespace(
        begin_line=20,
        builder="builder",
        jit_fn=SimpleNamespace(src="result = value"),
    )
    _flagprism.emit_statement_event("assignment", generator, node, node.targets[0], "value")

    assert options == {"instrumentation_mode": "debug"}
    compiler_event, statement_event = events
    assert compiler_event == _flagprism.CompilerEvent(
        phase="post_override",
        ir_kind="ttir",
        backend="cuda",
        module="module",
        metadata=metadata,
    )
    assert statement_event.kind == "assignment"
    assert statement_event.source == "result = value"
    assert statement_event.statement_id == 21000
    assert statement_event.builder == "builder"
    assert statement_event.results == (_flagprism.StatementResult(name="result", value="value"), )


def test_statement_normalization_is_skipped_without_a_consumer():
    _flagprism.register_component("profiler", _component("profiler"))
    _flagprism.emit_statement_event("assignment", "not-a-generator", "not-an-ast-node", None, "value")


def test_required_backend_neutral_launch_context_is_forwarded():
    context = nullcontext((123, ))
    events = []

    def launch_context(event):
        events.append(event)
        return context

    component = _component(
        "debugger",
        launch_context=launch_context,
    )
    _flagprism.register_component("debugger", component)

    result = _flagprism.debugger_launch_context("CANN", "metadata", (1, 2, 3), "stream", "launch_metadata", ("arg", ))
    assert result is context
    assert events == [
        _flagprism.LaunchEvent(
            backend="cann",
            metadata="metadata",
            grid=(1, 2, 3),
            stream="stream",
            launch_metadata="launch_metadata",
            kernel_args=("arg", ),
        )
    ]


def test_unknown_components_are_rejected():
    with pytest.raises(_flagprism.ComponentCompatibilityError, match="unsupported"):
        _flagprism.load_component("custom")


@pytest.mark.parametrize("enabled", (True, False))
def test_build_tree_cleanup_prevents_split_wheel_artifacts(build_helper, tmp_path, enabled):
    build_lib = tmp_path / "build-lib"
    triton_root = build_lib / "triton"
    flagtree_root = build_lib / "flagtree"
    native_root = triton_root / "_C"
    cache_root = triton_root / "__pycache__"
    config = build_helper.FlagPrismBuildConfig(
        enabled=enabled,
        relative_root=Path("third_party/FlagPrism"),
        root=tmp_path / "FlagPrism",
    )

    for path in (
            triton_root / "debugger" / "old.py",
            build_lib / "flagtree_debugger" / "old.py",
            flagtree_root / "debugger" / "old.py",
            native_root / "libproton.so",
            cache_root / "_components.cpython-311.pyc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    config.prepare_build_tree(str(build_lib))
    assert not (triton_root / "debugger").exists()
    assert not (flagtree_root / "debugger").exists()
    assert not list(native_root.glob("libproton*"))

    expected_native = flagtree_root / "profiler" / ("_native" + (sysconfig.get_config_var("EXT_SUFFIX") or ".so"))
    if enabled:
        for path in (
                flagtree_root / "debugger" / "__init__.py",
                flagtree_root / "profiler" / "__init__.py",
                expected_native,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"current")

    # build_py can copy these stale source-tree files after CMake completes.
    for path in (
            native_root / "libproton.so",
            cache_root / "_components.cpython-311.pyc",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    config.finalize_build_tree(str(build_lib))
    assert not (cache_root / "_components.cpython-311.pyc").exists()
    if enabled:
        assert expected_native.is_file()
        assert not list(native_root.glob("libproton*"))
        assert (flagtree_root / "debugger").is_dir()
        assert (flagtree_root / "profiler").is_dir()
    else:
        assert not list(native_root.glob("libproton*"))
        assert not (flagtree_root / "debugger").exists()
        assert not (flagtree_root / "profiler").exists()
