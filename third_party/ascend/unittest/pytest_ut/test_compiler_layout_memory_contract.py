# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
"""Source-level contracts for the layout / memory-access compiler closure.

These tests intentionally load ``backend/compiler.py`` from this checkout.
The installed Triton package can point at another worktree, so importing
``triton.backends.ascend.compiler`` directly would not validate the source
being changed here.
"""

import ast
import importlib.util
import itertools
import json
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")

_UNSET = object()


def _stub_ub_size_in_kbytes_for_arch(arch):
    """Mirror the raw UB architecture table for the compiler import shim.

    The table itself is covered by the source-loaded backend-utils test.  This
    local shim keeps this compiler-only contract independent of the installed
    Ascend package while retaining meaningful architecture-budget assertions.
    """
    if not isinstance(arch, str) or not arch:
        return 0
    if arch.startswith(("Ascend910_95", "Ascend950")):
        return 256
    if arch.startswith((
            "Ascend910A",
            "Ascend910B",
            "Ascend910D",
            "Ascend910_93",
            "Ascend310B",
    )):
        return 192
    return 0


def _stub_graph_ub_budget_bytes_for_arch(arch):
    return _stub_ub_size_in_kbytes_for_arch(arch) * 1024 * 80 // 100


class _FakeModule:

    def __init__(self, events):
        self.context = object()
        self._events = events
        self._string_count = 0

    def __str__(self):
        self._events.append(f"str:{self._string_count}")
        self._string_count += 1
        return "module {}"


class _FakePassManager:

    def __init__(self, events):
        self._events = events

    def enable_debug(self):
        self._events.append("enable_debug")

    def run(self, _module, _pipeline_name):
        self._events.append("run_row")


@pytest.fixture(scope="module")
def compiler_module():
    """Load this checkout's compiler without depending on installed Ascend utils.

    The test invokes only ``ttir_to_npubin`` and replaces all external tool
    interactions below.  A tiny import-time shim keeps the source-level
    contract test runnable when the installed Triton wheel predates an import
    added by this checkout (for example ``_enable_msdebug``).
    """
    compiler_path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
    module_name = "triton.backends.ascend.compiler_layout_memory_contract_under_test"
    utils_name = "triton.backends.ascend.utils"
    driver_name = "triton.backends.ascend.driver"
    cache_name = "triton.runtime.cache"

    def return_false(*_args, **_kwargs):
        return False

    def remove_deprecated_npu_options(options, *, in_place=False):
        normalized = options if in_place else dict(options)
        normalized.pop("compile_on_910_95", None)
        return normalized

    utils_stub = types.ModuleType(utils_name)
    utils_path = compiler_path.with_name("utils.py")
    utils_tree = ast.parse(utils_path.read_text())
    mode_helper = next(node for node in utils_tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "_multibuffer_mode_to_tuple")
    exec(compile(ast.Module(body=[mode_helper], type_ignores=[]), str(utils_path), "exec"), utils_stub.__dict__)
    for name in (
            "_check_bishengir_api_change",
            "_check_bishengir_able_save_ir",
            "_check_bishengir_is_regbased",
            "_enable_print_ub_bits",
            "_enable_dump_memory_info",
            "_enable_msdebug",
            "_is_ascend_sanitizer_enabled",
            "_is_debug_line_info_disabled",
            "_is_auto_map_parallel_blocks_enabled",
            "force_disable_ffts",
            "is_cann_version_at_least",
    ):
        setattr(utils_stub, name, return_false)
    for name in (
            "_get_kernel_target",
            "_get_triton_adapter_opt_path",
            "_get_triton_mlir_opt_path",
            "_get_triton_opt_path",
            "_get_bishengir_opt_path",
    ):
        setattr(utils_stub, name, lambda *_args, **_kwargs: "")
    utils_stub._get_npucompiler_path = lambda *_args, **_kwargs: ("", {})
    utils_stub._get_auto_blockify_blacklist_reasons = lambda *_args, **_kwargs: []
    utils_stub._warn_auto_blockify_disabled = lambda *_args, **_kwargs: None
    utils_stub._remove_deprecated_npu_options = remove_deprecated_npu_options
    utils_stub._warn_deprecated_npu_option = lambda name: warnings.warn(
        f"Ascend compile option '{name}' is deprecated and ignored.", FutureWarning)
    utils_stub._warn_deprecated_ascend_env_vars = lambda: None
    utils_stub.downgrade_llir = lambda llir: llir
    utils_stub.get_cann_version_file_hash = lambda: ""
    utils_stub.graph_ub_budget_bytes_for_arch = _stub_graph_ub_budget_bytes_for_arch
    utils_stub.ub_size_in_kbytes_for_arch = _stub_ub_size_in_kbytes_for_arch

    class UnusedNPUUtils:
        pass

    driver_stub = types.ModuleType(driver_name)
    driver_stub.NPUUtils = UnusedNPUUtils

    cache_stub = types.ModuleType(cache_name)
    cache_stub._base32 = lambda value: str(value)
    dump_dir = Path("/fake/dump")
    cache_stub.get_dump_manager = lambda *_args, **_kwargs: SimpleNamespace(
        cache_dir=str(dump_dir),
        _make_path=lambda filename: str(dump_dir / filename),
        put=lambda *_args, **_kwargs: None,
    )

    # Initialize the installed Triton package before temporarily replacing its
    # cache module below.  Loading compiler.py starts from triton._C; doing it
    # in the reverse order would make unrelated backend imports see the tiny
    # test stub instead of the runtime cache API they require.
    importlib.import_module("triton")

    previous_utils = sys.modules.get(utils_name)
    previous_driver = sys.modules.get(driver_name)
    previous_cache = sys.modules.get(cache_name)
    sys.modules[utils_name] = utils_stub
    sys.modules[driver_name] = driver_stub
    sys.modules[cache_name] = cache_stub
    sys.modules.pop(module_name, None)
    debug_line_rewriter_name = "triton.backends.ascend.debug_line_rewriter"
    previous_debug_line_rewriter = sys.modules.get(debug_line_rewriter_name)
    sys.modules.pop(debug_line_rewriter_name, None)
    try:
        debug_line_rewriter_path = compiler_path.with_name("debug_line_rewriter.py")
        debug_line_rewriter_spec = importlib.util.spec_from_file_location(debug_line_rewriter_name,
                                                                          debug_line_rewriter_path)
        debug_line_rewriter = importlib.util.module_from_spec(debug_line_rewriter_spec)
        assert debug_line_rewriter_spec is not None and debug_line_rewriter_spec.loader is not None
        sys.modules[debug_line_rewriter_name] = debug_line_rewriter
        debug_line_rewriter_spec.loader.exec_module(debug_line_rewriter)
        spec = importlib.util.spec_from_file_location(module_name, compiler_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_debug_line_rewriter is None:
            sys.modules.pop(debug_line_rewriter_name, None)
        else:
            sys.modules[debug_line_rewriter_name] = previous_debug_line_rewriter
        if previous_utils is None:
            sys.modules.pop(utils_name, None)
        else:
            sys.modules[utils_name] = previous_utils
        if previous_driver is None:
            sys.modules.pop(driver_name, None)
        else:
            sys.modules[driver_name] = previous_driver
        if previous_cache is None:
            sys.modules.pop(cache_name, None)
        else:
            sys.modules[cache_name] = previous_cache
    return module


def _parse_options(compiler, arch, opts=None):
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch=arch))
    return backend.parse_options({} if opts is None else opts)


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "requested_capacity", "expected_capacity"),
    (
        ("Ascend910B1", _UNSET, 192 * 1024 * 80 // 100),
        ("Ascend910B1", None, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", None, 256 * 1024 * 80 // 100),
        ("Ascend950A3", None, 256 * 1024 * 80 // 100),
        ("Ascend910B1", 0, 0),
        ("Ascend910B1", 4096, 4096),
        ("Ascend910B1", 192 * 1024 * 80 // 100 + 1, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", 256 * 1024 * 80 // 100 + 1, 256 * 1024 * 80 // 100),
        ("unknown-arch", None, 0),
    ),
)
def test_npu_options_normalizes_graph_ub_budget(compiler_module, arch, requested_capacity, expected_capacity):
    """Direct NPUOptions users receive the same final integer as JIT users."""
    kwargs = {"arch": arch}
    if requested_capacity is not _UNSET:
        kwargs["graph_optimize_ub_capacity_bytes"] = requested_capacity

    options = compiler_module.NPUOptions(**kwargs)

    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "requested_capacity", "expected_capacity"),
    (
        ("Ascend910B1", _UNSET, 192 * 1024 * 80 // 100),
        ("Ascend910B1", None, 192 * 1024 * 80 // 100),
        ("Ascend910_9581", None, 256 * 1024 * 80 // 100),
        ("Ascend950A3", None, 256 * 1024 * 80 // 100),
        ("Ascend910B1", 0, 0),
        ("Ascend910B1", 4096, 4096),
        ("Ascend910B1", 192 * 1024 * 80 // 100 + 1, 192 * 1024 * 80 // 100),
    ),
)
def test_parse_options_normalizes_graph_ub_budget(compiler_module, arch, requested_capacity, expected_capacity):
    opts = {}
    if requested_capacity is not _UNSET:
        opts["graph_optimize_ub_capacity_bytes"] = requested_capacity

    options = _parse_options(compiler_module, arch, opts)

    assert options.arch == arch
    assert not hasattr(options, "_arch")
    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_normalized_graph_ub_budget_contributes_to_npu_hash(compiler_module):
    auto = compiler_module.NPUOptions(arch="Ascend910B1")
    explicit_none = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=None)
    disabled = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=0)
    small = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=4096)
    clamped = compiler_module.NPUOptions(arch="Ascend910B1",
                                         graph_optimize_ub_capacity_bytes=192 * 1024 * 80 // 100 + 1)

    assert auto.__dict__["graph_optimize_ub_capacity_bytes"] == 192 * 1024 * 80 // 100
    assert explicit_none.graph_optimize_ub_capacity_bytes == 192 * 1024 * 80 // 100
    assert clamped.graph_optimize_ub_capacity_bytes == 192 * 1024 * 80 // 100
    assert auto.hash() == explicit_none.hash() == clamped.hash()
    assert auto.hash() != disabled.hash()
    assert auto.hash() != small.hash()


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("requested_capacity", "error_type"),
    (
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ),
)
def test_npu_options_rejects_invalid_graph_ub_budget_requests(compiler_module, requested_capacity, error_type):
    with pytest.raises(error_type):
        compiler_module.NPUOptions(
            arch="Ascend910B1",
            graph_optimize_ub_capacity_bytes=requested_capacity,
        )


def _make_opt(
    *,
    is_pure_simt,
    superblock_factor=0,
    simt_optimization_mode=0,
    shared_mem_dynamic_size=None,
    disable_fma=False,
    compile_on_910_95=False,
):
    return SimpleNamespace(
        is_pure_simt=is_pure_simt,
        num_warps=4,
        warp_size=32,
        simt_optimization_mode=simt_optimization_mode,
        shared_mem_dynamic_size=shared_mem_dynamic_size,
        disable_fma=disable_fma,
        superblock_factor=superblock_factor,
        compile_on_910_95=compile_on_910_95,
    )


def _make_program_grid_contract(*transforms):
    return {
        "version": 2,
        "extent_source": "runtime_original_grid",
        "hidden_extent_axes": [0, 1],
        "hidden_argument_order": ["originalGridX", "originalGridY"],
        "hidden_argument_types": ["i32", "i32"],
        "transforms": list(transforms),
    }


def _run_ttir_to_npubin(
    compiler,
    monkeypatch,
    *,
    is_pure_simt=True,
    auto_map_enabled=False,
    has_blacklist_op=False,
    row_coalescing_applied=False,
    superblock_factor=0,
    common_options=(),
    simt_optimization_mode=0,
    resolved_simt_stack_limit=1152,
    shared_mem_dynamic_size=None,
    disable_fma=False,
):
    events = []
    commands = []
    module = _FakeModule(events)
    pass_manager = _FakePassManager(events)

    def parse_ttir_metadata(_ttir, metadata):
        events.append("parse")
        return {
            **metadata,
            "has_auto_blockify_blacklist_op": has_blacklist_op,
            "mix_mode": "aiv",
            "row_coalescing_applied": row_coalescing_applied,
        }

    def export_coalesce_metadata(_mod, _metadata, *, require_row_contract=False):
        events.append(f"export:{require_row_contract}")

    def run_bisheng(command, **_kwargs):
        commands.append(list(command))
        output = Path(command[command.index("-o") + 1] + ".o")
        output.write_bytes(b"npubin")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        compiler,
        "ir",
        SimpleNamespace(pass_manager=lambda _context: (events.append("pass_manager") or pass_manager)),
    )
    monkeypatch.setattr(compiler, "_parse_ttir_metadata", parse_ttir_metadata)
    monkeypatch.setattr(compiler, "_export_coalesce_metadata", export_coalesce_metadata)
    monkeypatch.setattr(
        compiler,
        "get_common_bishengir_compile_options",
        lambda _metadata: list(common_options),
    )
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bisheng", {}))
    monkeypatch.setattr(
        compiler,
        "_is_auto_map_parallel_blocks_enabled",
        lambda: auto_map_enabled,
    )

    # Keep this argv matrix independent of the host torch_npu configuration
    # while checking that Pure-SIMT uses the backend-resolved stack limit.
    def get_simt_stack_limit():
        return resolved_simt_stack_limit

    monkeypatch.setattr(compiler, "get_simt_stack_limit", get_simt_stack_limit)
    monkeypatch.setattr(compiler.subprocess, "run", run_bisheng)

    result = compiler.ttir_to_npubin(
        module,
        {"bisheng_options": None},
        _make_opt(
            is_pure_simt=is_pure_simt,
            superblock_factor=superblock_factor,
            simt_optimization_mode=simt_optimization_mode,
            shared_mem_dynamic_size=shared_mem_dynamic_size,
            disable_fma=disable_fma,
        ),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    return events, commands[0]


def test_simt_stack_limit_is_not_a_compile_option(compiler_module):
    assert "simt_stack_limit" not in compiler_module.NPUOptions.__dataclass_fields__
    assert "simt_stack_limit" not in _parse_options(compiler_module, "Ascend910_9581").__dict__
    with pytest.raises(TypeError, match="simt_stack_limit"):
        compiler_module.NPUOptions(arch="Ascend910_9581", simt_stack_limit=8192)


def _run_linalg_to_npubin(compiler, monkeypatch, function_name, has_blacklist_op):
    commands = []
    parsed_metadata = defaultdict(
        lambda: None, {
            "hash": "layout-memory-contract-test",
            "target": SimpleNamespace(arch="Ascend910B"),
            "program_grid_transforms": None,
            "program_grid_mapping_applied": False,
            "row_coalescing_applied": False,
            "has_auto_blockify_blacklist_op": has_blacklist_op,
            "auto_blockify_enabled": not has_blacklist_op,
            "ptsm_cap_authorized": False,
            "mix_mode": "aiv",
        })

    class FakeNPUUtils:

        def has_device_limit(self):
            return False

        def get_arch(self):
            return "Ascend910B"

    def parse_linalg_metadata(linalg, _metadata):
        return linalg, parsed_metadata

    def run_bisheng(command, **_kwargs):
        commands.append(list(command))
        Path(command[command.index("-o") + 1] + "_reloc.o").write_bytes(b"npubin")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(compiler, "_parse_linalg_metadata", parse_linalg_metadata)
    monkeypatch.setattr(compiler, "get_common_bishengir_compile_options", lambda _metadata: [])
    monkeypatch.setattr(compiler, "get_auto_bind_sub_block_option", lambda _metadata: False)
    monkeypatch.setattr(compiler, "NPUUtils", FakeNPUUtils)
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bishengir-compile", {}))
    monkeypatch.setattr(compiler, "_is_auto_map_parallel_blocks_enabled", lambda: True)
    monkeypatch.setattr(compiler.subprocess, "run", run_bisheng)

    result = getattr(compiler, function_name)(
        "module {}",
        {},
        SimpleNamespace(debug=False, target_arch="Ascend950PR", is_pure_simt=False),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    return commands[0]


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_export_coalesce_metadata_removes_attrs_and_marks_row(compiler_module, monkeypatch):
    removed = []

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        removed.append(name)
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            remove_attr=remove_attr,
        )),
    )

    coalesced = SimpleNamespace(attrs={
        "hacc.coalesce_factor": 4,
        "hacc.coalesce_axis": 2,
        "hacc.coalesce_grid_ceil_div": 1,
    })
    metadata = {}
    compiler_module._export_coalesce_metadata(coalesced, metadata)

    assert metadata == {
        "coalesce_factor": 4,
        "coalesce_axis": 2,
        "coalesce_grid_ceil_div": True,
        "row_coalescing_applied": True,
    }
    assert coalesced.attrs == {}
    assert removed == [
        "hacc.coalesce_factor",
        "hacc.coalesce_axis",
        "hacc.coalesce_grid_ceil_div",
    ]

    uncoalesced = SimpleNamespace(attrs={})
    uncoalesced_metadata = {}
    compiler_module._export_coalesce_metadata(uncoalesced, uncoalesced_metadata)
    assert uncoalesced_metadata == {
        "coalesce_factor": 1,
        "coalesce_axis": -1,
        "coalesce_grid_ceil_div": False,
        "row_coalescing_applied": False,
    }


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_export_coalesce_metadata_rejects_partial_row_contract(compiler_module, monkeypatch):

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            remove_attr=remove_attr,
        )),
    )

    with pytest.raises(RuntimeError, match="launch contract"):
        compiler_module._export_coalesce_metadata(
            SimpleNamespace(attrs={"hacc.coalesce_factor": 4}),
            {},
            require_row_contract=True,
        )

    with pytest.raises(RuntimeError, match="RowCoalescing"):
        compiler_module._export_coalesce_metadata(
            SimpleNamespace(attrs={
                "hacc.coalesce_factor": 4,
                "hacc.coalesce_axis": 0,
            }),
            {},
            require_row_contract=True,
        )


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_ttir_to_npubin_exports_make_ttir_row_contract_only_for_pure_simt(compiler_module, monkeypatch):
    events, _command = _run_ttir_to_npubin(
        compiler_module,
        monkeypatch,
        is_pure_simt=True,
    )
    assert events == [
        "str:0",
        "parse",
        "export:True",
        "str:1",
    ]

    with monkeypatch.context() as pure_simt_off:
        events, _command = _run_ttir_to_npubin(
            compiler_module,
            pure_simt_off,
            is_pure_simt=False,
        )
    assert events == ["str:0", "parse"]


def _run_make_ttir_with_recorded_graph_options(compiler, monkeypatch, options):
    events = []
    module = _FakeModule(events)
    pass_manager = _FakePassManager(events)
    graph_calls = []

    def record(name):
        return lambda _pm, *args, **kwargs: events.append((name, args, kwargs))

    monkeypatch.setattr(
        compiler,
        "ir",
        SimpleNamespace(pass_manager=lambda _context: pass_manager),
    )
    monkeypatch.setattr(
        compiler,
        "passes",
        SimpleNamespace(
            common=SimpleNamespace(
                add_inliner=record("inliner"),
                add_canonicalizer=record("canonicalizer"),
                add_cse=record("cse"),
                add_licm=record("licm"),
                add_symbol_dce=record("symbol_dce"),
            ),
            ttir=SimpleNamespace(
                add_rewrite_tensor_descriptor_to_pointer=record("rewrite_tensor_descriptor_to_pointer"),
                add_combine=record("combine"),
                add_reorder_broadcast=record("reorder_broadcast"),
                add_loop_unroll=record("loop_unroll"),
            ),
        ),
    )
    monkeypatch.setattr(
        compiler,
        "ascend",
        SimpleNamespace(passes=SimpleNamespace(ttir=SimpleNamespace(
            add_graph_optimize=lambda _pm, **kwargs: graph_calls.append(kwargs)))),
    )

    assert compiler.make_ttir(module, {}, options) is module
    return events, graph_calls


def test_make_ttir_passes_canonical_compile_mode_to_graph_optimize(compiler_module, monkeypatch):
    options = SimpleNamespace(
        enable_graph_optimize=True,
        target_arch="Ascend910B1",
        compile_mode="simt_only",
        debug=False,
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls == [{
        "ub_capacity_bytes": 192 * 1024 * 80 // 100,
        "compile_mode": "simt_only",
    }]
    assert events[-1] == "run_row"


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
def test_npu_options_keep_graph_remark_compatibility_default(compiler_module):
    """The legacy graph-remarks name remains discoverable with a fixed default."""
    options = compiler_module.NPUOptions(arch="Ascend910B1")
    assert options.__dict__["graph_optimize_emit_remarks"] is False


@pytest.mark.skip(reason="The case is not supported on Ascend 950, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "expected_capacity"),
    (
        ("Ascend910B1", 192 * 1024 * 80 // 100),
        ("Ascend910_9581", 256 * 1024 * 80 // 100),
        ("Ascend950A3", 256 * 1024 * 80 // 100),
        ("unknown-arch", 0),
    ),
)
def test_make_ttir_forwards_normalized_graph_ub_budget(compiler_module, monkeypatch, arch, expected_capacity):
    options = compiler_module.NPUOptions(arch=arch)

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls[0]["ub_capacity_bytes"] == expected_capacity
    assert events[-1] == "run_row"


def test_ttir_to_npubin_auto_blockify_argv_matrix(compiler_module, monkeypatch):
    common_options = ["--common-before-pure-simt", "--common-after-pure-simt"]
    pure_simt_prefix = [
        "--enable-hivm-compile=false",
        "--enable-triton-ir-compile",
        "--pure-simt",
        "--num-warps=4",
        "--threads-per-warp=32",
        "--simt-optimization-mode=1000017",
        "--simt-stack-limit=64",
        "--shared-mem-dynamic-size=4096",
        "--disable-fma",
    ]
    auto_blockify_flag = "--enable-auto-blockify-loop"

    for env_enabled, blacklisted, row_applied, superblock in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        (0, 7),
    ):
        with monkeypatch.context() as case_monkeypatch:
            _events, command = _run_ttir_to_npubin(
                compiler_module,
                case_monkeypatch,
                auto_map_enabled=env_enabled,
                has_blacklist_op=blacklisted,
                row_coalescing_applied=row_applied,
                superblock_factor=superblock,
                common_options=common_options,
                simt_optimization_mode=1000017,
                resolved_simt_stack_limit=64,
                shared_mem_dynamic_size=4096,
                disable_fma=True,
            )

        second_injection = env_enabled and not row_applied
        case = f"E={env_enabled}, B={blacklisted}, R={row_applied}, superblock={superblock}"

        expected_options = [*common_options, *pure_simt_prefix]
        if second_injection:
            expected_options.append(auto_blockify_flag)
            if superblock > 0:
                expected_options.append(f"--super-block-factor={superblock}")

        assert command[0] == "/fake/bisheng", case
        assert Path(command[1]).name == "kernel.ttir.mlir", case
        assert command[2:-2] == expected_options, case
        assert command[-2] == "-o", case
        assert Path(command[-1]).name == "kernel", case


@pytest.mark.parametrize(
    ("auto_map_enabled", "blacklisted", "row_applied", "transforms", "expected_auto", "expected_ptsm"),
    (
        (False, False, False, None, False, False),
        (True, False, False, None, True, False),
        (True, False, True, None, True, False),
        (True, True, False, None, False, False),
        (True, False, False, ((0, 1, 16, False, False), ), True, False),
        (True, False, False, ((0, 0, 64, True, True), ), False, True),
        (True, False, False, ((0, 1, 16, False, False), (1, 0, 4, True, True)), False, True),
    ),
)
def test_finalize_program_launch_policy_uses_linalg_ptsm_gate(
    compiler_module,
    monkeypatch,
    auto_map_enabled,
    blacklisted,
    row_applied,
    transforms,
    expected_auto,
    expected_ptsm,
):
    contract = None
    if transforms is not None:
        contract = _make_program_grid_contract(*(dict(
            order=order,
            kind="ceil_div",
            axis=axis,
            factor=factor,
            persistent_coverage=persistent,
            grid_stride_abi_verified=grid_stride,
        ) for order, axis, factor, persistent, grid_stride in transforms))
    metadata = {
        "program_grid_transforms": contract,
        "program_grid_mapping_applied": contract is not None,
        "row_coalescing_applied": row_applied,
        "has_auto_blockify_blacklist_op": blacklisted,
        "mix_mode": "aiv",
    }
    monkeypatch.setattr(compiler_module, "_is_auto_map_parallel_blocks_enabled", lambda: auto_map_enabled)

    compiler_module._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=False))

    assert metadata["auto_blockify_enabled"] is expected_auto
    assert metadata["ptsm_cap_authorized"] is expected_ptsm


@pytest.mark.parametrize(
    ("auto_map_enabled", "blacklisted", "row_applied", "expected_auto"),
    (
        (False, False, False, False),
        (False, True, False, False),
        (True, False, False, True),
        (True, True, False, True),
        (True, False, True, False),
        (True, True, True, False),
    ),
)
def test_finalize_program_launch_policy_pure_simt_ignores_blacklist(
    compiler_module,
    monkeypatch,
    auto_map_enabled,
    blacklisted,
    row_applied,
    expected_auto,
):
    metadata = {
        "program_grid_transforms": None,
        "program_grid_mapping_applied": False,
        "row_coalescing_applied": row_applied,
        "has_auto_blockify_blacklist_op": blacklisted,
        "mix_mode": "aiv",
    }
    monkeypatch.setattr(compiler_module, "_is_auto_map_parallel_blocks_enabled", lambda: auto_map_enabled)

    compiler_module._finalize_program_launch_policy(metadata, SimpleNamespace(is_pure_simt=True))

    assert metadata["auto_blockify_enabled"] is expected_auto
    assert metadata["ptsm_cap_authorized"] is False


@pytest.mark.parametrize(
    "function_name",
    (
        "linalg_to_bin_enable_npu_compile_910_95",
        "linalg_to_bin_enable_npu_compile_A2_A3",
    ),
)
@pytest.mark.parametrize("has_blacklist_op", (False, True))
def test_non_pure_simt_linalg_compilers_keep_blacklist_auto_blockify_gate(
    compiler_module,
    monkeypatch,
    function_name,
    has_blacklist_op,
):
    command = _run_linalg_to_npubin(
        compiler_module,
        monkeypatch,
        function_name,
        has_blacklist_op,
    )

    assert ("--enable-auto-blockify-loop" in command) is not has_blacklist_op
    assert "--pure-simt" not in command


def test_default_compile_mode_keeps_the_91095_layout_memory_gate_prepared(compiler_module):
    """The canonical default is portable and enables the Ascend 950 template gate."""

    a2_default = compiler_module.NPUOptions(arch="Ascend910B1")
    assert a2_default.compile_on_910_95 is False
    assert a2_default.compile_mode == "simd_simt_template"
    assert a2_default.is_pure_simt is False

    a5_default = compiler_module.NPUOptions(arch="Ascend910_9589")
    assert a5_default.compile_on_910_95 is True
    assert a5_default.compile_mode == "simd_simt_template"
    assert a5_default.is_pure_simt is False

    with pytest.warns(FutureWarning, match="compile_on_910_95"):
        ignored_legacy_value = compiler_module.NPUOptions(
            arch="Ascend910_9589",
            compile_on_910_95=False,
        )
    assert ignored_legacy_value.compile_on_910_95 is True

    canonical = compiler_module.NPUOptions(
        arch="Ascend910_9589",
        compile_mode="simd_simt_template",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = compiler_module.NPUOptions(
            arch="Ascend910_9589",
            compile_mode="unstructured_in_simt",
        )
    assert not caught
    assert alias.compile_mode == canonical.compile_mode == "simd_simt_template"
    assert alias.hash() == canonical.hash()

    with pytest.raises(ValueError, match=r"invalid compile_mode='simt_template'"):
        compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simt_template")

    explicit_simd = compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simd")
    assert explicit_simd.compile_mode == "simd"
    assert explicit_simd.is_pure_simt is False

    explicit_template = compiler_module.NPUOptions(
        arch="Ascend910_9589",
        compile_mode="simd_simt_template",
    )
    assert explicit_template.compile_mode == "simd_simt_template"
    assert explicit_template.is_pure_simt is False

    explicit_only = compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simt_only")
    assert explicit_only.compile_mode == "simt_only"
    assert explicit_only.is_pure_simt is True

    # Legacy spellings remain discoverable while compile_mode controls lowering.
    assert explicit_only.__dict__["force_simt_only"] is False
    assert explicit_template.__dict__["force_simt_template"] is False


@pytest.mark.parametrize("arch", ["Ascend910B4", "Ascend910_9391", "Ascend950PR"])
@pytest.mark.parametrize("mode", [
    {"gm": 4, "l1": 2, "l0c": 1, "ub": 2},
    {"future": 0, "gm": -1},
    {},
])
def test_multibuffer_mode_preserves_values_and_metadata(compiler_module, arch, mode):
    options = _parse_options(compiler_module, arch, {"multibuffer_mode": mode})
    expected = tuple(sorted(mode.items()))
    assert options.multibuffer_mode == expected
    assert options.num_stages is None
    assert options.multibuffer is True
    metadata = dict(options.__dict__, multibuffer_mode=json.loads(json.dumps(options.multibuffer_mode)))
    restored = _parse_options(compiler_module, arch, metadata)
    assert restored.multibuffer_mode == expected
    assert restored.hash() == options.hash()
    assert restored.hash() != _parse_options(compiler_module, arch).hash()


@pytest.mark.parametrize("mode", [
    "[(gm,2)]",
    True,
    2,
    {"gm": "2"},
    {1: 2},
    {"gm": 2.0},
    {"gm": True},
    ("gm", 2),
    (("gm", ), ),
    (("gm", 2, 3), ),
    ((1, 2), ),
    (("gm", 2.0), ),
    (("gm", True), ),
])
def test_multibuffer_mode_requires_string_keys_and_integer_counts(compiler_module, mode):
    with pytest.raises(TypeError, match="multibuffer_mode must be a dict"):
        compiler_module.NPUOptions(multibuffer_mode=mode)


@pytest.mark.parametrize("name,value", [
    ("limit_auto_multi_buffer_only_for_local_buffer", False),
    ("limit_auto_multi_buffer_of_local_buffer", "no-l0c"),
    ("limit_auto_multi_buffer_buffer", "only-vector"),
])
@pytest.mark.parametrize("mode", [None, {"ub": 2}])
def test_multibuffer_legacy_options_are_preserved_without_ta_warnings(compiler_module, name, value, mode):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = compiler_module.NPUOptions(multibuffer_mode=mode, **{name: value})
    assert not caught
    assert getattr(options, name) == value
    assert options.multibuffer is True


def _capture_multibuffer_command(compiler, monkeypatch, is_a5, requested, derived=None):
    options = _parse_options(compiler, "Ascend950PR" if is_a5 else "Ascend910B4", requested)
    metadata = dict(options.__dict__, hash="layout-memory-contract-test", mix_mode="mix", bitcodes=None,
                    auto_blockify_enabled=False)
    metadata.update(derived or {})
    monkeypatch.setattr(compiler, "_parse_linalg_metadata", lambda source, meta: (source, meta))
    monkeypatch.setattr(compiler, "_finalize_program_launch_policy", lambda *_args: None)
    monkeypatch.setattr(compiler, "get_common_bishengir_compile_options", lambda _meta: [])
    monkeypatch.setattr(compiler, "get_auto_bind_sub_block_option", lambda _meta: True)
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/compiler", {}))
    monkeypatch.setattr(compiler, "NPUUtils",
                        lambda: SimpleNamespace(has_device_limit=lambda: False, get_arch=lambda: options.target_arch))
    monkeypatch.delenv("TRITON_ENABLE_LIBDEVICE", raising=False)

    class CommandCaptured(BaseException):
        pass

    commands = []

    def capture(command, **_kwargs):
        commands.append(command)
        raise CommandCaptured

    monkeypatch.setattr(compiler.subprocess, "run", capture)
    compile_fn = (compiler.linalg_to_bin_enable_npu_compile_910_95
                  if is_a5 else compiler.linalg_to_bin_enable_npu_compile_A2_A3)
    with pytest.raises(CommandCaptured):
        compile_fn("module {}", metadata, options)
    assert len(commands) == 1
    return commands[0]


@pytest.mark.parametrize("is_a5", [False, True])
@pytest.mark.parametrize("requested,enabled", [({}, True), ({"num_stages": 1}, False), ({"num_stages": 3}, True),
                                               ({"multibuffer": False}, False)])
def test_multibuffer_legacy_command_stays_on_old_interface(compiler_module, monkeypatch, is_a5, requested, enabled):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5, requested)
    assert not any(arg.startswith("--multibuffer-mode=") for arg in command)
    assert f"--enable-auto-multi-buffer={enabled}" in command
    assert ("--limit-auto-multi-buffer-of-local-buffer=no-limit" in command) == is_a5


@pytest.mark.parametrize("is_a5", [False, True])
def test_multibuffer_mode_is_forwarded_alongside_existing_defaults(compiler_module, monkeypatch, is_a5):
    mode = {"ub": 2, "gm": 4, "l0c": 1, "l1": 2}
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5, {"multibuffer_mode": mode})
    assert command.count("--multibuffer-mode=[(gm,4),(l0c,1),(l1,2),(ub,2)]") == 1
    assert "--enable-auto-multi-buffer=True" in command
    assert ("--limit-auto-multi-buffer-of-local-buffer=no-limit" in command) == is_a5


@pytest.mark.parametrize("is_a5", [False, True])
@pytest.mark.parametrize("switch", [False, True])
def test_multibuffer_explicit_old_options_are_still_forwarded(compiler_module, monkeypatch, is_a5, switch):
    requested = {
        "multibuffer_mode": {"gm": 4, "l1": 2, "l0c": 2, "ub": 2},
        "multibuffer": switch,
        "limit_auto_multi_buffer_only_for_local_buffer": True,
        "limit_auto_multi_buffer_of_local_buffer": "no-l0c",
        "limit_auto_multi_buffer_buffer": "only-vector",
        "set_workspace_multibuffer": 0,
    }
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5, requested)
    assert "--multibuffer-mode=[(gm,4),(l0c,2),(l1,2),(ub,2)]" in command
    assert f"--enable-auto-multi-buffer={switch}" in command
    assert "--limit-auto-multi-buffer-only-for-local-buffer=True" in command
    assert "--limit-auto-multi-buffer-of-local-buffer=no-l0c" in command
    assert "--set-workspace-multibuffer=0" in command
    if is_a5:
        assert "--limit-auto-multi-buffer-buffer=only-vector" in command


def test_multibuffer_keeps_dynamic_cv_workspace_constraint(compiler_module, monkeypatch):
    command = _capture_multibuffer_command(compiler_module, monkeypatch, True,
                                           {"multibuffer_mode": {"gm": 4, "l1": 2, "l0c": 1, "ub": 2}},
                                           derived={"set_workspace_multibuffer": 0})
    assert "--set-workspace-multibuffer=0" in command


@pytest.mark.parametrize("is_a5", [False, True])
def test_multibuffer_value_semantics_are_left_to_npuir(compiler_module, monkeypatch, is_a5):
    # Level names and count ranges are vendor decisions; partial modes are forwarded as supplied.
    command = _capture_multibuffer_command(compiler_module, monkeypatch, is_a5,
                                           {"multibuffer_mode": {"future": 0, "ub": -1}})
    assert "--multibuffer-mode=[(future,0),(ub,-1)]" in command


@pytest.mark.parametrize("arch", ["Ascend910B4", "Ascend910_9391", "Ascend950PR"])
@pytest.mark.parametrize("stages", [0, 1, 2, 3])
@pytest.mark.parametrize("mode", [{"gm": 2, "l1": 2, "l0c": 2, "ub": 2}, {}])
def test_multibuffer_mode_rejects_num_stages_even_when_equivalent(compiler_module, arch, stages, mode):
    with pytest.raises(ValueError, match="num_stages and multibuffer_mode cannot be specified together"):
        _parse_options(compiler_module, arch, {"multibuffer_mode": mode, "num_stages": stages})


def test_multibuffer_default_num_stages_does_not_conflict(compiler_module):
    assert compiler_module.NPUOptions().num_stages == 2
    assert compiler_module.NPUOptions(num_stages=1).num_stages == 1
    options = compiler_module.NPUOptions(multibuffer_mode={"ub": 2}, num_stages=None)
    assert options.num_stages is None


@pytest.mark.parametrize("arch", ["Ascend910B4", "Ascend950PR"])
def test_multibuffer_dictionary_order_does_not_change_compiler_hash(compiler_module, arch):
    mode = {"ub": 2, "gm": 4, "l1": 2}
    options = _parse_options(compiler_module, arch, {"multibuffer_mode": mode})
    reordered = _parse_options(compiler_module, arch, {"multibuffer_mode": dict(reversed(list(mode.items())))})
    assert reordered.multibuffer_mode == options.multibuffer_mode
    assert reordered.hash() == options.hash()
    mode["ub"] = 8
    assert dict(options.multibuffer_mode)["ub"] == 2
