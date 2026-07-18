import enum
import hashlib
import importlib.util
import json
import re
import pytest
import stat
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def _repo_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "third_party" / "rpu" / "backend" / "compiler.py").exists():
            return parent
    return here.parents[5]


def _rpu_backend_path():
    return _repo_root() / "third_party" / "rpu" / "backend"


def _source_between(source, start_marker, end_marker):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


class _FakePassManager:

    def __init__(self, context):
        self.context = context
        self.added = []
        self.ran = False

    def run(self, module, stage_name=""):
        self.ran = True
        self.last_stage_name = stage_name
        if "emit_executable_to_rpurc" in self.added:
            module.rpurc_ran_pm = self
        else:
            module.ran_pm = self
        if "lower_supported_ttir_to_executable" in self.added:
            if _FakeRPU.direct_pass_error is not None:
                raise RuntimeError(_FakeRPU.direct_pass_error)
            plan = json.loads(module.attrs["rpu.plan.json"])
            module.rpuexec_kernel_name = plan["kernel_name"]
            module.rpuexec_pattern = plan["pattern"]
            module.has_rpuexec_kernel = True
            module.has_rpuplan_kernel = False
        if "recognize_rpu_plan" in self.added:
            module.has_rpuplan_kernel = True
        if "emit_executable_to_rpurc" in self.added:
            if _FakeRPU.executable_emit_pass_error is not None:
                raise RuntimeError(_FakeRPU.executable_emit_pass_error)
            summary = _FakeRPU._get_rpuexec_kernel_summary(module)
            module.attrs["rpu.rpurc.kernel_name"] = summary["kernel_name"]
            module.attrs["rpu.rpurc.source_kind"] = "rpu_executable"
            module.attrs["rpu.rpurc.source"] = ('#include "rpu_tile.h"\n'
                                                "using namespace rpu;\n"
                                                f"__rprog__ void {summary['kernel_name']}() {{\n"
                                                "    Context ctx;\n"
                                                "}\n")


class _FakeIR:
    parsed_modules = []
    parse_error = None

    @staticmethod
    def pass_manager(context):
        return _FakePassManager(context)

    @staticmethod
    def parse_mlir_module(path, context):
        if _FakeIR.parse_error is not None:
            raise RuntimeError(_FakeIR.parse_error)
        text = Path(path).read_text()
        module = _FakeMLIRModule(_add_plan_json())
        module.context = context
        module.has_rpuplan_kernel = "rpu_plan.kernel" in text
        module.has_rpuexec_kernel = "rpu.kernel" in text
        module.parsed_from = str(path)
        module.parsed_text = text
        _FakeIR.parsed_modules.append(module)
        return module


class _FakeCommonPasses:

    @staticmethod
    def add_inliner(pm):
        pm.added.append("common.inliner")

    @staticmethod
    def add_canonicalizer(pm):
        pm.added.append("common.canonicalizer")

    @staticmethod
    def add_cse(pm):
        pm.added.append("common.cse")

    @staticmethod
    def add_licm(pm):
        pm.added.append("common.licm")

    @staticmethod
    def add_symbol_dce(pm):
        pm.added.append("common.symbol_dce")


class _FakeTTIRPasses:

    @staticmethod
    def add_rewrite_tensor_pointer(pm):
        pm.added.append("ttir.rewrite_tensor_pointer")

    @staticmethod
    def add_combine(pm):
        pm.added.append("ttir.combine")

    @staticmethod
    def add_reorder_broadcast(pm):
        pm.added.append("ttir.reorder_broadcast")


class _FakePasses:
    common = _FakeCommonPasses
    ttir = _FakeTTIRPasses


class _FakeRPUPasses:

    @staticmethod
    def add_recognize_plan(pm):
        pm.added.append("recognize_rpu_plan")

    @staticmethod
    def add_lower_supported_ttir_to_executable(pm):
        pm.added.append("lower_supported_ttir_to_executable")

    @staticmethod
    def add_emit_executable_to_rpurc(pm):
        pm.added.append("emit_executable_to_rpurc")


class _FakeRPU:
    passes = _FakeRPUPasses()
    direct_result = None
    direct_error = None
    direct_pass_error = None
    executable_emit_pass_error = None
    summary = None
    supported_patterns = (
        "add",
        "gemm",
        "softmax",
        "convkxk",
        "resnet_block",
        "resnet50_bottleneck",
    )

    @staticmethod
    def get_module_str_attr(module, name):
        return module.attrs[name]

    @staticmethod
    def _direct_rpurc_supported_patterns():
        return list(_FakeRPU.supported_patterns)

    @staticmethod
    def _supported_executable_kernel_kinds():
        return list(_FakeRPU.supported_patterns)

    @staticmethod
    def _get_rpuplan_kernel_summary(module):
        if _FakeRPU.summary is not None:
            return dict(_FakeRPU.summary)
        if not getattr(module, "has_rpuplan_kernel", True):
            raise RuntimeError("missing rpu_plan.kernel")
        plan = json.loads(module.attrs["rpu.plan.json"])
        return {
            "kernel_name": plan["kernel_name"],
            "pattern": plan["pattern"],
        }

    @staticmethod
    def _get_rpuexec_kernel_summary(module):
        if not getattr(module, "has_rpuexec_kernel", False):
            raise RuntimeError("missing rpu.kernel")
        return {
            "kernel_name": getattr(module, "rpuexec_kernel_name", "rpu_add_kernel"),
            "pattern": getattr(module, "rpuexec_pattern", "add"),
        }

    @staticmethod
    def _emit_rpurc_from_plan(module):
        if _FakeRPU.direct_error is not None:
            raise RuntimeError(_FakeRPU.direct_error)
        if _FakeRPU.direct_result is not None:
            return dict(_FakeRPU.direct_result)
        summary = _FakeRPU._get_rpuplan_kernel_summary(module)
        return {
            "kernel_name":
            summary["kernel_name"],
            "pattern":
            summary["pattern"],
            "source": ('#include "rpu_tile.h"\n'
                       "using namespace rpu;\n"
                       f"__rprog__ void {summary['kernel_name']}() {{\n"
                       "    Context ctx;\n"
                       "}\n"),
        }

    @staticmethod
    def _emit_rpurc_from_executable(module):
        if _FakeRPU.direct_error is not None:
            raise RuntimeError(_FakeRPU.direct_error)
        if _FakeRPU.direct_result is not None:
            result = dict(_FakeRPU.direct_result)
            result.setdefault("source_kind", "rpu_executable")
            return result
        summary = _FakeRPU._get_rpuexec_kernel_summary(module)
        return {
            "kernel_name":
            summary["kernel_name"],
            "pattern":
            summary["pattern"],
            "source_kind":
            "rpu_executable",
            "source": ('#include "rpu_tile.h"\n'
                       "using namespace rpu;\n"
                       f"__rprog__ void {summary['kernel_name']}() {{\n"
                       "    Context ctx;\n"
                       "}\n"),
        }


class _FakeMLIRModule:

    def __init__(self, plan_json, trace_json=None):
        self.context = object()
        self.attrs = {"rpu.plan.json": plan_json}
        if trace_json is not None:
            self.attrs["rpu.plan.trace"] = trace_json
        self.ran_pm = None
        self.rpurc_ran_pm = None
        self.has_rpuplan_kernel = False
        self.has_rpuexec_kernel = False
        self.rpuexec_kernel_name = "rpu_add_kernel"
        self.rpuexec_pattern = "add"

    def __str__(self):
        if self.has_rpuexec_kernel:
            return ("module {\n"
                    f"  rpu.kernel @{self.rpuexec_kernel_name}() {{\n"
                    "    rpu.return\n"
                    f"  }} {{kind = \"{self.rpuexec_pattern}\"}}\n"
                    "}\n")
        if not self.has_rpuplan_kernel:
            return "module {\n  tt.func public @rpu_add_kernel() {\n    tt.return\n  }\n}\n"
        return ("module attributes {rpu.plan.json = \"...\"} {\n"
                "  tt.func public @rpu_add_kernel() {\n"
                "    tt.return\n"
                "  }\n"
                "  rpu_plan.kernel @rpu_add_kernel_plan attributes {pattern = \"add\"}\n"
                "}\n")


def _add_plan_json():
    return json.dumps(
        {
            "args": {"lhs": 1, "out": 0, "rhs": 2},
            "emission": {"kind": "add", "lhs": 1, "logical_n": 16, "n": 16, "out": 0, "rhs": 2},
            "kernel_name": "rpu_add_kernel",
            "layout": {"access": "linear", "memory": "contiguous_vector"},
            "mask": {"masked": False},
            "pattern": "add",
            "required_dsl_features": ["ctx.load_contig", "tile.add", "ctx.store_contig"],
            "shape": {"logical_n": 16, "n": 16},
            "signature": {
                "params": [
                    {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
                    {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
                    {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
                ],
                "return_type":
                "void",
            },
            "version": 1,
        },
        sort_keys=True,
    )


def _plan_json_with_identity(kernel_name, pattern):
    data = json.loads(_add_plan_json())
    data["kernel_name"] = kernel_name
    data["pattern"] = pattern
    data["emission"] = dict(data["emission"])
    data["emission"]["kind"] = pattern
    return json.dumps(data, sort_keys=True)


def _gemm_plan_json(m=16, n=16, k=16):
    return json.dumps(
        {
            "args": {"lhs": 1, "out": 0, "rhs": 2},
            "emission": {
                "kind": "gemm",
                "lhs": 1,
                "m": m,
                "n": n,
                "k": k,
                "out": 0,
                "rhs": 2,
            },
            "kernel_name": "rpu_gemm_kernel",
            "layout": {
                "access": "matrix_tile",
                "memory": "array2d",
                "order": "row_major",
            },
            "mask": {"masked": False},
            "pattern": "gemm",
            "required_dsl_features": [
                "rpu.Array",
                "ctx.load",
                "ctx.zeros",
                "ctx.mma",
                "ctx.store",
            ],
            "shape": {"m": m, "n": n, "k": k},
            "signature": {
                "params": [
                    {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
                    {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
                    {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
                ],
                "return_type":
                "void",
            },
            "version": 1,
        },
        sort_keys=True,
    )


def _softmax_plan_json(n=16):
    return json.dumps(
        {
            "args": {"input": 1, "out": 0},
            "emission": {
                "kind": "softmax",
                "input": 1,
                "n": n,
                "out": 0,
            },
            "kernel_name":
            "rpu_softmax_kernel",
            "layout": {"access": "linear", "memory": "contiguous_vector"},
            "mask": {"masked": False},
            "pattern":
            "softmax",
            "required_dsl_features": [
                "ctx.load_contig",
                "ctx.reduce_max_all",
                "rpu.exp",
                "ctx.reduce_sum_all",
                "rpu.reciprocal",
                "ctx.store_contig",
            ],
            "shape": {"n": n},
            "signature": {
                "params": [
                    {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
                    {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
                ],
                "return_type":
                "void",
            },
            "version":
            1,
        },
        sort_keys=True,
    )


def _rpuplan_artifact(result):
    return str(result)


def _trace_json():
    return json.dumps(
        {
            "attempts": [{
                "anchor": {
                    "kind": "op",
                    "location": "loc(\"add.mlir\":7:9)",
                    "op": "tt.store",
                },
                "location": "loc(\"add.mlir\":7:9)",
                "pattern": "add",
                "reason": "selected",
                "status": "matched",
            }],
            "function": {"location": "loc(\"add.mlir\":1:1)", "name": "rpu_add_kernel"},
            "matched":
            True,
            "selected":
            "add",
            "version":
            1,
        },
        sort_keys=True,
    )


def _debug_bundle(metadata):
    bundle = metadata["rpu_debug_bundle"]
    bundle_dir = Path(bundle["dir"])
    manifest = json.loads(Path(bundle["manifest"]).read_text())
    return bundle, bundle_dir, manifest


def _debug_files(bundle_dir):
    return {path.name for path in bundle_dir.iterdir()}


def _assert_rpurc_fail_detail(fail, *, kind, kernel_name, pattern, error_contains, extra=None):
    assert fail["stage"] == "rpurc_emit"
    assert fail["event"] == "fail"
    assert fail["detail"]["failure_kind"] == kind
    assert fail["detail"]["kernel_name"] == kernel_name
    assert fail["detail"]["pattern"] == pattern
    assert fail["detail"]["source"] == "rpu_plan.kernel"
    assert error_contains in fail["detail"]["error"]
    for key, value in (extra or {}).items():
        assert fail["detail"][key] == value


def _assert_rpuexec_fail_detail(fail, *, pattern, error_contains, extra=None):
    assert fail["stage"] == "rpuexec_lower"
    assert fail["event"] == "fail"
    assert fail["detail"]["pattern"] == pattern
    assert error_contains in fail["detail"]["error"]
    for key, value in (extra or {}).items():
        assert fail["detail"][key] == value


def _make_rpurc_from_rpuplan_bundle(compiler, bundle, metadata):
    exec_result = compiler.RPUBackend.make_rpuexec(bundle, metadata)
    assert isinstance(exec_result, compiler.RPUExecutableStageBundle)
    return compiler.RPUBackend.make_rpurc(exec_result, metadata)


def _install_triton_backend_stubs(monkeypatch):
    triton = types.ModuleType("triton")
    c_mod = types.ModuleType("triton._C")
    libtriton_mod = types.ModuleType("triton._C.libtriton")
    backends = types.ModuleType("triton.backends")
    compiler_mod = types.ModuleType("triton.backends.compiler")
    driver_mod = types.ModuleType("triton.backends.driver")

    @dataclass(frozen=True)
    class GPUTarget:
        backend: str
        arch: object
        warp_size: int

    class Language(enum.Enum):
        TRITON = 0
        GLUON = 1

    class BaseBackend:

        def __init__(self, target):
            self.target = target
            assert self.supports_target(target)

    class DriverBase:
        pass

    compiler_mod.BaseBackend = BaseBackend
    compiler_mod.GPUTarget = GPUTarget
    compiler_mod.Language = Language
    driver_mod.DriverBase = DriverBase
    backends.compiler = compiler_mod
    backends.driver = driver_mod
    libtriton_mod.ir = _FakeIR
    libtriton_mod.passes = _FakePasses
    libtriton_mod.rpu = _FakeRPU
    _FakeIR.parsed_modules = []
    _FakeIR.parse_error = None
    _FakeRPU.direct_result = None
    _FakeRPU.direct_error = None
    _FakeRPU.direct_pass_error = None
    _FakeRPU.executable_emit_pass_error = None
    _FakeRPU.summary = None
    _FakeRPU.supported_patterns = (
        "add",
        "gemm",
        "softmax",
        "convkxk",
        "resnet_block",
        "resnet50_bottleneck",
    )
    triton.backends = backends
    triton._C = c_mod
    c_mod.libtriton = libtriton_mod

    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton._C", c_mod)
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", libtriton_mod)
    monkeypatch.setitem(sys.modules, "triton.backends", backends)
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", compiler_mod)
    monkeypatch.setitem(sys.modules, "triton.backends.driver", driver_mod)
    monkeypatch.syspath_prepend(str(_rpu_backend_path()))
    for module_name in ("compiler", "driver", "aot"):
        sys.modules.pop(module_name, None)


def _load_generic_compiler_with_stage_stubs(monkeypatch):
    root = _repo_root()
    triton = types.ModuleType("triton")
    triton.__version__ = "test"
    triton.__path__ = [str(root / "python" / "triton")]
    c_mod = types.ModuleType("triton._C")
    libtriton_mod = types.ModuleType("triton._C.libtriton")
    libtriton_mod.get_cache_invalidating_env_vars = lambda: {}
    libtriton_mod.getenv = lambda *args, **kwargs: ""
    libtriton_mod.getenv_bool = lambda *args, **kwargs: False
    libtriton_mod.ir = types.SimpleNamespace(context=lambda: object())
    backends_mod = types.ModuleType("triton.backends")
    backends_mod.__path__ = [str(root / "python" / "triton" / "backends")]
    backends_mod.backends = {}
    compiler_pkg = types.ModuleType("triton.compiler")
    compiler_pkg.__path__ = [str(root / "python" / "triton" / "compiler")]
    codegen_mod = types.ModuleType("triton.compiler.code_generator")
    codegen_mod.ast_to_ttir = lambda *args, **kwargs: None
    runtime_mod = types.ModuleType("triton.runtime")
    runtime_mod.__path__ = [str(root / "python" / "triton" / "runtime")]
    autotuner_mod = types.ModuleType("triton.runtime.autotuner")

    class OutOfResources(Exception):
        pass

    autotuner_mod.OutOfResources = OutOfResources
    cache_mod = types.ModuleType("triton.runtime.cache")
    cache_mod.get_cache_manager = lambda *args, **kwargs: None
    cache_mod.get_dump_manager = lambda *args, **kwargs: None
    cache_mod.get_override_manager = lambda *args, **kwargs: None
    cache_mod.get_cache_key = lambda *args, **kwargs: ""
    driver_mod = types.ModuleType("triton.runtime.driver")

    class _Driver:
        active = None

    def spec(function_name, *args, **kwargs):
        active = _Driver.active
        if active is not None and hasattr(active, "spec") and hasattr(active.spec, function_name):
            return getattr(active.spec, function_name)(*args, **kwargs)
        return None

    driver_mod.driver = _Driver
    driver_mod.spec = spec

    tools_mod = types.ModuleType("triton.tools")
    disasm_mod = types.ModuleType("triton.tools.disasm")
    disasm_mod.get_sass = lambda *args, **kwargs: ""

    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton._C", c_mod)
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", libtriton_mod)
    monkeypatch.setitem(sys.modules, "triton.backends", backends_mod)
    monkeypatch.setitem(sys.modules, "triton.compiler", compiler_pkg)
    monkeypatch.setitem(sys.modules, "triton.compiler.code_generator", codegen_mod)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "triton.runtime.autotuner", autotuner_mod)
    monkeypatch.setitem(sys.modules, "triton.runtime.cache", cache_mod)
    monkeypatch.setitem(sys.modules, "triton.runtime.driver", driver_mod)
    monkeypatch.setitem(sys.modules, "triton.tools", tools_mod)
    monkeypatch.setitem(sys.modules, "triton.tools.disasm", disasm_mod)

    backend_spec = importlib.util.spec_from_file_location(
        "triton.backends.compiler",
        root / "python" / "triton" / "backends" / "compiler.py",
    )
    backend_compiler = importlib.util.module_from_spec(backend_spec)
    monkeypatch.setitem(sys.modules, "triton.backends.compiler", backend_compiler)
    backend_spec.loader.exec_module(backend_compiler)

    compiler_spec = importlib.util.spec_from_file_location(
        "triton.compiler.compiler",
        root / "python" / "triton" / "compiler" / "compiler.py",
    )
    triton_compiler = importlib.util.module_from_spec(compiler_spec)
    monkeypatch.setitem(sys.modules, "triton.compiler.compiler", triton_compiler)
    compiler_spec.loader.exec_module(triton_compiler)
    compiler_pkg.compiler = triton_compiler
    return backend_compiler, triton_compiler, driver_mod


def test_rpu_backend_modules_importable(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler
    import driver

    assert compiler.RPUBackend.supports_target(compiler.GPUTarget("rpu", "rpu-v1", 1))
    assert not compiler.RPUBackend.supports_target(compiler.GPUTarget("cuda", 90, 32))
    assert driver.RPUDriver.is_active() is False


def test_rpuplan_json_contains_signature_and_stable_schema(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    metadata = {}
    result = compiler.RPUBackend.make_rpuplan(_FakeMLIRModule(_add_plan_json()), metadata)

    assert isinstance(result, compiler.RPUPlanStageBundle)
    assert result.kernel_name == "rpu_add_kernel"
    assert result.pattern == "add"
    assert "_rpu_plan_module" not in metadata
    plan = json.loads(str(result))

    assert list(plan.keys()) == [
        "args",
        "emission",
        "kernel_name",
        "layout",
        "mask",
        "pattern",
        "required_dsl_features",
        "shape",
        "signature",
        "version",
    ]
    assert plan["version"] == 1
    assert plan["kernel_name"] == "rpu_add_kernel"
    assert plan["signature"] == {
        "params": [
            {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
            {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
            {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
        ],
        "return_type":
        "void",
    }
    assert plan["pattern"] == "add"
    assert plan["shape"] == {"logical_n": 16, "n": 16}
    assert plan["args"] == {"lhs": 1, "out": 0, "rhs": 2}
    assert plan["layout"] == {"access": "linear", "memory": "contiguous_vector"}
    assert plan["mask"] == {"masked": False}
    assert plan["required_dsl_features"] == [
        "ctx.load_contig",
        "tile.add",
        "ctx.store_contig",
    ]
    assert "trace" not in plan
    assert metadata["rpu_pattern"] == "add"
    assert metadata["name"] == "rpu_add_kernel"


def test_rpuplan_mlir_pass_path_reads_module_attr_without_regex(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    plan_json = _add_plan_json()
    module = _FakeMLIRModule(plan_json)
    metadata = {}
    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(module, metadata)

    assert str(result) == plan_json
    assert isinstance(result, compiler.RPUPlanStageBundle)
    assert result.module is module
    assert result.kernel_name == "rpu_add_kernel"
    assert result.pattern == "add"
    assert module.ran_pm.added == ["recognize_rpu_plan"]
    assert metadata["name"] == "rpu_add_kernel"
    assert metadata["shared"] == 0
    assert metadata["rpu_pattern"] == "add"


def test_rpuplan_metadata_comes_from_dialect_summary_not_json(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    if hasattr(compiler, "RPUKernelPlan"):
        monkeypatch.setattr(
            compiler.RPUKernelPlan,
            "from_json",
            staticmethod(lambda payload:
                         (_ for _ in ()).throw(AssertionError("JSON parser used for rpuplan metadata"))),
        )
    monkeypatch.setattr(
        _FakeRPU,
        "summary",
        {
            "kernel_name": "dialect_kernel",
            "pattern": "add",
        },
    )
    module = _FakeMLIRModule(_plan_json_with_identity("json_kernel", "json_pattern"))
    metadata = {}

    result = compiler.RPUBackend.make_rpuplan(module, metadata)

    assert json.loads(str(result))["kernel_name"] == "json_kernel"
    assert json.loads(str(result))["pattern"] == "json_pattern"
    assert result.kernel_name == "dialect_kernel"
    assert result.pattern == "add"
    assert result.module is module
    assert metadata["name"] == "dialect_kernel"
    assert metadata["rpu_pattern"] == "add"
    assert "_rpu_plan_module" not in metadata


def test_rpuplan_mlir_pass_path_records_optional_trace_metadata(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    trace = {
        "attempts": [{
            "anchor": {
                "kind": "op",
                "location": "loc(\"add.mlir\":7:9)",
                "op": "tt.store",
            },
            "location": "loc(\"add.mlir\":7:9)",
            "pattern": "add",
            "reason": "selected",
            "status": "matched",
        }],
        "matched":
        True,
        "selected":
        "add",
        "version":
        1,
    }
    module = _FakeMLIRModule(_add_plan_json(), json.dumps(trace, sort_keys=True))
    metadata = {}

    compiler.RPUBackend._make_rpuplan_from_mlir_module(module, metadata)

    assert metadata["rpu_plan_trace"] == trace
    attempt = metadata["rpu_plan_trace"]["attempts"][0]
    assert attempt["anchor"]["kind"] == "op"
    assert attempt["anchor"]["op"] == "tt.store"
    assert attempt["location"] == attempt["anchor"]["location"]


class _TraceFailingPassManager(_FakePassManager):

    def run(self, module, stage_name=""):
        self.ran = True
        self.last_stage_name = stage_name
        module.ran_pm = self
        module.attrs["rpu.plan.trace"] = json.dumps(
            {
                "attempts": [{
                    "anchor": {
                        "kind": "op",
                        "location": "loc(\"nested_add.mlir\":4:5)",
                        "op": "scf.if",
                    },
                    "location": "loc(\"nested_add.mlir\":4:5)",
                    "pattern": "add",
                    "reason": "did not match supported vector add",
                    "status": "failed",
                }],
                "function": {
                    "location": "loc(\"nested_add.mlir\":2:3)",
                    "name": "nested_add",
                },
                "matched":
                False,
                "selected":
                None,
                "version":
                1,
            },
            sort_keys=True,
        )
        raise RuntimeError("PassManager::run failed")


class _TraceFailingIR:

    @staticmethod
    def pass_manager(context):
        return _TraceFailingPassManager(context)


class _NoTraceFailingPassManager(_FakePassManager):

    def run(self, module, stage_name=""):
        self.ran = True
        self.last_stage_name = stage_name
        module.ran_pm = self
        module.attrs.pop("rpu.plan.trace", None)
        raise RuntimeError("PassManager::run failed")


class _NoTraceFailingIR:

    @staticmethod
    def pass_manager(context):
        return _NoTraceFailingPassManager(context)


def test_rpuplan_mlir_pass_failure_records_trace_metadata(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setattr(compiler, "ir", _TraceFailingIR)
    metadata = {}

    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compiler.RPUBackend.make_rpuplan(_FakeMLIRModule(_add_plan_json()), metadata)

    assert metadata["rpu_plan_trace"]["matched"] is False
    attempt = metadata["rpu_plan_trace"]["attempts"][0]
    assert attempt["anchor"]["kind"] == "op"
    assert attempt["anchor"]["op"] == "scf.if"
    assert attempt["location"] == attempt["anchor"]["location"]
    assert "_rpu_direct_rpurc_result" not in metadata
    assert metadata["rpu_pipeline_events"][-1]["stage"] == "rpuplan_recognize"
    assert metadata["rpu_pipeline_events"][-1]["event"] == "fail"


class _MalformedTraceFailingPassManager(_FakePassManager):

    def run(self, module, stage_name=""):
        self.ran = True
        self.last_stage_name = stage_name
        module.ran_pm = self
        module.attrs["rpu.plan.trace"] = "{"
        raise RuntimeError("PassManager::run failed")


class _MalformedTraceFailingIR:

    @staticmethod
    def pass_manager(context):
        return _MalformedTraceFailingPassManager(context)


def test_rpuplan_mlir_pass_failure_ignores_malformed_trace_metadata(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setattr(compiler, "ir", _MalformedTraceFailingIR)
    metadata = {}

    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compiler.RPUBackend.make_rpuplan(_FakeMLIRModule(_add_plan_json()), metadata)

    assert "rpu_plan_trace" not in metadata
    assert "_rpu_direct_rpurc_result" not in metadata
    assert metadata["rpu_pipeline_events"][-1]["stage"] == "rpuplan_recognize"
    assert metadata["rpu_pipeline_events"][-1]["event"] == "fail"


def test_rpuplan_mlir_pass_path_tolerates_missing_trace_metadata(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    metadata = {}

    compiler.RPUBackend._make_rpuplan_from_mlir_module(_FakeMLIRModule(_add_plan_json()), metadata)

    assert "rpu_plan_trace" not in metadata


def test_make_ttir_runs_cleanup_and_preserves_module_by_default(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    module = _FakeMLIRModule("{}")
    metadata = {}

    assert compiler.RPUBackend.make_ttir(module, metadata) is module
    assert metadata["shared"] == 0
    assert module.ran_pm.added == [
        "common.inliner",
        "ttir.rewrite_tensor_pointer",
        "ttir.combine",
        "common.canonicalizer",
        "ttir.reorder_broadcast",
        "common.cse",
        "common.licm",
        "common.symbol_dce",
    ]


def test_rpuplan_mlir_path_reports_missing_plan_json_artifact(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    module = _FakeMLIRModule(_add_plan_json())
    module.attrs.pop("rpu.plan.json")

    with pytest.raises(RuntimeError, match="failed to produce rpu\\.plan\\.json artifact"):
        compiler.RPUBackend._make_rpuplan_from_mlir_module(module, {})


class _StringLikeTTIR:

    def __str__(self):
        return _add_ttir()


def test_rpuplan_string_like_non_mlir_object_rejects_legacy_fallback(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    with pytest.raises(RuntimeError, match="TTIR text fallback is disabled"):
        compiler.RPUBackend.make_rpuplan(_StringLikeTTIR(), {})


def test_rpuplan_string_ttir_rejects_legacy_fallback(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    with pytest.raises(RuntimeError, match="TTIR text fallback is disabled"):
        compiler.RPUBackend.make_rpuplan(_add_ttir(), {})


def test_rpurc_requires_rpuexec_stage_bundle(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    metadata = {}
    with pytest.raises(RuntimeError, match="RPU make_rpurc requires RPUExecutableStageBundle from rpuexec stage"):
        compiler.RPUBackend.make_rpurc("{not-json", metadata)

    assert "name" not in metadata
    assert "rpu_pattern" not in metadata
    assert "shared" not in metadata
    fail = metadata["rpu_pipeline_events"][-1]
    assert fail["stage"] == "rpurc_emit"
    assert fail["event"] == "fail"
    assert fail["detail"]["failure_kind"] == "missing_rpuexec_bundle"
    assert fail["detail"]["kernel_name"] is None
    assert fail["detail"]["pattern"] is None
    assert fail["detail"]["source"] == "rpu_plan.kernel"
    assert "RPUExecutableStageBundle" in fail["detail"]["error"]


def test_rpurc_consumes_stage_bundle_not_metadata_side_channel(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setattr(
        _FakeRPU,
        "summary",
        {
            "kernel_name": "dialect_kernel",
            "pattern": "add",
        },
    )
    bundle = compiler.RPUPlanStageBundle(
        module=_FakeMLIRModule(_add_plan_json()),
        kernel_name="dialect_kernel",
        pattern="add",
    )
    metadata = {}

    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, bundle, metadata)

    assert "dialect_kernel" in rpurc
    assert "_rpu_plan_module" not in metadata
    assert metadata["name"] == "dialect_kernel"
    assert metadata["rpu_pattern"] == "add"
    assert metadata["rpu_rpurc_source_kind"] == "rpu_plan_kernel"
    assert metadata["rpu_pipeline_events"][-1] == {
        "stage": "rpurc_emit",
        "event": "rpu_plan_kernel",
        "detail": {
            "kernel_name": "dialect_kernel",
            "pattern": "add",
            "source": "rpu_plan.kernel",
        },
    }


def test_rpurc_emits_from_stage_bundle(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json())
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)
    assert isinstance(plan_result, compiler.RPUPlanStageBundle)
    assert plan_result.module is module
    assert "_rpu_plan_module" not in metadata
    assert "_rpu_direct_rpurc_result" not in metadata
    assert "rpu_rpurc_source_kind" not in metadata
    assert ("rpurc_emit", "fail") not in [(event["stage"], event["event"]) for event in metadata["rpu_pipeline_events"]]
    assert ("rpurc_emit", "rpu_plan_kernel") not in [(event["stage"], event["event"])
                                                     for event in metadata["rpu_pipeline_events"]]

    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "rpu_add_kernel" in rpurc
    assert "_rpu_plan_module" not in metadata
    assert "_rpu_direct_rpurc_result" not in metadata
    assert metadata["rpu_rpurc_source_kind"] == "rpu_plan_kernel"
    assert metadata["rpu_pipeline_events"][-1] == {
        "stage": "rpurc_emit",
        "event": "rpu_plan_kernel",
        "detail": {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
            "source": "rpu_plan.kernel",
        },
    }


def test_rpurc_uses_direct_result_as_authoritative_emission_identity(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setattr(
        _FakeRPU,
        "direct_result",
        {
            "kernel_name": "wrong_kernel",
            "pattern": "add",
            "source": "__rprog__ void wrong_kernel() {}\n",
        },
    )
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json())
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)

    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "__rprog__ void wrong_kernel()" in rpurc
    assert "_rpu_plan_module" not in metadata
    assert "_rpu_direct_rpurc_result" not in metadata
    assert metadata["name"] == "wrong_kernel"
    assert metadata["rpu_pattern"] == "add"
    assert metadata["rpu_pipeline_events"][-1] == {
        "stage": "rpurc_emit",
        "event": "rpu_plan_kernel",
        "detail": {
            "kernel_name": "wrong_kernel",
            "pattern": "add",
            "source": "rpu_plan.kernel",
        },
    }


def _convkxk_plan_json():
    return json.dumps(
        {
            "args": {"input": 1, "out": 0, "weight": 2},
            "emission": {
                "kind": "convkxk",
                "kernel_size": 3,
                "m": 16,
                "in_channels": 16,
                "out_channels": 16,
                "input_width": 16,
                "input": 1,
                "out": 0,
                "weight": 2,
            },
            "kernel_name": "rpu_convkxk_kernel",
            "layout": {
                "memory": "array2d",
                "access": "row_window",
                "order": "row_major",
                "window": {
                    "kernel_size": 3,
                    "input_width": 16,
                    "stride": [1, 1],
                    "padding": [0, 0],
                },
                "tile": {"m": 16, "n": 16},
            },
            "mask": {"masked": False},
            "pattern": "convkxk",
            "required_dsl_features": [
                "rpu.Array",
                "ctx.load",
                "ctx.zeros",
                "ctx.mma",
                "ctx.store",
            ],
            "shape": {
                "kernel_size": 3,
                "m": 16,
                "in_channels": 16,
                "out_channels": 16,
                "input_width": 16,
            },
            "signature": {
                "params": [
                    {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
                    {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
                    {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
                ],
                "return_type":
                "void",
            },
            "version": 1,
        },
        sort_keys=True,
    )


def _residual_plan_json(hidden=16):
    return json.dumps(
        {
            "args": {"out": 0, "w1": 2, "w2": 3, "x": 1},
            "emission": {
                "kind": "resnet_block",
                "m": 16,
                "channels": 16,
                "hidden": hidden,
                "out": 0,
                "x": 1,
                "w1": 2,
                "w2": 3,
            },
            "kernel_name":
            "rpu_resnet_block_kernel",
            "layout": {
                "memory": "array2d",
                "access": "matrix_tile",
                "order": "row_major",
            },
            "mask": {"masked": False},
            "pattern":
            "resnet_block",
            "required_dsl_features": [
                "rpu.Array",
                "ctx.load",
                "ctx.zeros",
                "ctx.mma",
                "tile.add",
                "rpu.max_binop",
                "ctx.store",
            ],
            "shape": {"m": 16, "channels": 16, "hidden": hidden},
            "signature": {
                "params": [
                    {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
                    {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
                    {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
                    {"index": 3, "name": "arg3", "kind": "ptr", "element_type": "f16"},
                ],
                "return_type":
                "void",
            },
            "version":
            1,
        },
        sort_keys=True,
    )


def _bottleneck_plan_json(bottleneck=16):
    return json.dumps(
        {
            "args": {"input": 1, "out": 0, "w1": 2, "w2": 3, "w3": 4},
            "emission": {
                "kind": "resnet50_bottleneck",
                "kernel_size": 3,
                "m": 16,
                "channels": 16,
                "bottleneck": bottleneck,
                "input_width": 16,
                "out": 0,
                "input": 1,
                "w1": 2,
                "w2": 3,
                "w3": 4,
            },
            "kernel_name":
            "rpu_resnet50_bottleneck_kernel",
            "layout": {
                "memory": "array2d",
                "access": "bottleneck_row_window",
                "order": "row_major",
                "window": {
                    "kernel_size": 3,
                    "input_width": 16,
                    "stride": [1, 1],
                    "padding": [0, 0],
                },
                "tile": {"m": 16, "n": 16},
            },
            "mask": {"masked": False},
            "pattern":
            "resnet50_bottleneck",
            "required_dsl_features": [
                "rpu.Array",
                "ctx.load",
                "ctx.zeros",
                "ctx.mma",
                "tile.add",
                "rpu.max_binop",
                "ctx.store",
            ],
            "shape": {
                "kernel_size": 3,
                "m": 16,
                "channels": 16,
                "bottleneck": bottleneck,
                "input_width": 16,
            },
            "signature": {
                "params": [
                    {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
                    {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
                    {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
                    {"index": 3, "name": "arg3", "kind": "ptr", "element_type": "f16"},
                    {"index": 4, "name": "arg4", "kind": "ptr", "element_type": "f16"},
                ],
                "return_type":
                "void",
            },
            "version":
            1,
        },
        sort_keys=True,
    )


def test_rpurc_fails_when_recognized_pattern_has_no_direct_emitter(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setattr(
        _FakeRPU,
        "summary",
        {
            "kernel_name": "rpu_convkxk_kernel",
            "pattern": "convkxk",
        },
    )
    monkeypatch.setattr(_FakeRPU, "supported_patterns", ("add", "gemm", "softmax"))
    metadata = {}
    module = _FakeMLIRModule(_convkxk_plan_json())

    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)
    assert isinstance(plan_result, compiler.RPUPlanStageBundle)
    assert "_rpu_plan_module" not in metadata
    assert ("rpurc_emit", "fail") not in [(event["stage"], event["event"]) for event in metadata["rpu_pipeline_events"]]

    with pytest.raises(RuntimeError, match="does not have a .* rpurc stage emitter"):
        _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert metadata.get("rpu_rpurc_source_kind") != "json_compat"
    assert "_rpu_plan_module" not in metadata
    fail = metadata["rpu_pipeline_events"][-1]
    _assert_rpuexec_fail_detail(
        fail,
        pattern="convkxk",
        error_contains="does not have a",
        extra={"supported_patterns": ["add", "gemm", "softmax"]},
    )


def test_rpurc_direct_supported_pattern_failure_is_not_json_compat(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setattr(
        _FakeRPU,
        "summary",
        {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
    )
    monkeypatch.setattr(_FakeRPU, "direct_error", "direct emitter failed")
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json())

    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)
    assert isinstance(plan_result, compiler.RPUPlanStageBundle)
    assert "_rpu_plan_module" not in metadata

    with pytest.raises(RuntimeError, match="RPU rpurc stage emission failed for pattern add"):
        _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "_rpu_plan_module" not in metadata
    assert [(event["stage"], event["event"]) for event in metadata["rpu_pipeline_events"]] == [
        ("rpuplan_recognize", "start"),
        ("rpuplan_recognize", "end"),
        ("rpuexec_lower", "rpu_plan_kernel"),
        ("rpurc_emit", "fail"),
    ]
    fail = metadata["rpu_pipeline_events"][-1]
    assert fail["detail"]["failure_kind"] == "direct_emitter_failure"
    assert fail["detail"]["kernel_name"] == "rpu_add_kernel"
    assert fail["detail"]["pattern"] == "add"
    assert fail["detail"]["source"] == "rpu_plan.kernel"
    assert fail["detail"]["error"] == "direct emitter failed"


def test_rpu_backend_no_legacy_regex_matcher_surface(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    forbidden = [
        "_extract_kernel_name",
        "_extract_kernel_signature",
        "_trace_pointer_args",
        "_select_pattern_plan",
        "_pattern_passes",
        "_pattern_trace_enabled",
        "_format_pattern_trace",
        "_make_kernel_plan",
        "_use_mlir_plan_recognizer",
        "_match_add_kernel",
        "_match_gemm_kernel",
        "_match_softmax_kernel",
        "_match_resnet_block_kernel",
        "_match_convkxk_kernel",
        "_match_resnet50_bottleneck_kernel",
    ]
    assert all(not hasattr(compiler.RPUBackend, name) for name in forbidden)


def test_rpurc_rejects_unknown_plan_pattern_from_stage_bundle(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    bundle = compiler.RPUPlanStageBundle(
        module=_FakeMLIRModule(_add_plan_json()),
        kernel_name="rpu_add_kernel",
        pattern="unsupported",
    )
    metadata = {}

    with pytest.raises(RuntimeError, match="does not have a .* rpurc stage emitter"):
        _make_rpurc_from_rpuplan_bundle(compiler, bundle, metadata)

    assert "_rpu_plan_module" not in metadata
    _assert_rpuexec_fail_detail(
        metadata["rpu_pipeline_events"][-1],
        pattern="unsupported",
        error_contains="does not have a",
    )


def test_rpuinstr_records_compile_debug_metadata(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")
    instr_payload = "0706050403020100\n"
    bin_payload = bytes.fromhex("0001020304050607")
    captured = {}

    def fake_run(cmd, cwd, env, text, capture_output):
        captured["cmd"] = cmd
        captured["cwd"] = Path(cwd)
        assert text is True
        assert capture_output is True
        assert env["TMPDIR"] == cwd
        (Path(cwd) / "rpu_debug_kernel.o").write_text(instr_payload)
        (Path(cwd) / "rpu_debug_kernel-0.bin").write_bytes(bin_payload)
        return types.SimpleNamespace(returncode=0, stdout="compiled ok\n", stderr="note\n")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {"name": "rpu_debug_kernel"}

    rpuinstr = compiler.RPUBackend.make_rpuinstr(
        "__rprog__ void rpu_debug_kernel() {}",
        metadata,
        compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
    )

    assert rpuinstr == instr_payload
    assert metadata["rpu_clang_command"] == captured["cmd"]
    assert metadata["rpu_clang_command"][0] == str(clang)
    assert metadata["rpu_clang_command"][-2] == "-c"
    assert metadata["rpu_clang_command"][-1].endswith("rpu_debug_kernel.rc")
    assert metadata["rpu_clang_output"] == {"stdout": "compiled ok", "stderr": "note"}
    assert metadata["rpu_artifacts"] == {
        "rpubin": {
            "bytes": len(bin_payload),
            "sha256": hashlib.sha256(bin_payload).hexdigest(),
        },
        "rpuinstr": {
            "bytes": len(instr_payload.encode()),
            "sha256": hashlib.sha256(instr_payload.encode()).hexdigest(),
        },
    }


def test_rpu_debug_bundle_disabled_by_default(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    monkeypatch.delenv("RPU_DEBUG_ARTIFACT_DIR", raising=False)
    import compiler

    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")

    def fake_run(cmd, cwd, env, text, capture_output):
        (Path(cwd) / "rpu_debug_kernel.o").write_text("0706050403020100\n")
        (Path(cwd) / "rpu_debug_kernel-0.bin").write_bytes(bytes.fromhex("0001020304050607"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {"name": "rpu_debug_kernel"}

    compiler.RPUBackend.make_rpuinstr(
        "__rprog__ void rpu_debug_kernel() {}",
        metadata,
        compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
    )

    assert "rpu_debug_bundle" not in metadata


def test_rpu_debug_bundle_records_success_path(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    debug_dir = tmp_path / "debug"
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(debug_dir))
    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")
    instr_payload = "0706050403020100\n"
    bin_payload = bytes.fromhex("0001020304050607")
    captured = {}

    def fake_run(cmd, cwd, env, text, capture_output):
        captured["cwd"] = Path(cwd)
        captured["source"] = Path(cmd[-1])
        (Path(cwd) / "rpu_add_kernel.o").write_text(instr_payload)
        (Path(cwd) / "rpu_add_kernel-0.bin").write_bytes(bin_payload)
        return types.SimpleNamespace(returncode=0, stdout="compiled ok\n", stderr="note\n")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())

    compiler.RPUBackend.make_ttir(module, metadata)
    ttir_mlir = str(module)
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)
    plan_json = str(plan_result)
    _bundle_before_rpurc, bundle_dir_before_rpurc, manifest_before_rpurc = _debug_bundle(metadata)
    assert "rpu.plan.json" in _debug_files(bundle_dir_before_rpurc)
    assert "rpu.plan.mlir" in _debug_files(bundle_dir_before_rpurc)
    assert "rpurc.rc" not in _debug_files(bundle_dir_before_rpurc)
    assert ("rpurc_emit", "rpu_plan_kernel") not in [(event["stage"], event["event"])
                                                     for event in manifest_before_rpurc["pipeline_events"]]
    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)
    rpuinstr = compiler.RPUBackend.make_rpuinstr(
        rpurc,
        metadata,
        compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
    )
    rpubin = compiler.RPUBackend.make_rpubin(rpuinstr, metadata)

    assert rpubin == bin_payload
    bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert Path(bundle["dir"]).is_absolute()
    assert Path(bundle["manifest"]).is_absolute()
    assert _debug_files(bundle_dir) == {
        "ttir.mlir",
        "rpu.plan.json",
        "rpu.plan.mlir",
        "rpu.plan.trace.json",
        "rpurc.rc",
        "clang_command.json",
        "clang_stdout.txt",
        "clang_stderr.txt",
        "rpuinstr.txt",
        "rpubin.bin",
        "manifest.json",
    }
    assert (bundle_dir / "ttir.mlir").read_text() == ttir_mlir
    assert (bundle_dir / "rpu.plan.json").read_text() == plan_json
    rpuplan_mlir = (bundle_dir / "rpu.plan.mlir").read_text()
    assert "rpu_plan.kernel @rpu_add_kernel_plan" in rpuplan_mlir
    assert "RPUPlanStageBundle" not in rpuplan_mlir
    assert json.loads((bundle_dir / "rpu.plan.trace.json").read_text())["selected"] == "add"
    assert (bundle_dir / "rpurc.rc").read_text() == rpurc
    assert (bundle_dir / "clang_stdout.txt").read_text() == "compiled ok\n"
    assert (bundle_dir / "clang_stderr.txt").read_text() == "note\n"
    assert (bundle_dir / "rpuinstr.txt").read_text() == instr_payload
    assert (bundle_dir / "rpubin.bin").read_bytes() == bin_payload
    command = json.loads((bundle_dir / "clang_command.json").read_text())
    assert command["cwd"] == "."
    assert command["argv"][-2:] == ["-c", "rpurc.rc"]
    assert "triton_rpu_" not in json.dumps(command)
    assert str(captured["cwd"]) not in json.dumps(command)
    assert str(captured["source"]) not in json.dumps(command)
    assert manifest["version"] == 1
    assert manifest["kernel_name"] == "rpu_add_kernel"
    assert manifest["pattern"] == "add"
    assert manifest["files"] == bundle["files"]
    assert set(manifest["files"]) == {
        "ttir",
        "rpuplan",
        "rpuplan_mlir",
        "rpuplan_trace",
        "rpurc",
        "clang_command",
        "clang_stdout",
        "clang_stderr",
        "rpuinstr",
        "rpubin",
    }
    rpurc_events = [
        event for event in manifest["pipeline_events"]
        if event["stage"] == "rpurc_emit" and event["event"] == "rpu_plan_kernel"
    ]
    assert rpurc_events == [{
        "stage": "rpurc_emit",
        "event": "rpu_plan_kernel",
        "detail": {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
            "source": "rpu_plan.kernel",
        },
    }]
    assert [event["event"] for event in manifest["pipeline_events"]][-1] == "end"


def test_rpu_debug_bundle_records_rpuplan_mlir_after_recognizer(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())

    compiler.RPUBackend.make_ttir(module, metadata)
    _bundle, bundle_dir, _manifest = _debug_bundle(metadata)
    assert "rpu.plan.mlir" not in _debug_files(bundle_dir)

    result = compiler.RPUBackend.make_rpuplan(module, metadata)

    bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert result.module is module
    assert "rpu.plan.mlir" in _debug_files(bundle_dir)
    assert (bundle_dir / "rpu.plan.mlir").read_text() == str(module)
    assert "rpu_plan.kernel @rpu_add_kernel_plan" in (bundle_dir / "rpu.plan.mlir").read_text()
    assert "rpuplan_mlir" in bundle["files"]
    assert manifest["files"]["rpuplan_mlir"] == bundle["files"]["rpuplan_mlir"]


def test_rpu_backend_rpuexec_mlir_debug_artifact_for_direct_stage_value(
    tmp_path,
    monkeypatch,
):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ExecutableModule:

        def __str__(self):
            return 'module { rpu.kernel @rpu_add_kernel() { rpu.return } {kind = "add"} }'

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    executable_module = ExecutableModule()
    bundle = compiler.RPUExecutableStageBundle(
        module=executable_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
        source_kind="rpu_executable",
        fallback_reason=None,
    )
    metadata = {}

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert result is bundle
    debug_bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert "rpu.exec.mlir" in _debug_files(bundle_dir)
    assert (bundle_dir / "rpu.exec.mlir").read_text() == str(executable_module)
    assert "rpuexec_mlir" in debug_bundle["files"]
    assert manifest["files"]["rpuexec_mlir"] == debug_bundle["files"]["rpuexec_mlir"]


def test_rpu_backend_rpuexec_mlir_debug_artifact_for_plan_lowering_success(
    tmp_path,
    monkeypatch,
):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ExecutableModule:

        def __str__(self):
            return 'module { rpu.kernel @rpu_add_kernel() { rpu.return } {kind = "add"} }'

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    plan_module = object()
    executable_module = ExecutableModule()
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: module is plan_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: executable_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is executable_module
    debug_bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert "rpu.exec.mlir" in _debug_files(bundle_dir)
    assert (bundle_dir / "rpu.exec.mlir").read_text() == str(executable_module)
    assert "rpuexec_mlir" in debug_bundle["files"]
    assert manifest["files"]["rpuexec_mlir"] == debug_bundle["files"]["rpuexec_mlir"]


def test_rpu_backend_legacy_direct_fallback_has_no_rpuexec_mlir_debug_artifact(
    tmp_path,
    monkeypatch,
):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    bundle = compiler.RPUPlanStageBundle(
        module=_FakeMLIRModule(_add_plan_json()),
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.source_kind == "rpu_plan_kernel"
    assert result.fallback_reason == "missing_executable_hooks"
    assert "rpu_debug_bundle" not in metadata


def test_rpu_backend_direct_executable_success_has_own_pipeline_events(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    direct_events = [event for event in metadata["rpu_pipeline_events"] if event["stage"] == "direct_executable_lower"]
    assert [event["event"] for event in direct_events] == [
        "start",
        "candidate",
        "artifact",
        "end",
    ]
    assert direct_events[0]["detail"]["patterns"] == [
        "add",
        "gemm",
        "softmax",
        "convkxk",
        "resnet_block",
        "resnet50_bottleneck",
    ]
    assert direct_events[1]["detail"]["status"] == "available"
    assert direct_events[2]["detail"]["source_boundary"] == "rpu_executable"
    assert direct_events[2]["detail"]["artifact_kind"] == "rpu_exec_mlir"
    assert direct_events[3]["detail"]["artifact_source"] == "rpu_exec_mlir"
    compatibility_events = [
        event["event"] for event in metadata["rpu_pipeline_events"] if event["stage"] == "rpuplan_recognize"
    ]
    assert compatibility_events == []


def test_rpu_backend_direct_success_uses_python_pass_pipeline_not_private_helper(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    pass_calls = []

    def add_lower_supported(pm):
        pass_calls.append(pm)
        pm.added.append("lower_supported_ttir_to_executable")

    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_lower_supported_ttir_to_executable",
        add_lower_supported,
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert pass_calls == [source_module.ran_pm]
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is source_module
    assert result.source_kind == "rpu_executable"
    assert source_module.has_rpuexec_kernel is True
    assert source_module.has_rpuplan_kernel is False
    assert all(event["stage"] != "rpuplan_recognize" for event in metadata["rpu_pipeline_events"])


def test_rpu_backend_direct_executable_skip_falls_back_to_recognizer(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    compiler.rpu.direct_pass_error = "not supported"
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUPlanStageBundle)
    assert source_module.ran_pm is not None
    direct_events = [event for event in metadata["rpu_pipeline_events"] if event["stage"] == "direct_executable_lower"]
    assert [event["event"] for event in direct_events] == ["start", "skip"]
    assert direct_events[-1]["detail"]["reason"] == "unsupported_ttir"
    assert "not supported" in direct_events[-1]["detail"]["error"]
    assert [event["event"]
            for event in metadata["rpu_pipeline_events"]
            if event["stage"] == "rpuplan_recognize"] == ["start", "end"]


def test_rpu_backend_direct_executable_summary_failure_has_own_fail_event(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: (_ for _ in ()).throw(RuntimeError("summary exploded")),
        raising=False,
    )
    metadata = {}

    with pytest.raises(RuntimeError, match="direct executable stage summary failed"):
        compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    event_pairs = [(event["stage"], event["event"]) for event in metadata["rpu_pipeline_events"]]
    assert ("direct_executable_lower", "fail") in event_pairs
    assert all(stage != "rpuplan_recognize" for stage, _ in event_pairs)
    fail = [
        event for event in metadata["rpu_pipeline_events"]
        if event["stage"] == "direct_executable_lower" and event["event"] == "fail"
    ][0]
    assert "RPU executable stage summary failed" in fail["detail"]["error"]


def _add_exec_mlir_text():
    return """
module {
  rpu.kernel @rpu_add_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
    %lhs = rpu.load_contig %arg1 {n = 16 : i32} : !tt.ptr<f16> -> <16xf16>
    %rhs = rpu.load_contig %arg2 {n = 16 : i32} : !tt.ptr<f16> -> <16xf16>
    %sum = rpu.add %lhs, %rhs : <16xf16>
    rpu.store_contig %arg0, %sum {logical_n = 16 : i32, masked = false, n = 16 : i32} : !tt.ptr<f16>, <16xf16>
    rpu.return
  } {kind = "add"}
}
""".strip()


def test_rpu_backend_direct_artifact_prefers_native_executable_ir(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.RPUBackend,
        "_serialize_rpuexec_plan_sidecar_from_stale_text_compat",
        staticmethod(lambda module:
                     (_ for _ in ()).throw(AssertionError("stale text compatibility serializer must not run"))),
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(
        source_module,
        metadata,
    )

    assert str(result) == str(source_module)
    assert "rpu_direct_executable_sidecar_serializer" not in metadata
    artifact_event = [
        event for event in metadata["rpu_pipeline_events"]
        if event["stage"] == "direct_executable_lower" and event["event"] == "artifact"
    ][0]
    assert artifact_event["detail"]["artifact_kind"] == "rpu_exec_mlir"


def test_rpu_backend_missing_native_serializer_does_not_block_direct_artifact(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(
        source_module,
        metadata,
    )

    assert str(result) == str(source_module)
    assert "rpu_direct_executable_sidecar_serializer" not in metadata
    artifact_event = [
        event for event in metadata["rpu_pipeline_events"]
        if event["stage"] == "direct_executable_lower" and event["event"] == "artifact"
    ][0]
    assert artifact_event["detail"]["artifact_kind"] == "rpu_exec_mlir"
    assert "python_text_compat" not in json.dumps(metadata, sort_keys=True)


def test_rpu_backend_direct_artifact_has_no_text_compat_fallback():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    direct_helper_body = _source_between(
        source,
        "    def _try_make_direct_executable_stage_from_ttir(",
        "    @staticmethod\n    def _make_rpuplan_stage_from_mlir_module_via_recognizer",
    )

    assert "def _serialize_rpuexec_plan_sidecar_from_text(" not in source
    assert "_serialize_direct_executable_plan_sidecar" not in source
    assert "RPUBackend._serialize_rpuexec_plan_sidecar_from_stale_text_compat" not in direct_helper_body
    assert "python_text_compat" not in direct_helper_body
    assert "missing_native_serializer" not in direct_helper_body


def test_rpu_backend_stale_text_sidecar_compat_parser_removed():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()

    for token in [
            "import re",
            "_RPU_EXEC_",
            "_serialize_rpuexec_plan_sidecar_from_stale_text_compat",
            "_parse_rpuexec_",
            "_f16_signature",
            "_add_sidecar_features",
            "_matrix_features",
            "_resnet_features",
            "_build_add_sidecar",
            "_build_gemm_sidecar",
            "_build_softmax_sidecar",
            "_matrix_input_width",
            "_build_convkxk_sidecar",
            "_arg_extent",
            "_build_resnet_block_sidecar",
            "_build_resnet50_bottleneck_sidecar",
            "python_text_compat",
            "missing_native_serializer",
    ]:
        assert token not in source


def test_rpu_debug_bundle_records_recognizer_failure_trace(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json())
    compiler.RPUBackend.make_ttir(module, metadata)
    monkeypatch.setattr(compiler, "ir", _TraceFailingIR)

    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        compiler.RPUBackend.make_rpuplan(module, metadata)

    _bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert "ttir.mlir" in _debug_files(bundle_dir)
    assert json.loads((bundle_dir / "rpu.plan.trace.json").read_text())["matched"] is False
    assert ("rpuplan_recognize", "fail") in [(event["stage"], event["event"]) for event in manifest["pipeline_events"]]


def test_rpu_debug_bundle_ignores_missing_or_malformed_trace(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    for ir_cls in (_NoTraceFailingIR, _MalformedTraceFailingIR):
        monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / ir_cls.__name__))
        monkeypatch.setattr(compiler, "ir", _FakeIR)
        metadata = {}
        module = _FakeMLIRModule(_add_plan_json())
        compiler.RPUBackend.make_ttir(module, metadata)
        monkeypatch.setattr(compiler, "ir", ir_cls)

        with pytest.raises(RuntimeError, match="PassManager::run failed"):
            compiler.RPUBackend.make_rpuplan(module, metadata)

        _bundle, bundle_dir, manifest = _debug_bundle(metadata)
        assert "ttir.mlir" in _debug_files(bundle_dir)
        assert "rpu.plan.trace.json" not in _debug_files(bundle_dir)
        assert ("rpuplan_recognize", "fail") in [(event["stage"], event["event"])
                                                 for event in manifest["pipeline_events"]]


def test_rpu_debug_bundle_records_missing_plan_json_artifact_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())
    module.attrs.pop("rpu.plan.json")
    compiler.RPUBackend.make_ttir(module, metadata)

    with pytest.raises(RuntimeError, match="failed to produce rpu\\.plan\\.json artifact"):
        compiler.RPUBackend.make_rpuplan(module, metadata)

    _bundle, _bundle_dir, manifest = _debug_bundle(metadata)
    assert ("rpuplan_recognize", "fail") in [(event["stage"], event["event"]) for event in manifest["pipeline_events"]]


def test_rpu_debug_bundle_records_unsupported_rpurc_failure_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    monkeypatch.setattr(
        _FakeRPU,
        "summary",
        {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
    )
    monkeypatch.setattr(_FakeRPU, "supported_patterns", ())
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())
    compiler.RPUBackend.make_ttir(module, metadata)
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)

    _bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert "rpurc.rc" not in _debug_files(bundle_dir)
    assert ("rpurc_emit", "fail") not in [(event["stage"], event["event"]) for event in manifest["pipeline_events"]]

    with pytest.raises(RuntimeError, match="does not have a .* rpurc stage emitter"):
        _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "_rpu_plan_module" not in metadata
    _bundle, _bundle_dir, manifest = _debug_bundle(metadata)
    fail = manifest["pipeline_events"][-1]
    _assert_rpuexec_fail_detail(
        fail,
        pattern="add",
        error_contains="does not have a",
        extra={"supported_patterns": []},
    )


def test_rpu_debug_bundle_records_rpurc_emitter_failure_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    monkeypatch.setattr(_FakeRPU, "direct_error", "direct emitter broke")
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())
    compiler.RPUBackend.make_ttir(module, metadata)
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)

    with pytest.raises(RuntimeError, match="rpurc stage emission failed"):
        _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "_rpu_plan_module" not in metadata
    _bundle, _bundle_dir, manifest = _debug_bundle(metadata)
    fail = manifest["pipeline_events"][-1]
    _assert_rpurc_fail_detail(
        fail,
        kind="direct_emitter_failure",
        kernel_name="rpu_add_kernel",
        pattern="add",
        error_contains="direct emitter broke",
    )


def test_rpu_debug_bundle_uses_stage_bundle_for_rpurc(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)

    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "rpu_add_kernel" in rpurc
    assert "_rpu_plan_module" not in metadata
    _bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert "rpurc.rc" in _debug_files(bundle_dir)
    assert manifest["pipeline_events"][-1]["event"] == "rpu_plan_kernel"


def test_rpu_debug_bundle_records_missing_stage_bundle_failure_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    metadata = {}

    with pytest.raises(RuntimeError, match="requires RPUExecutableStageBundle"):
        compiler.RPUBackend.make_rpurc("{not-json", metadata)

    _bundle, _bundle_dir, manifest = _debug_bundle(metadata)
    fail = manifest["pipeline_events"][-1]
    _assert_rpurc_fail_detail(
        fail,
        kind="missing_rpuexec_bundle",
        kernel_name=None,
        pattern=None,
        error_contains="requires RPUExecutableStageBundle",
    )


def test_rpu_debug_bundle_records_direct_result_identity_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    monkeypatch.setattr(
        _FakeRPU,
        "direct_result",
        {
            "kernel_name": "wrong_kernel",
            "pattern": "wrong_pattern",
            "source": "__rprog__ void wrong_kernel() {}\n",
        },
    )
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json(), _trace_json())
    compiler.RPUBackend.make_ttir(module, metadata)
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)

    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)

    assert "__rprog__ void wrong_kernel()" in rpurc
    assert "_rpu_plan_module" not in metadata
    _bundle, _bundle_dir, manifest = _debug_bundle(metadata)
    event = manifest["pipeline_events"][-1]
    assert event == {
        "stage": "rpurc_emit",
        "event": "rpu_plan_kernel",
        "detail": {
            "kernel_name": "wrong_kernel",
            "pattern": "wrong_pattern",
            "source": "rpu_plan.kernel",
        },
    }
    assert manifest["kernel_name"] == "wrong_kernel"
    assert manifest["pattern"] == "wrong_pattern"


def test_rpu_debug_bundle_records_rpuinstr_failure(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")
    captured = {}

    def fake_run(cmd, cwd, env, text, capture_output):
        captured["cwd"] = Path(cwd)
        captured["source"] = Path(cmd[-1])
        return types.SimpleNamespace(returncode=17, stdout="before fail\n", stderr="bad rc\n")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {"name": "rpu_bad_kernel"}

    with pytest.raises(RuntimeError, match="RPU clang failed"):
        compiler.RPUBackend.make_rpuinstr(
            "__rprog__ void rpu_bad_kernel() {}",
            metadata,
            compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
        )

    _bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert {"rpurc.rc", "clang_command.json", "clang_stdout.txt", "clang_stderr.txt",
            "manifest.json"} <= _debug_files(bundle_dir)
    assert "rpuinstr.txt" not in _debug_files(bundle_dir)
    assert "rpubin.bin" not in _debug_files(bundle_dir)
    assert (bundle_dir / "clang_stdout.txt").read_text() == "before fail\n"
    assert (bundle_dir / "clang_stderr.txt").read_text() == "bad rc\n"
    command = json.loads((bundle_dir / "clang_command.json").read_text())
    assert command["argv"][-2:] == ["-c", "rpurc.rc"]
    assert "triton_rpu_" not in json.dumps(command)
    assert str(captured["cwd"]) not in json.dumps(command)
    assert str(captured["source"]) not in json.dumps(command)
    assert ("rpuinstr_compile", "fail") in [(event["stage"], event["event"]) for event in manifest["pipeline_events"]]


def test_rpu_debug_bundle_records_missing_compiler_outputs(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")

    def fake_run(cmd, cwd, env, text, capture_output):
        return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {"name": "rpu_missing_outputs"}

    with pytest.raises(RuntimeError, match="did not emit expected outputs"):
        compiler.RPUBackend.make_rpuinstr(
            "__rprog__ void rpu_missing_outputs() {}",
            metadata,
            compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
        )

    _bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert {"rpurc.rc", "clang_command.json", "clang_stdout.txt", "clang_stderr.txt",
            "manifest.json"} <= _debug_files(bundle_dir)
    assert "rpuinstr.txt" not in _debug_files(bundle_dir)
    assert "rpubin.bin" not in _debug_files(bundle_dir)
    assert ("rpuinstr_compile", "fail") in [(event["stage"], event["event"]) for event in manifest["pipeline_events"]]


def test_rpu_debug_bundle_records_clang_command_failure(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    metadata = {"name": "rpu_missing_clang"}

    with pytest.raises(RuntimeError, match="RPU clang not found"):
        compiler.RPUBackend.make_rpuinstr(
            "__rprog__ void rpu_missing_clang() {}",
            metadata,
            compiler.RPUOptions(llvm_root=str(tmp_path / "missing_llvm"), rpu_asm_path="/fake/rpuasm"),
        )

    _bundle, bundle_dir, manifest = _debug_bundle(metadata)
    assert "rpurc.rc" in _debug_files(bundle_dir)
    assert "clang_command.json" not in _debug_files(bundle_dir)
    assert ("rpuinstr_compile", "fail") in [(event["stage"], event["event"]) for event in manifest["pipeline_events"]]


def test_rpu_debug_bundle_reports_filesystem_error(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    bad_root = tmp_path / "not_a_directory"
    bad_root.write_text("file")
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(bad_root))
    metadata = {}

    with pytest.raises(RuntimeError, match="RPU debug bundle"):
        compiler.RPUBackend.make_ttir(_FakeMLIRModule(_add_plan_json()), metadata)


def test_rpu_pipeline_events_record_success_path(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")

    def fake_run(cmd, cwd, env, text, capture_output):
        (Path(cwd) / "rpu_add_kernel.o").write_text("0706050403020100\n")
        (Path(cwd) / "rpu_add_kernel-0.bin").write_bytes(bytes.fromhex("0001020304050607"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {}
    module = _FakeMLIRModule(_add_plan_json())

    compiler.RPUBackend.make_ttir(module, metadata)
    plan_result = compiler.RPUBackend.make_rpuplan(module, metadata)
    rpurc = _make_rpurc_from_rpuplan_bundle(compiler, plan_result, metadata)
    compiler.RPUBackend.make_rpuinstr(
        rpurc,
        metadata,
        compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
    )

    events = metadata["rpu_pipeline_events"]
    assert [(event["stage"], event["event"]) for event in events] == [
        ("ttir_cleanup", "start"),
        ("ttir_cleanup", "end"),
        ("rpuplan_recognize", "start"),
        ("rpuplan_recognize", "end"),
        ("rpuexec_lower", "rpu_plan_kernel"),
        ("rpurc_emit", "rpu_plan_kernel"),
        ("rpuinstr_compile", "start"),
        ("rpuinstr_compile", "end"),
    ]
    assert events[1]["detail"]["passes"] == [
        "common.inliner",
        "ttir.rewrite_tensor_pointer",
        "ttir.combine",
        "common.canonicalizer",
        "ttir.reorder_broadcast",
        "common.cse",
        "common.licm",
        "common.symbol_dce",
    ]
    assert events[3]["detail"] == {"kernel_name": "rpu_add_kernel", "pattern": "add"}
    assert events[4]["detail"] == {
        "kernel_name": "rpu_add_kernel",
        "pattern": "add",
        "reason": "env_opt_out",
    }
    assert events[5]["detail"] == {
        "kernel_name": "rpu_add_kernel",
        "pattern": "add",
        "source": "rpu_plan.kernel",
    }
    assert events[6]["detail"]["kernel_name"] == "rpu_add_kernel"
    assert events[7]["detail"]["artifacts"] == metadata["rpu_artifacts"]


def test_rpu_pipeline_events_record_rpuinstr_failure(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    llvm_root = tmp_path / "llvm"
    clang = llvm_root / "build" / "bin" / "clang"
    clang.parent.mkdir(parents=True)
    clang.write_text("#!/bin/sh\n")

    def fake_run(cmd, cwd, env, text, capture_output):
        return types.SimpleNamespace(returncode=17, stdout="before fail\n", stderr="bad rc\n")

    monkeypatch.setattr(compiler.subprocess, "run", fake_run)
    metadata = {"name": "rpu_bad_kernel"}

    with pytest.raises(RuntimeError, match="RPU clang failed"):
        compiler.RPUBackend.make_rpuinstr(
            "__rprog__ void rpu_bad_kernel() {}",
            metadata,
            compiler.RPUOptions(llvm_root=str(llvm_root), rpu_asm_path="/fake/rpuasm"),
        )

    assert [(event["stage"], event["event"]) for event in metadata["rpu_pipeline_events"]] == [
        ("rpuinstr_compile", "start"),
        ("rpuinstr_compile", "fail"),
    ]
    fail_detail = metadata["rpu_pipeline_events"][-1]["detail"]
    assert fail_detail["kernel_name"] == "rpu_bad_kernel"
    assert fail_detail["returncode"] == 17
    assert fail_detail["stdout"] == "before fail"
    assert fail_detail["stderr"] == "bad rc"


def test_rpu_driver_active_and_target(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu-test")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("RPU_CORE_COUNT", "8")
    monkeypatch.setenv("RPU_LOCAL_SPM_BYTES", "65536")
    monkeypatch.setenv("RPU_VLM_BYTES", "32768")

    import driver

    drv = driver.RPUDriver()
    target = drv.get_current_target()
    props = drv.utils.get_device_properties(0)
    assert driver.RPUDriver.is_active() is True
    assert target.backend == "rpu"
    assert target.arch == "rpu-test"
    assert target.warp_size == 16
    assert props["rpu_core_count"] == 8
    assert props["rpu_local_spm_bytes"] == 65536
    assert props["rpu_vlm_bytes"] == 32768


def test_rpu_aot_kernel_writes_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    artifact = tmp_path / "kernel.rpubin"
    artifact.write_bytes(b"fake-rpu-binary")
    captured = tmp_path / "captured.json"
    launcher = tmp_path / "fake_launcher.py"
    launcher.write_text("#!/usr/bin/env python3\n"
                        "import sys\n"
                        f"open({str(captured)!r}, 'w').write(open(sys.argv[1]).read())\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)

    import aot

    result = aot.RPUAOTKernel(
        artifact,
        kernel_name="vec_add",
        launcher=str(launcher),
        metadata={"core_count": 8},
    ).run((1, 2, 3), "arg0", 7)

    manifest = json.loads(captured.read_text())
    assert result.returncode == 0
    assert manifest["artifact_path"] == str(artifact.resolve())
    assert manifest["kernel_name"] == "vec_add"
    assert manifest["grid"] == [1, 2, 3]
    assert manifest["args"] == ["arg0", 7]
    assert manifest["metadata"] == {"core_count": 8}


def test_rpu_aot_kernel_reports_launcher_failure_and_cleans_manifest(tmp_path, monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    artifact = tmp_path / "kernel.rpubin"
    artifact.write_bytes(b"fake-rpu-binary")
    captured_manifest_path = tmp_path / "manifest_path.txt"
    launcher = tmp_path / "failing_launcher.py"
    launcher.write_text("#!/usr/bin/env python3\n"
                        "import sys\n"
                        f"open({str(captured_manifest_path)!r}, 'w').write(sys.argv[1])\n"
                        "print('launcher stdout before failure')\n"
                        "print('launcher stderr before failure', file=sys.stderr)\n"
                        "sys.exit(17)\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)

    import aot

    with pytest.raises(RuntimeError) as excinfo:
        aot.RPUAOTKernel(artifact, launcher=str(launcher)).run(1, "arg0")

    message = str(excinfo.value)
    assert str(launcher) in message
    assert "exit code 17" in message
    assert "launcher stdout before failure" in message
    assert "launcher stderr before failure" in message
    assert not Path(captured_manifest_path.read_text()).exists()


def test_rpu_backend_uses_json_artifact_and_direct_source_without_op_traversal():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()

    recognizer_body = _source_between(
        compiler_source,
        "    def _make_rpuplan_stage_from_mlir_module_via_recognizer(",
        "    @staticmethod\n    def _make_rpuplan_from_mlir_module_via_recognizer",
    )
    make_rpuexec_body = _source_between(
        compiler_source,
        "    def make_rpuexec(",
        "    @staticmethod\n    def make_rpurc(",
    )
    make_rpurc_body = _source_between(
        compiler_source,
        "    def make_rpurc(",
        "    @staticmethod\n    def _clang_command",
    )
    summary_helper_body = _source_between(
        compiler_source,
        "    def _get_rpuplan_kernel_summary(",
        "    @staticmethod\n    def make_rpuexec(",
    )

    assert 'rpu.get_module_str_attr(mod, "rpu.plan.json")' in recognizer_body
    assert "summary = RPUBackend._get_rpuplan_kernel_summary(mod)" in recognizer_body
    for forbidden in ("RPUKernelPlan.from_json", "RPUKernelPlan.from_dict", "json.loads"):
        assert forbidden not in recognizer_body
    assert "RPUPlanStageBundle(" in recognizer_body
    assert "_rpu_plan_module" not in recognizer_body
    assert "_emit_rpurc_from_plan" not in recognizer_body
    assert "_direct_rpurc_supported_patterns" not in recognizer_body
    assert "str(mod)" not in recognizer_body

    assert "isinstance(src, RPUPlanStageBundle)" in make_rpuexec_body
    assert "summary = {\"kernel_name\": src.kernel_name, \"pattern\": src.pattern}" in make_rpuexec_body
    assert "isinstance(src, RPUExecutableStageBundle)" in make_rpurc_body
    assert "RPURCStageBundle" not in make_rpurc_body
    assert "metadata.pop(\"_rpu_plan_module\"" not in make_rpuexec_body
    assert "metadata.pop(\"_rpu_plan_module\"" not in make_rpurc_body
    assert "_rpu_plan_module" not in make_rpuexec_body
    assert "_rpu_plan_module" not in make_rpurc_body
    for forbidden in ("RPUKernelPlan.from_json", "RPUKernelPlan.from_dict", "json.loads"):
        assert forbidden not in make_rpuexec_body
        assert forbidden not in make_rpurc_body
    assert "rpu._direct_rpurc_supported_patterns()" in make_rpuexec_body
    assert "rpu._emit_rpurc_from_plan(src.module)" in make_rpurc_body
    assert "rpu._get_rpuplan_kernel_summary(mod)" in summary_helper_body
    for forbidden in ("RPUKernelPlan.from_json", "RPUKernelPlan.from_dict", "json.loads"):
        assert forbidden not in summary_helper_body
    assert "_rpu_direct_rpurc_result" not in compiler_source
    broad_access = [
        "walk_" + "rpuplan",
        "parse_" + "rpuplan",
        "get_" + "rpuplan_attr",
        "get_" + "rpuplan_op",
        "get_" + "rpuplan_operand",
        "get_" + "rpuplan_region",
    ]
    assert all(name not in compiler_source for name in broad_access)
    assert "_get_rpuplan_kernel_summary" in compiler_source
    json_replay_helpers = [
        "rpuplan_" + "to_rpurc",
        "replay_" + "rpuplan",
        "emit_rpurc_" + "from_json",
    ]
    assert all(name not in compiler_source for name in json_replay_helpers)


def test_rpu_backend_json_usage_is_artifact_or_diagnostic_only():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()

    recognizer_body = _source_between(
        compiler_source,
        "    def _make_rpuplan_stage_from_mlir_module_via_recognizer(",
        "    @staticmethod\n    def _make_rpuplan_from_mlir_module_via_recognizer",
    )
    make_rpuexec_body = _source_between(
        compiler_source,
        "    def make_rpuexec(",
        "    @staticmethod\n    def make_rpurc(",
    )
    make_rpurc_body = _source_between(
        compiler_source,
        "    def make_rpurc(",
        "    @staticmethod\n    def _clang_command",
    )
    trace_body = _source_between(
        compiler_source,
        "    def _record_optional_plan_trace(",
        "    @staticmethod\n    def _make_rpuplan_from_mlir_module",
    )

    assert 'rpu.get_module_str_attr(mod, "rpu.plan.json")' in recognizer_body
    assert "artifact_text=plan_json" in recognizer_body
    assert "RPUPlanStageBundle(" in recognizer_body
    assert "json.loads" not in recognizer_body
    assert "RPUKernelPlan.from_json" not in recognizer_body
    assert "RPUKernelPlan.from_dict" not in recognizer_body

    assert "isinstance(src, RPUPlanStageBundle)" in make_rpuexec_body
    assert "isinstance(src, RPUExecutableStageBundle)" in make_rpurc_body
    assert "RPURCStageBundle" not in make_rpurc_body
    assert "rpu._emit_rpurc_from_plan(src.module)" in make_rpurc_body
    assert "json.loads" not in make_rpurc_body
    assert "json.loads" not in make_rpuexec_body
    assert "RPUKernelPlan.from_json" not in make_rpurc_body
    assert "RPUKernelPlan.from_json" not in make_rpuexec_body
    assert "RPUKernelPlan.from_dict" not in make_rpurc_body
    assert "RPUKernelPlan.from_dict" not in make_rpuexec_body

    assert compiler_source.count("json.loads(") == 1
    assert "trace = json.loads(trace_json)" in trace_body
    assert "metadata[\"rpu_plan_trace\"] = trace" in trace_body
    assert "json.dumps(manifest, indent=2, sort_keys=True)" in compiler_source
    assert "json.dumps(RPUBackend._repro_clang_invocation(cmd), indent=2, sort_keys=True)" in compiler_source


def test_rpu_backend_removes_transient_stage_metadata_bridge():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()

    assert "_rpu_plan_module" not in compiler_source
    assert "_rpu_direct_rpurc_result" not in compiler_source
    assert "RPUPlanStageBundle" in compiler_source
    assert "metadata[\"_rpu_plan_module\"]" not in compiler_source
    assert "metadata.pop(\"_rpu_plan_module\"" not in compiler_source
    assert "missing_transient_module" not in compiler_source
    assert "missing_rpuexec_bundle" in compiler_source
    transient_keys = sorted(set(re.findall(r'"(_rpu_[^"]*)"', compiler_source)))
    forbidden = [key for key in transient_keys if key.startswith("_rpu_plan_") or "module" in key]
    assert forbidden == []


def test_rpu_backend_has_explicit_rpuexec_stage_between_plan_and_rpurc():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()
    add_stages_body = _source_between(
        compiler_source,
        "    def add_stages(",
        "    def load_dialects(",
    )

    assert 'stages["rpuplan"]' in add_stages_body
    assert 'stages["rpuexec"]' in add_stages_body
    assert 'stages["rpurc"]' in add_stages_body
    assert add_stages_body.index('stages["rpuplan"]') < add_stages_body.index('stages["rpuexec"]')
    assert add_stages_body.index('stages["rpuexec"]') < add_stages_body.index('stages["rpurc"]')


def test_rpu_backend_rpurc_consumes_lowered_stage_bundle():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()
    make_rpuexec_body = _source_between(
        compiler_source,
        "    def make_rpuexec(",
        "    @staticmethod\n    def make_rpurc(",
    )
    make_rpurc_body = _source_between(
        compiler_source,
        "    def make_rpurc(",
        "    @staticmethod\n    def _clang_command",
    )

    assert "class RPUExecutableStageBundle" in compiler_source
    assert "class RPURCStageBundle" not in compiler_source
    assert "isinstance(src, RPUPlanStageBundle)" in make_rpuexec_body
    assert "RPUBackend._make_rpuexec_stage_bundle_from_module" in make_rpuexec_body
    assert "exec_module" in make_rpuexec_body
    assert "RPUExecutableStageBundle(" in compiler_source
    assert "isinstance(src, RPUExecutableStageBundle)" in make_rpurc_body
    assert "_emit_rpurc_from_plan_via_executable" not in make_rpurc_body
    assert "RPUBackend._emit_rpurc_from_executable_via_pass(src.module)" in make_rpurc_body
    assert "rpu._emit_rpurc_from_plan(src.module)" in make_rpurc_body


def test_rpu_backend_make_rpurc_uses_executable_rpurc_pass_hook():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()
    helper_body = _source_between(
        compiler_source,
        "    def _emit_rpurc_from_executable_via_pass(",
        "    @staticmethod\n    def make_rpurc(",
    )
    make_rpurc_body = _source_between(
        compiler_source,
        "    def make_rpurc(",
        "    @staticmethod\n    def _clang_command",
    )

    assert "context = RPUBackend._get_module_context(module)" in helper_body
    assert "pm = ir.pass_manager(context)" in helper_body
    assert "rpu.passes.add_emit_executable_to_rpurc(pm)" in helper_body
    assert "pm.run(module," in helper_body
    assert 'rpu.get_module_str_attr(module, "rpu.rpurc.kernel_name")' in helper_body
    assert 'rpu.get_module_str_attr(module, "rpu.rpurc.source_kind")' in helper_body
    assert 'rpu.get_module_str_attr(module, "rpu.rpurc.source")' in helper_body
    assert "RPUBackend._emit_rpurc_from_executable_via_pass(src.module)" in make_rpurc_body
    assert "rpu._emit_rpurc_from_executable(src.module)" not in make_rpurc_body
    assert "rpu._emit_rpurc_from_plan(src.module)" in make_rpurc_body


def test_rpu_backend_executable_rpurc_route_ignores_direct_helper(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    module = _FakeMLIRModule(_add_plan_json())
    module.has_rpuexec_kernel = True
    bundle = compiler.RPUExecutableStageBundle(
        module=module,
        kernel_name="rpu_add_kernel",
        pattern="add",
        source_kind="rpu_executable",
        fallback_reason=None,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: (_ for _ in ()).throw(AssertionError("direct executable helper must not run")),
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend.make_rpurc(bundle, metadata)

    assert "__rprog__ void rpu_add_kernel()" in result
    assert module.rpurc_ran_pm.added == ["emit_executable_to_rpurc"]
    assert metadata["rpu_rpurc_source_kind"] == "rpu_executable"
    assert metadata["rpu_pipeline_events"][-1] == {
        "stage": "rpurc_emit",
        "event": "rpu_executable",
        "detail": {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
            "source": "rpu.executable",
        },
    }


def test_rpu_backend_executable_rpurc_pass_uses_remembered_replay_context(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ReplayModuleWithoutContext:
        __slots__ = (
            "attrs",
            "has_rpuexec_kernel",
            "rpuexec_kernel_name",
            "rpuexec_pattern",
            "rpurc_ran_pm",
        )

        def __init__(self):
            self.attrs = {}
            self.has_rpuexec_kernel = True
            self.rpuexec_kernel_name = "rpu_add_kernel"
            self.rpuexec_pattern = "add"
            self.rpurc_ran_pm = None

    module = ReplayModuleWithoutContext()
    context = object()

    compiler.RPUBackend._remember_module_context(module, context)
    result = compiler.RPUBackend._emit_rpurc_from_executable_via_pass(module)

    assert result["kernel_name"] == "rpu_add_kernel"
    assert result["source_kind"] == "rpu_executable"
    assert "__rprog__ void rpu_add_kernel()" in result["source"]
    assert module.rpurc_ran_pm.context is context
    assert not hasattr(module, "context")


def test_rpu_backend_make_rpuexec_returns_executable_bundle(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ExecutableModule:

        def __str__(self):
            return 'module { rpu.kernel @rpu_add_kernel() { rpu.return } {kind = "add"} }'

    plan_module = object()
    executable_module = ExecutableModule()
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}

    monkeypatch.setattr(compiler.rpu, "_direct_rpurc_supported_patterns", lambda: ("add", ))
    monkeypatch.setattr(compiler.rpu, "_rpuplan_supports_executable_lowering", lambda module: module is plan_module,
                        raising=False)
    monkeypatch.setattr(compiler.rpu, "_lower_rpuplan_to_executable_module", lambda module: executable_module,
                        raising=False)
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(compiler.rpu, "_emit_rpurc_from_executable", lambda module: {}, raising=False)

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is executable_module
    assert result.kernel_name == "rpu_add_kernel"
    assert result.pattern == "add"
    assert result.source_kind == "rpu_executable"
    assert result.fallback_reason is None
    assert "rpu.kernel" in str(result)
    assert 'kind = "add"' in str(result)
    assert metadata["rpu_rpuexec_source_kind"] == "rpu_executable"


def test_rpu_backend_make_rpuplan_records_direct_executable_candidate(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is source_module
    assert metadata["rpu_direct_executable_candidate"] is True


def test_rpu_backend_make_rpuplan_prefers_supported_direct_executable_binding(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_ttir_add_gemm_to_executable_module",
        lambda module: (_ for _ in ()).throw(AssertionError("legacy alias used")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is source_module
    event = [
        event for event in metadata["rpu_pipeline_events"]
        if event["stage"] == "direct_executable_lower" and event["event"] == "candidate"
    ][0]
    assert event["detail"]["patterns"] == [
        "add",
        "gemm",
        "softmax",
        "convkxk",
        "resnet_block",
        "resnet50_bottleneck",
    ]


def test_rpu_backend_direct_candidate_patterns_use_cpp_kind_contract(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())

    monkeypatch.setattr(
        compiler.rpu,
        "_supported_executable_kernel_kinds",
        lambda: ("softmax", "gemm"),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is source_module
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    event = [
        event for event in metadata["rpu_pipeline_events"]
        if event["stage"] == "direct_executable_lower" and event["event"] == "candidate"
    ][0]
    assert event["detail"]["patterns"] == ["softmax", "gemm"]


def test_rpu_backend_make_rpuplan_has_no_legacy_add_gemm_direct_alias():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    body = _source_between(
        source,
        "    def _make_rpuplan_from_mlir_module(",
        "    @staticmethod\n    def _get_optional_module_str_attr",
    )

    assert "_lower_ttir_add_gemm_to_executable_module" not in body


def test_rpu_backend_make_rpuplan_returns_executable_bundle_for_direct_supported(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is source_module
    assert result.source_kind == "rpu_executable"
    assert result.fallback_reason is None
    assert str(result) == str(source_module)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert source_module.has_rpuplan_kernel is False
    event = [
        event for event in metadata["rpu_pipeline_events"]
        if event["stage"] == "direct_executable_lower" and event["event"] == "end"
    ][0]
    assert event["detail"]["artifact_source"] == "rpu_exec_mlir"


def test_rpu_backend_make_rpuexec_passes_through_executable_bundle(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ExecutableModule:

        def __str__(self):
            return 'module { rpu.kernel @rpu_add_kernel() { rpu.return } {kind = "add"} }'

    executable_module = ExecutableModule()
    bundle = compiler.RPUExecutableStageBundle(
        module=executable_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
        source_kind="rpu_executable",
        fallback_reason=None,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: (_ for _ in ()).throw(AssertionError("plan-owned lowering should not run")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_direct_rpurc_supported_patterns",
        lambda: (_ for _ in ()).throw(AssertionError("direct fallback should not run")),
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert result is bundle
    assert str(result) == str(executable_module)
    event = metadata["rpu_pipeline_events"][-1]
    assert event["stage"] == "rpuexec_lower"
    assert event["event"] == "rpu_executable"
    assert event["detail"]["source_boundary"] == "executable_stage_value"


def test_rpu_backend_plan_bundle_has_no_direct_executable_payload(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    assert "direct_executable_module" not in compiler.RPUPlanStageBundle.__dataclass_fields__
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    assert "direct_executable_module: object | None = None" not in source
    assert "direct_executable_module=direct_executable_module" not in source
    assert 'getattr(src, "direct_executable_module", None)' not in source


def test_rpu_backend_supported_direct_stage_chain_is_executable_trunk(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    calls = []

    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: (_ for _ in ()).throw(AssertionError("plan-to-executable lowering must not run")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_plan",
        lambda module: (_ for _ in ()).throw(AssertionError("plan rc emission must not run")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: (_ for _ in ()).throw(AssertionError("direct executable helper must not run")),
        raising=False,
    )

    metadata = {}
    plan_result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)
    exec_result = compiler.RPUBackend.make_rpuexec(plan_result, metadata)
    rc_result = compiler.RPUBackend.make_rpurc(exec_result, metadata)

    assert isinstance(plan_result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert plan_result.module is source_module
    assert str(plan_result) == str(source_module)
    assert exec_result is plan_result
    assert str(exec_result) == str(source_module)
    assert "__rprog__ void rpu_add_kernel()" in rc_result
    assert source_module.rpurc_ran_pm.added == ["emit_executable_to_rpurc"]
    assert calls == []


def test_rpu_backend_add_direct_artifact_skips_rpuplan_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm: (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run for executable Add sidecar")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata
    assert any(event["stage"] == "direct_executable_lower" and event["event"] == "artifact"
               and event["detail"]["status"] == "available" and event["detail"]["artifact_kind"] == "rpu_exec_mlir"
               for event in metadata["rpu_pipeline_events"])


def test_rpu_backend_direct_artifact_branch_has_no_fallback_to_rpuplan(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm: (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not recover direct sidecar failure")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert all(event["stage"] != "rpuplan_recognize" for event in metadata["rpu_pipeline_events"])


def test_rpu_backend_direct_artifact_source_has_no_fallback_to_rpuplan():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    direct_helper_body = _source_between(
        source,
        "    def _try_make_direct_executable_stage_from_ttir(",
        "    @staticmethod\n    def _make_rpuplan_stage_from_mlir_module_via_recognizer",
    )

    assert "fallback_to_rpuplan" not in direct_helper_body
    assert "rpu.passes.add_recognize_plan" not in direct_helper_body
    assert "rpu.passes.add_lower_supported_ttir_to_executable(pm)" in direct_helper_body
    assert "pm.run(mod," in direct_helper_body
    assert "RPU direct executable sidecar generation failed" not in direct_helper_body
    assert "RPUBackend._serialize_direct_executable_plan_sidecar" not in direct_helper_body
    assert "_serialize_rpuexec_plan_sidecar" not in direct_helper_body
    assert '"rpuplan_recognize"' not in direct_helper_body


def test_rpu_backend_direct_executable_events_do_not_impersonate_rpuplan_recognizer():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    dispatcher_body = _source_between(
        source,
        "    def _make_post_ttir_stage_from_mlir_module(",
        "    @staticmethod\n    def _make_rpuplan_from_mlir_module",
    )
    direct_helper_body = _source_between(
        source,
        "    def _try_make_direct_executable_stage_from_ttir(",
        "    @staticmethod\n    def _make_rpuplan_stage_from_mlir_module_via_recognizer",
    )
    recognizer_body = _source_between(
        source,
        "    def _make_rpuplan_stage_from_mlir_module_via_recognizer(",
        "    @staticmethod\n    def _make_rpuplan_from_mlir_module_via_recognizer",
    )

    assert "_record_pipeline_event" not in dispatcher_body
    assert '"rpuplan_recognize"' not in direct_helper_body
    assert ('RPUBackend._record_pipeline_event(\n'
            '            metadata,\n'
            '            "rpuplan_recognize",\n'
            '            "start",') in recognizer_body
    assert "rpu.passes.add_recognize_plan(pm)" in recognizer_body


def test_rpu_backend_make_rpuplan_dispatcher_has_no_inline_direct_or_pass_mechanics():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    start = source.index("    def _make_post_ttir_stage_from_mlir_module(")
    end = source.index("\n    @staticmethod\n    def ", start + 1)
    dispatcher_body = source[start:end]

    assert "RPUBackend._try_make_direct_executable_stage_from_ttir" in dispatcher_body
    assert "RPUBackend._make_rpuplan_stage_from_mlir_module_via_recognizer" in dispatcher_body
    for forbidden in [
            "direct_executable_module",
            "direct_executable_plan_json",
            "_lower_ttir_supported_to_executable_module",
            "_serialize_direct_executable_plan_sidecar",
            "ir.pass_manager",
            "add_recognize_plan",
            "pm.run",
    ]:
        assert forbidden not in dispatcher_body


def test_rpu_backend_make_rpuplan_uses_post_ttir_dispatcher_name():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    make_rpuplan_body = _source_between(
        source,
        "    def make_rpuplan(",
        "    @staticmethod\n    def _record_optional_plan_trace",
    )

    assert "return RPUBackend.make_post_ttir(src, metadata)" in make_rpuplan_body
    assert "RPUBackend._make_post_ttir_stage_from_mlir_module(src, metadata)" not in make_rpuplan_body
    assert "RPUBackend._make_rpuplan_from_mlir_module(src, metadata)" not in make_rpuplan_body


def test_rpu_backend_rpuplan_stage_uses_post_ttir_entrypoint():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    add_stages_body = _source_between(
        source,
        "    def add_stages(",
        "    def load_dialects(",
    )
    make_post_ttir_body = _source_between(
        source,
        "    def make_post_ttir(",
        "    @staticmethod\n    def make_rpuplan",
    )
    make_rpuplan_body = _source_between(
        source,
        "    def make_rpuplan(",
        "    @staticmethod\n    def _record_optional_plan_trace",
    )

    assert 'stages["rpuplan"] = lambda src, metadata: self.make_post_ttir(src, metadata)' in add_stages_body
    assert 'stages["rpuplan"] = lambda src, metadata: self.make_rpuplan(src, metadata)' not in add_stages_body
    assert "def make_post_ttir(src, metadata):" in make_post_ttir_body
    assert "RPUBackend._make_post_ttir_stage_from_mlir_module(src, metadata)" in make_post_ttir_body
    assert "return RPUBackend.make_post_ttir(src, metadata)" in make_rpuplan_body
    assert "RPUBackend._make_post_ttir_stage_from_mlir_module(src, metadata)" not in make_rpuplan_body
    assert "RPUBackend._make_rpuplan_from_mlir_module(src, metadata)" not in make_rpuplan_body


def test_rpu_backend_post_ttir_dispatcher_is_executable_aware_not_rpuplan_named():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    dispatcher_body = _source_between(
        source,
        "    def _make_post_ttir_stage_from_mlir_module(",
        "    @staticmethod\n    def _make_rpuplan_from_mlir_module",
    )
    legacy_start = source.index("    def _make_rpuplan_from_mlir_module(")
    legacy_end = source.index("\n    @staticmethod\n    def ", legacy_start + 1)
    legacy_wrapper_body = source[legacy_start:legacy_end]

    assert "RPUBackend._try_make_direct_executable_stage_from_ttir(mod, metadata)" in dispatcher_body
    assert "RPUBackend._make_rpuplan_stage_from_mlir_module_via_recognizer(mod, metadata)" in dispatcher_body
    assert "return RPUBackend._make_post_ttir_stage_from_mlir_module(mod, metadata)" in legacy_wrapper_body
    for forbidden in [
            "rpu.passes.add_recognize_plan",
            "rpu.get_module_str_attr",
            "RPUPlanStageBundle(",
            "RPUExecutableStageBundle(",
            "_serialize_direct_executable_plan_sidecar",
    ]:
        assert forbidden not in dispatcher_body
        assert forbidden not in legacy_wrapper_body


def test_rpu_backend_direct_executable_artifact_is_native_exec_mlir(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert str(result) == str(source_module)
    assert "rpuplan_artifact_source_kind" not in metadata
    direct_events = [event for event in metadata["rpu_pipeline_events"] if event["stage"] == "direct_executable_lower"]
    assert [event["event"] for event in direct_events] == [
        "start",
        "candidate",
        "artifact",
        "end",
    ]
    assert direct_events[2]["detail"]["source_boundary"] == "rpu_executable"
    assert direct_events[2]["detail"]["artifact_kind"] == "rpu_exec_mlir"
    assert direct_events[3]["detail"]["artifact_source"] == "rpu_exec_mlir"


def test_rpu_backend_missing_sidecar_serializer_does_not_block_direct_artifact(monkeypatch, ):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert str(result) == str(source_module)
    assert result.source_kind == "rpu_executable"


def test_rpu_backend_direct_stage_result_uses_summary_factory(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    summary_calls = []

    def summarize(module):
        summary_calls.append(module)
        return {"kernel_name": "rpu_custom_kernel", "pattern": "gemm"}

    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        summarize,
        raising=False,
    )

    result = compiler.RPUBackend._try_make_direct_executable_stage_from_ttir(
        source_module,
        {},
    )

    assert summary_calls == [source_module]
    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is source_module
    assert result.kernel_name == "rpu_custom_kernel"
    assert result.pattern == "gemm"
    assert result.source_kind == "rpu_executable"
    assert str(result) == str(source_module)


def test_rpu_backend_direct_pass_pipeline_source_uses_python_pass_manager():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    direct_pass_body = _source_between(
        source,
        "    def _try_make_direct_executable_stage_from_ttir(",
        "    @staticmethod\n    def _make_rpuplan_stage_from_mlir_module_via_recognizer",
    )

    assert "pm = ir.pass_manager(mod.context)" in direct_pass_body
    assert "rpu.passes.add_lower_supported_ttir_to_executable(pm)" in direct_pass_body
    assert "pm.run(mod," in direct_pass_body
    assert "_lower_ttir_supported_to_executable_module" not in direct_pass_body
    assert "direct_executable_helper" not in direct_pass_body


def test_rpu_backend_direct_pass_pipeline_source_has_no_sidecar_serializer():
    source = (_repo_root() / "third_party" / "rpu" / "backend" / "compiler.py").read_text()
    direct_pass_body = _source_between(
        source,
        "    def _try_make_direct_executable_stage_from_ttir(",
        "    @staticmethod\n    def _make_rpuplan_stage_from_mlir_module_via_recognizer",
    )

    for forbidden in [
            "_serialize_direct_executable_plan_sidecar",
            "_serialize_rpuexec_plan_sidecar",
            "direct_executable_plan_json",
            "rpuplan_artifact_source_kind",
            "rpu_executable_sidecar",
            '"sidecar"',
    ]:
        assert forbidden not in direct_pass_body
    assert "direct_executable_module = mod" in direct_pass_body
    assert "return direct_bundle" in direct_pass_body
    assert "RPUBackend._record_rpuexec_mlir_artifact(direct_executable_module, metadata)" in direct_pass_body
    assert '"artifact_kind": "rpu_exec_mlir"' in direct_pass_body


def test_rpu_backend_executable_first_pass_pipeline_skips_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm: (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run from direct pass pipeline")),
    )
    metadata = {}

    result = compiler.RPUBackend._try_make_direct_executable_stage_from_ttir(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata


def test_rpu_backend_gemm_direct_artifact_skips_rpuplan_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_gemm_kernel", "pattern": "gemm"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm:
        (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run for executable GEMM sidecar")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert result.pattern == "gemm"
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata


def test_rpu_backend_softmax_direct_artifact_skips_rpuplan_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_add_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_softmax_kernel", "pattern": "softmax"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm:
        (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run for executable Softmax sidecar")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert result.pattern == "softmax"
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata


def test_rpu_backend_convkxk_direct_artifact_skips_rpuplan_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_convkxk_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_convkxk_kernel", "pattern": "convkxk"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm:
        (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run for executable ConvKxK sidecar")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert result.pattern == "convkxk"
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata


def test_rpu_backend_residual_direct_artifact_skips_rpuplan_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_residual_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_resnet_block_kernel",
            "pattern": "resnet_block",
        },
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm:
        (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run for executable residual sidecar")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert result.pattern == "resnet_block"
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata


def test_rpu_backend_bottleneck_direct_artifact_skips_rpuplan_recognizer(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    source_module = _FakeMLIRModule(_bottleneck_plan_json())
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_resnet50_bottleneck_kernel",
            "pattern": "resnet50_bottleneck",
        },
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu.passes,
        "add_recognize_plan",
        lambda pm:
        (_ for _ in ()).throw(AssertionError("rpu_plan recognizer must not run for executable bottleneck sidecar")),
    )
    metadata = {}

    result = compiler.RPUBackend._make_rpuplan_from_mlir_module(source_module, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert source_module.ran_pm.added == ["lower_supported_ttir_to_executable"]
    assert result.module is source_module
    assert result.pattern == "resnet50_bottleneck"
    assert str(result) == str(source_module)
    assert source_module.has_rpuplan_kernel is False
    assert "rpuplan_artifact_source_kind" not in metadata


def test_rpu_backend_make_rpuexec_prefers_direct_candidate(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ExecutableModule:

        def __str__(self):
            return 'module { rpu.kernel @rpu_add_kernel() { rpu.return } {kind = "add"} }'

    direct_executable_module = ExecutableModule()
    bundle = compiler.RPUExecutableStageBundle(
        module=direct_executable_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
        source_kind="rpu_executable",
        fallback_reason=None,
    )
    metadata = {}

    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: (_ for _ in ()).throw(AssertionError("plan support query should not run")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: (_ for _ in ()).throw(AssertionError("plan-owned lowering should not run")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "rpu_add_kernel",
            "pattern": "add",
        },
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert result.module is direct_executable_module
    assert result.source_kind == "rpu_executable"
    assert result.fallback_reason is None
    assert metadata["rpu_rpuexec_source_kind"] == "rpu_executable"
    event = metadata["rpu_pipeline_events"][-1]
    assert event["stage"] == "rpuexec_lower"
    assert event["event"] == "rpu_executable"
    assert event["detail"]["source_boundary"] == "executable_stage_value"


def test_rpu_backend_executable_success_does_not_require_direct_hook(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    class ExecutableModule:

        def __str__(self):
            return 'module { rpu.kernel @rpu_add_kernel() { rpu.return } {kind = "add"} }'

    plan_module = object()
    executable_module = ExecutableModule()
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}

    monkeypatch.delattr(compiler.rpu, "_direct_rpurc_supported_patterns", raising=False)
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: module is plan_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: executable_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert result.module is executable_module
    assert result.source_kind == "rpu_executable"
    assert result.fallback_reason is None
    assert metadata["rpu_rpuexec_source_kind"] == "rpu_executable"
    assert "rpu_direct_rpurc_patterns" not in metadata


def test_rpu_backend_make_rpuexec_uses_executable_summary(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    plan_module = object()
    executable_module = object()
    metadata = {}
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="plan_kernel",
        pattern="softmax",
    )

    monkeypatch.setattr(
        compiler.rpu,
        "_direct_rpurc_supported_patterns",
        lambda: ("softmax", ),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: module is plan_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: executable_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {
            "kernel_name": "exec_kernel",
            "pattern": "softmax",
        },
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert result.module is executable_module
    assert result.kernel_name == "exec_kernel"
    assert result.pattern == "softmax"
    assert metadata["name"] == "exec_kernel"
    assert metadata["rpu_pattern"] == "softmax"
    event = metadata["rpu_pipeline_events"][-1]
    assert event["stage"] == "rpuexec_lower"
    assert event["event"] == "rpu_executable"
    assert event["detail"]["kernel_name"] == "exec_kernel"


def test_rpu_backend_missing_summary_hook_falls_back(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    plan_module = object()
    executable_module = object()
    metadata = {}
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )

    monkeypatch.setattr(compiler.rpu, "_direct_rpurc_supported_patterns", lambda: ("add", ))
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: module is plan_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: executable_module,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )
    if hasattr(compiler.rpu, "_get_rpuexec_kernel_summary"):
        monkeypatch.delattr(compiler.rpu, "_get_rpuexec_kernel_summary", raising=False)

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert result.module is plan_module
    assert result.source_kind == "rpu_plan_kernel"
    assert result.fallback_reason == "missing_executable_hooks"
    assert "reason: missing_executable_hooks" in str(result)
    assert metadata["rpu_rpuexec_source_kind"] == "rpu_plan_kernel"
    event = metadata["rpu_pipeline_events"][-1]
    assert event["stage"] == "rpuexec_lower"
    assert event["event"] == "rpu_plan_kernel"
    assert event["detail"]["reason"] == "missing_executable_hooks"


def test_rpu_backend_python_executable_pattern_contract_uses_ir_hook(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    plan_module = object()
    metadata = {}
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )

    monkeypatch.setattr(compiler.rpu, "_direct_rpurc_supported_patterns", lambda: ("add", ))
    monkeypatch.setattr(
        compiler.rpu,
        "_supported_executable_kernel_kinds",
        lambda: ("gemm", ),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: True,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: (_ for _ in ()).throw(AssertionError("should not lower")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )

    with pytest.raises(
            RuntimeError,
            match="does not allow direct fallback for reason unsupported_executable_pattern",
    ):
        compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert metadata["rpu_executable_patterns"] == ["gemm"]
    assert "rpu_rpuexec_source_kind" not in metadata
    event = metadata["rpu_pipeline_events"][-1]
    assert event["stage"] == "rpuexec_lower"
    assert event["event"] == "fail"
    assert event["detail"]["fallback_reason"] == "unsupported_executable_pattern"
    assert event["detail"]["supported_executable_patterns"] == ["gemm"]


def test_rpu_backend_make_rpuexec_direct_opt_out_returns_plan_bundle(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    plan_module = object()
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}
    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.setattr(compiler.rpu, "_direct_rpurc_supported_patterns", lambda: ("add", ))

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.module is plan_module
    assert result.source_kind == "rpu_plan_kernel"
    assert result.fallback_reason == "env_opt_out"
    assert "source_kind: rpu_plan_kernel" in str(result)
    assert "reason: env_opt_out" in str(result)
    assert metadata["rpu_rpuexec_source_kind"] == "rpu_plan_kernel"


def test_rpu_backend_direct_fallback_requires_direct_hook(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    plan_module = object()
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}

    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    monkeypatch.delattr(compiler.rpu, "_direct_rpurc_supported_patterns", raising=False)

    with pytest.raises(
            RuntimeError,
            match="direct fallback requires _direct_rpurc_supported_patterns",
    ):
        compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert metadata["rpu_pipeline_events"][-1]["stage"] == "rpuexec_lower"
    assert metadata["rpu_pipeline_events"][-1]["event"] == "fail"
    assert (metadata["rpu_pipeline_events"][-1]["detail"]["fallback_reason"] == "env_opt_out")


def test_rpu_backend_builder_fallback_detail_in_make_rpuexec(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    plan_module = object()
    bundle = compiler.RPUPlanStageBundle(
        module=plan_module,
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}
    detail = ("add requires unmasked contiguous linear f16 vector attrs with out=0 "
              "lhs=1 rhs=2, logical_n=n, positive 16-aligned n<=128")
    monkeypatch.setattr(compiler.rpu, "_direct_rpurc_supported_patterns", lambda: ("add", ))
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_supports_executable_lowering",
        lambda module: False,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_rpuplan_executable_lowering_failure_reason",
        lambda module: detail,
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_lower_rpuplan_to_executable_module",
        lambda module: (_ for _ in ()).throw(AssertionError("should not lower")),
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_emit_rpurc_from_executable",
        lambda module: {},
        raising=False,
    )
    monkeypatch.setattr(
        compiler.rpu,
        "_get_rpuexec_kernel_summary",
        lambda module: {"kernel_name": "rpu_add_kernel", "pattern": "add"},
        raising=False,
    )

    result = compiler.RPUBackend.make_rpuexec(bundle, metadata)

    assert isinstance(result, compiler.RPUExecutableStageBundle)
    assert result.source_kind == "rpu_plan_kernel"
    assert result.fallback_reason == "unsupported_executable_shape"
    assert "reason: unsupported_executable_shape" in str(result)
    assert f"detail: {detail}" in str(result)
    event = metadata["rpu_pipeline_events"][-1]
    assert event["stage"] == "rpuexec_lower"
    assert event["event"] == "rpu_plan_kernel"
    assert event["detail"]["reason"] == "unsupported_executable_shape"
    assert event["detail"]["executable_builder_failure"] == detail


def test_rpu_backend_make_rpurc_rejects_rpuplan_bundle(monkeypatch):
    _install_triton_backend_stubs(monkeypatch)
    import compiler

    bundle = compiler.RPUPlanStageBundle(
        module=object(),
        kernel_name="rpu_add_kernel",
        pattern="add",
    )
    metadata = {}

    with pytest.raises(RuntimeError, match="requires RPUExecutableStageBundle"):
        compiler.RPUBackend.make_rpurc(bundle, metadata)


def test_rpu_backend_has_no_python_json_rpurc_emitters():
    compiler_source = (_rpu_backend_path() / "compiler.py").read_text()

    assert "_ADD_MAX_CONTIG_NVEC" not in compiler_source
    assert "_ADD_MAX_TENSOR_ROWS" not in compiler_source
    assert "_format_signature_params" not in compiler_source
    assert "_ptr_expr" not in compiler_source
    assert "_plan_emitters" not in compiler_source
    assert "_emit_add_chunk" not in compiler_source
    assert not re.search(
        r"def _emit_(add|large_add|masked_add|gemm|softmax|convkxk|resnet|resnet50).*_body",
        compiler_source,
    )
    assert "json_compat" not in compiler_source


def test_rpu_direct_op_consumer_does_not_expose_broad_python_binding():
    rpu_binding = (_repo_root() / "third_party" / "rpu" / "triton_rpu.cc").read_text()

    def binding_names():
        return re.findall(r'm\.def\("([^"]+)"', rpu_binding)

    assert 'm.def("_emit_rpurc_from_plan"' in rpu_binding
    assert 'm.def("_direct_rpurc_supported_patterns"' in rpu_binding
    assert 'm.def("_get_rpuplan_kernel_summary"' in rpu_binding
    assert "get_module_str_attr" in binding_names()
    assert "_emit_rpurc_from_plan" in binding_names()
    assert "_direct_rpurc_supported_patterns" in binding_names()
    assert "_get_rpuplan_kernel_summary" in binding_names()
    required_bindings = {
        "get_module_str_attr",
        "_emit_rpurc_from_plan",
        "_direct_rpurc_supported_patterns",
        "_get_rpuplan_kernel_summary",
        "load_dialects",
    }
    exported_bindings = set(binding_names())
    assert required_bindings <= exported_bindings
    suspicious_extra_bindings = [
        name for name in sorted(exported_bindings - required_bindings)
        if (name.startswith(("_get_rpuplan", "get_rpuplan", "walk_",
                             "parse_")) or name.startswith(("rpuplan_", "emit_rpurc_")) or (any(
                                 token in name
                                 for token in ("attr", "op", "operand", "region")) and name != "get_module_str_attr"))
    ]
    assert suspicious_extra_bindings == []
    forbidden_names = [
        "walk_" + "rpuplan",
        "parse_" + "rpuplan",
        "get_" + "rpuplan_attr",
        "get_" + "rpuplan_op",
        "emit_rpurc_" + "from_json",
    ]
    assert all(name not in rpu_binding for name in forbidden_names)


def test_rpuplan_cpp_json_export_is_not_direct_emitter_input():
    transforms = _repo_root() / "third_party" / "rpu" / "lib" / "RPUTransforms"
    recognizer = (transforms / "RecognizeRPUPlan.cpp").read_text()
    json_export = (transforms / "RPUPlanJSON.cpp").read_text()
    direct_emitter = (transforms / "RPUDSLEmitter.cpp").read_text()

    assert "serializeRPUPlanKernelOpToJson(*op)" in recognizer
    assert 'module->setAttr("rpu.plan.json"' in recognizer
    assert "static std::string serializeRPUPlanToJson" in json_export
    assert "serializeRPUPlanKernelOpToJson" in json_export

    assert "serializeRPUPlanKernelOpToJson" not in direct_emitter
    assert "serializeRPUPlanToJson" not in direct_emitter
    assert "rpu.plan.json" not in direct_emitter


def _add_ttir():
    return """
tt.func public @rpu_add_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
  %2 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
  %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
  %4 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
  %5 = tt.addptr %2, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
  %6 = tt.addptr %3, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
  %7 = tt.load %4 : tensor<16x!tt.ptr<f16>>
  %8 = tt.load %5 : tensor<16x!tt.ptr<f16>>
  %9 = arith.addf %7, %8 : tensor<16xf16>
  tt.store %6, %9 : tensor<16x!tt.ptr<f16>>
  tt.return
}
"""


def _unsupported_ttir():
    return """
tt.func public @rpu_unsupported_kernel(%arg0: !tt.ptr<f16>) {
  tt.return
}
"""
