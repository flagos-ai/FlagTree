import importlib
import json
import os
from pathlib import Path

import pytest

from _rpu_env import make_ast_source, require_rpu_toolchain


def test_rpu_compile_softmax16_kernel_produces_binary(tmp_path, monkeypatch):
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=16)

    assert "ttir" in kernel.asm
    assert "rpuplan" in kernel.asm
    assert "rpurc" in kernel.asm
    assert "rpuinstr" in kernel.asm
    assert "rpubin" in kernel.asm
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "softmax")
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    plan = _recognized_plan_from_kernel(kernel, tmp_path, monkeypatch)
    assert plan["pattern"] == "softmax"
    assert plan["shape"] == {"n": 16}
    assert plan["args"] == {"input": 1, "out": 0}
    assert plan["layout"] == {"access": "linear", "memory": "contiguous_vector"}
    assert "ctx.reduce_sum_all" in plan["required_dsl_features"]
    assert len(kernel.asm["rpubin"]) > 0
    assert "ctx.reduce_max_all" in kernel.asm["rpurc"]
    assert "rpu::exp" in kernel.asm["rpurc"]
    assert "ctx.reduce_sum_all" in kernel.asm["rpurc"]
    assert "ctx.store_contig<1>" in kernel.asm["rpurc"]


def test_rpu_compile_softmax16_kernel_uses_mlir_plan_pass(tmp_path, monkeypatch):
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=16)

    plan = _recognized_plan_from_kernel(kernel, tmp_path, monkeypatch)
    assert plan == {
        "args": {"input": 1, "out": 0},
        "emission": {"input": 1, "kind": "softmax", "n": 16, "out": 0},
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
        "shape": {"n": 16},
        "signature": {
            "params": [
                {"element_type": "f16", "index": 0, "kind": "ptr", "name": "arg0"},
                {"element_type": "f16", "index": 1, "kind": "ptr", "name": "arg1"},
            ],
            "return_type":
            "void",
        },
        "version":
        1,
    }


def test_rpu_mlir_plan_pass_emits_rpuplan_kernel_for_softmax(tmp_path, monkeypatch):
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=16)

    plan_json, printed = _run_rpu_plan_pass_on_mlir(kernel.asm["ttir"], tmp_path, monkeypatch)
    plan = json.loads(plan_json)

    _assert_native_exec_artifact(kernel.asm["rpuplan"], "softmax")
    assert "rpu_plan.kernel @rpu_softmax_kernel_plan" in printed
    assert 'kernel_name = "rpu_softmax_kernel"' in printed
    assert 'pattern = "softmax"' in printed
    assert "source_func = @rpu_softmax_kernel" in printed
    assert "rpu_plan.load" not in printed
    assert "rpu_plan.store" not in printed
    assert "rpu_plan.tile" not in printed
    assert "rpu_plan.async" not in printed


def test_rpu_compile_softmax16_defaults_to_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "softmax")
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "softmax"' in kernel.asm["rpuexec"]
    assert "_rpu_direct_rpurc_result" not in kernel.metadata._fields


@pytest.mark.parametrize("n_elements,nvec", [(16, 1), (64, 4)])
def test_rpu_compile_softmax16_direct_opt_out_uses_plan_kernel(tmp_path, monkeypatch, n_elements, nvec):
    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=n_elements)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_plan_kernel"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_plan_kernel"
    assert "source_kind: rpu_plan_kernel" in kernel.asm["rpuexec"]
    assert "reason: env_opt_out" in kernel.asm["rpuexec"]
    assert f"ctx.load_contig<{nvec}>" in kernel.asm["rpurc"]
    assert "ctx.reduce_sum_all" in kernel.asm["rpurc"]


def test_rpu_compile_softmax16_executable_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "1")
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert "auto tile0 = ctx.load_contig<1>(arg1);" in kernel.asm["rpurc"]
    assert "auto tile3 = tile2 * scalar2;" in kernel.asm["rpurc"]
    assert "ctx.store_contig<1>(arg0, tile3);" in kernel.asm["rpurc"]
    assert "auto y = e * inv_s;" not in kernel.asm["rpurc"]


def test_rpu_compile_softmax32_defaults_to_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=32)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "softmax"' in kernel.asm["rpuexec"]
    assert "n = 2 : i32" in kernel.asm["rpuexec"]
    assert "auto tile0 = ctx.load_contig<2>(arg1);" in kernel.asm["rpurc"]
    assert "ctx.store_contig<2>" in kernel.asm["rpurc"]


def test_rpu_compile_softmax64_defaults_to_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements=64)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "softmax"' in kernel.asm["rpuexec"]
    assert "n = 4 : i32" in kernel.asm["rpuexec"]
    assert "auto tile0 = ctx.load_contig<4>(arg1);" in kernel.asm["rpurc"]
    assert "ctx.store_contig<4>" in kernel.asm["rpurc"]


def _compile_rpu_softmax_kernel(tmp_path, monkeypatch, n_elements):
    llvm_root = require_rpu_toolchain()

    monkeypatch.setenv("RPU_LLVM_ROOT", str(llvm_root))
    monkeypatch.delenv("RPU_BOARD_LOAD_CONTIG_NVEC", raising=False)
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    monkeypatch.setitem(globals(), "tl", tl)

    @triton.jit
    def rpu_softmax_kernel(out, x, n: tl.constexpr):
        offsets = tl.arange(0, n)
        values = tl.load(x + offsets)
        shifted = values - tl.max(values, axis=0)
        numerator = tl.exp(shifted)
        denominator = tl.sum(numerator, axis=0)
        tl.store(out + offsets, numerator / denominator)

    src = make_ast_source(
        fn=rpu_softmax_kernel,
        constants={2: n_elements},
        signature={0: "*fp16", 1: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _run_rpu_plan_pass_on_mlir(module_text, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "rpu")

    from triton._C.libtriton import ir, rpu

    context = ir.context()
    ir.load_dialects(context)
    rpu.load_dialects(context)
    module_path = tmp_path / "rpu_softmax_candidate.mlir"
    module_path.write_text(module_text)
    module = ir.parse_mlir_module(str(module_path), context)
    pm = ir.pass_manager(context)
    rpu.passes.add_recognize_plan(pm)
    pm.run(module, "rpu_test_helper")
    printed = str(module)
    roundtrip_path = tmp_path / "rpu_softmax_candidate_roundtrip.mlir"
    roundtrip_path.write_text(printed)
    ir.parse_mlir_module(str(roundtrip_path), context)
    return rpu.get_module_str_attr(module, "rpu.plan.json"), printed


def _assert_native_exec_artifact(artifact, pattern):
    assert "rpu.kernel" in artifact
    assert f'kind = "{pattern}"' in artifact
    assert "rpu_plan.kernel" not in artifact
    assert "RPUPlanStageBundle" not in artifact


def _recognized_plan_from_kernel(kernel, tmp_path, monkeypatch):
    plan_json, _printed = _run_rpu_plan_pass_on_mlir(kernel.asm["ttir"], tmp_path, monkeypatch)
    return json.loads(plan_json)
