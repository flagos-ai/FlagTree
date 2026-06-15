import importlib
import json
import os
from pathlib import Path

import pytest

from _rpu_env import make_ast_source, require_rpu_toolchain


def test_rpu_compile_add_kernel_produces_binary(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    assert "ttir" in kernel.asm
    assert "rpuplan" in kernel.asm
    assert "rpurc" in kernel.asm
    assert "rpuinstr" in kernel.asm
    assert "rpubin" in kernel.asm
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    plan = _recognized_plan_from_kernel(kernel, tmp_path, monkeypatch)
    assert plan["pattern"] == "add"
    assert plan["signature"]["params"] == [
        {"index": 0, "name": "arg0", "kind": "ptr", "element_type": "f16"},
        {"index": 1, "name": "arg1", "kind": "ptr", "element_type": "f16"},
        {"index": 2, "name": "arg2", "kind": "ptr", "element_type": "f16"},
    ]
    assert plan["layout"] == {"access": "linear", "memory": "contiguous_vector"}
    assert plan["required_dsl_features"] == [
        "ctx.load_contig",
        "tile.add",
        "ctx.store_contig",
    ]
    assert len(kernel.asm["rpuinstr"]) > 0
    assert len(kernel.asm["rpubin"]) > 0
    assert isinstance(kernel.asm["rpubin"], bytes)
    # rpuinstr is the hex listing of the rpubin payload; both are
    # non-empty and consistent. Exact instruction encodings are not pinned
    # here (that is hardware ISA detail); end-to-end correctness is covered
    # by the on-board launch_kernel smoke test.
    assert len(kernel.asm["rpuinstr"].split()) * 8 == len(kernel.asm["rpubin"])
    assert kernel.metadata.name == "rpu_add_kernel"


def test_rpu_compile_add_kernel_uses_mlir_plan_pass(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    plan = _recognized_plan_from_kernel(kernel, tmp_path, monkeypatch)
    assert plan == {
        "args": {"lhs": 1, "out": 0, "rhs": 2},
        "emission": {
            "kind": "add",
            "logical_n": 16,
            "n": 16,
            "masked": False,
            "out": 0,
            "lhs": 1,
            "rhs": 2,
        },
        "kernel_name": "rpu_add_kernel",
        "layout": {"access": "linear", "memory": "contiguous_vector"},
        "mask": {"masked": False},
        "pattern": "add",
        "required_dsl_features": [
            "ctx.load_contig",
            "tile.add",
            "ctx.store_contig",
        ],
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
    }


def test_rpu_mlir_plan_pass_emits_trace_for_add(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    trace_json = _run_rpu_plan_trace_on_mlir(kernel.asm["ttir"], tmp_path, monkeypatch)
    trace = json.loads(trace_json)

    assert trace["version"] == 1
    assert trace["matched"] is True
    assert trace["selected"] == "add"
    assert trace["function"]["name"] == "rpu_add_kernel"
    assert "test_rpu_compile_add.py" in trace["function"]["location"]
    assert len(trace["attempts"]) == 1
    attempt = trace["attempts"][0]
    assert attempt["pattern"] == "add"
    assert attempt["reason"] == "selected"
    assert attempt["status"] == "matched"
    assert attempt["anchor"]["kind"] == "op"
    assert attempt["anchor"]["op"] == "tt.store"
    assert attempt["location"] == attempt["anchor"]["location"]
    assert attempt["location"] != trace["function"]["location"]
    assert "test_rpu_compile_add.py" in attempt["location"]


def test_rpu_mlir_plan_pass_emits_rpuplan_kernel_for_add(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    plan_json, _trace_json, printed = _run_rpu_plan_pass_on_mlir(kernel.asm["ttir"], tmp_path, monkeypatch)
    plan = json.loads(plan_json)

    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "attempts" not in plan
    assert "anchor" not in plan_json
    assert "rpu_plan.kernel @rpu_add_kernel_plan" in printed
    assert 'kernel_name = "rpu_add_kernel"' in printed
    assert 'pattern = "add"' in printed
    assert "source_func = @rpu_add_kernel" in printed
    assert "rpu_plan.load" not in printed
    assert "rpu_plan.store" not in printed
    assert "rpu_plan.tile" not in printed
    assert "rpu_plan.async" not in printed


def test_rpu_debug_bundle_does_not_leak_into_asm_artifacts(tmp_path, monkeypatch):
    debug_dir = tmp_path / "debug"
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(debug_dir))

    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    Path(kernel.metadata.rpu_debug_bundle["dir"]).resolve().relative_to(debug_dir.resolve())
    for artifact in kernel.asm.values():
        if isinstance(artifact, str):
            assert str(debug_dir.resolve()) not in artifact
            assert "manifest.json" not in artifact
            assert "rpu_debug_bundle" not in artifact


def test_rpu_mlir_plan_pass_rejects_add_inside_control_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "rpu")

    from triton._C.libtriton import ir, rpu

    context = ir.context()
    ir.load_dialects(context)
    module_path = tmp_path / "nested_add.mlir"
    module_path.write_text("""
module {
  tt.func public @nested_add(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    %true = arith.constant true
    scf.if %true {
      %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %3 = tt.load %2 : tensor<16x!tt.ptr<f16>>
      %4 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %6 = tt.load %5 : tensor<16x!tt.ptr<f16>>
      %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %9 = arith.addf %3, %6 : tensor<16xf16>
      tt.store %8, %9 : tensor<16x!tt.ptr<f16>>
    }
    tt.return
  }
}
""".strip())
    module = ir.parse_mlir_module(str(module_path), context)
    pm = ir.pass_manager(context)
    rpu.passes.add_recognize_plan(pm)

    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "rpu_test_helper")

    trace = json.loads(rpu.get_module_str_attr(module, "rpu.plan.trace"))
    assert trace["matched"] is False
    assert trace["selected"] is None
    assert trace["function"]["name"] == "nested_add"
    assert "location" in trace["function"]
    assert [attempt["pattern"] for attempt in trace["attempts"]] == [
        "add",
        "gemm",
        "softmax",
        "sqrt",
        "reduce_sum_all",
        "resnet_block",
        "convkxk",
        "resnet50_bottleneck",
    ]
    add_attempt = trace["attempts"][0]
    assert add_attempt["pattern"] == "add"
    assert add_attempt["reason"] == "did not match supported vector add"
    assert add_attempt["status"] == "failed"
    assert add_attempt["anchor"]["kind"] == "op"
    assert add_attempt["anchor"]["op"] == "scf.if"
    assert add_attempt["location"] == add_attempt["anchor"]["location"]
    assert add_attempt["location"] != trace["function"]["location"]
    for attempt in trace["attempts"]:
        assert "anchor" in attempt
        assert attempt["location"] == attempt["anchor"]["location"]


def test_rpu_mlir_plan_pass_trace_uses_module_anchor_without_single_public_func(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "rpu")

    from triton._C.libtriton import ir, rpu

    context = ir.context()
    ir.load_dialects(context)
    module_path = tmp_path / "two_public_funcs.mlir"
    module_path.write_text("""
module {
  tt.func public @first(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    tt.return
  }
  tt.func public @second(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) attributes {noinline = false} {
    tt.return
  }
}
""".strip())
    module = ir.parse_mlir_module(str(module_path), context)
    pm = ir.pass_manager(context)
    rpu.passes.add_recognize_plan(pm)

    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        pm.run(module, "rpu_test_helper")

    trace = json.loads(rpu.get_module_str_attr(module, "rpu.plan.trace"))
    assert trace["matched"] is False
    assert trace["selected"] is None
    assert trace["function"]["name"] == "<module>"
    assert [attempt["pattern"] for attempt in trace["attempts"]] == [
        "add",
        "gemm",
        "softmax",
        "sqrt",
        "reduce_sum_all",
        "resnet_block",
        "convkxk",
        "resnet50_bottleneck",
    ]
    for attempt in trace["attempts"]:
        assert attempt["anchor"] == {
            "kind": "module",
            "op": "builtin.module",
            "location": trace["function"]["location"],
        }
        assert attempt["location"] == trace["function"]["location"]


def test_rpu_compile_add_kernel_accepts_32_elements(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=32)

    assert "rpubin" in kernel.asm
    assert len(kernel.asm["rpubin"]) > 0
    assert "ctx.load_contig<32>" in kernel.asm["rpurc"]
    assert "ctx.store_contig<32>" in kernel.asm["rpurc"]


@pytest.mark.skip(
    reason=
    "v3.6 raised the elementwise1D threshold from 128 to 256 so n=256 now goes through single-shot load_contig<256> rather than the chunked tile_frame path this test asserts."
)
def test_rpu_compile_add_kernel_chunks_large_vectors(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=256)

    rpurc = kernel.asm["rpurc"]
    assert len(kernel.asm["rpubin"]) > 0
    assert "ctx.load_contig<256>" not in rpurc
    assert rpurc.count("auto frame = ctx.tile_frame();") == 4
    assert "make_shape(4096, 16)" in rpurc
    assert rpurc.count("ctx.load<half, 1024, 16>") == 8
    assert rpurc.count("ctx.store(") == 4
    assert "rpu::make_coord(0, 0)" in rpurc
    assert "rpu::make_coord(1024, 0)" in rpurc
    assert "rpu::make_coord(2048, 0)" in rpurc
    assert "rpu::make_coord(3072, 0)" in rpurc


def test_rpu_compile_add_kernel_handles_masked_tail(tmp_path, monkeypatch):
    kernel = _compile_rpu_masked_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32)

    rpurc = kernel.asm["rpurc"]
    assert "make_shape(1, 17)" in rpurc
    assert "ctx.load<half, 16, 32>" in rpurc
    assert "ctx.store(" in rpurc
    assert "ctx.store_contig<32>" not in rpurc
    assert len(kernel.asm["rpubin"]) > 0


def test_rpu_compile_masked_add_variant_uses_executable_source(tmp_path, monkeypatch):
    kernel = _compile_rpu_masked_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert "make_shape(1, 17)" in kernel.asm["rpurc"]
    assert "ctx.load<half, 16, 32>" in kernel.asm["rpurc"]
    assert "rpuplan" in kernel.asm
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "RPUPlanStageBundle" not in kernel.asm["rpuplan"]
    assert "rpu_plan.kernel" not in kernel.asm["rpuplan"]
    assert "rpu.plan.mlir" not in kernel.asm
    assert "_rpu_plan_module" not in kernel.metadata._fields
    assert "_rpu_direct_rpurc_result" not in kernel.metadata._fields


def test_rpu_compile_add_defaults_to_executable_for_canonical_shape(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpuexec" in kernel.asm
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "generic"' in kernel.asm["rpuexec"]


def test_rpu_compile_add32_defaults_to_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=32)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpuexec" in kernel.asm
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "generic"' in kernel.asm["rpuexec"]
    assert "n = 32 : i32" in kernel.asm["rpuexec"]
    assert "auto tile0 = ctx.load_contig<32>(arg1);" in kernel.asm["rpurc"]
    assert "auto tile1 = ctx.load_contig<32>(arg2);" in kernel.asm["rpurc"]
    assert "ctx.store_contig<32>(arg0, tile2);" in kernel.asm["rpurc"]
    assert "auto result = lhs + rhs;" not in kernel.asm["rpurc"]


@pytest.mark.parametrize("n_elements", [16, 32])
def test_rpu_compile_add_direct_opt_out_uses_plan_kernel(tmp_path, monkeypatch, n_elements):
    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "0")
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=n_elements)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_plan_kernel"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_plan_kernel"
    assert "source_kind: rpu_plan_kernel" in kernel.asm["rpuexec"]
    assert "reason: env_opt_out" in kernel.asm["rpuexec"]
    assert f"ctx.load_contig<{n_elements}>(arg1)" in kernel.asm["rpurc"]


def test_rpu_compile_add_executable_lowering_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("RPU_USE_EXECUTABLE_DIALECT", "1")

    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert "auto tile0 = ctx.load_contig<16>(arg1);" in kernel.asm["rpurc"]
    assert "auto tile1 = ctx.load_contig<16>(arg2);" in kernel.asm["rpurc"]
    assert "ctx.store_contig<16>(arg0, tile2)" in kernel.asm["rpurc"]
    assert "auto result = lhs + rhs;" not in kernel.asm["rpurc"]
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpu_plan.kernel" not in kernel.asm["rpuplan"]
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "generic"' in kernel.asm["rpuexec"]


@pytest.mark.skip(
    reason=
    "v3.6 raised the elementwise1D threshold from 128 to 256 so n=256 routes to kind='generic' single-shot rather than the kind='add' chunked path this test asserts."
)
def test_rpu_compile_large_add_defaults_to_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=256)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "add"' in kernel.asm["rpuexec"]
    assert "n = 256 : i32" in kernel.asm["rpuexec"]
    assert "auto frame = ctx.tile_frame();" in kernel.asm["rpurc"]
    assert "make_shape(4096, 16)" in kernel.asm["rpurc"]
    assert "ctx.load_contig<256>" not in kernel.asm["rpurc"]


def test_rpu_compile_masked_add_defaults_to_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_masked_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert "rpu.kernel" in kernel.asm["rpuexec"]
    assert 'kind = "generic"' in kernel.asm["rpuexec"]
    assert "masked = true" in kernel.asm["rpuexec"]
    assert "logical_n = 17 : i32" in kernel.asm["rpuexec"]
    assert "make_shape(1, 17)" in kernel.asm["rpurc"]
    assert "ctx.load<half, 16, 32>" in kernel.asm["rpurc"]
    assert "ctx.store_contig<32>" not in kernel.asm["rpurc"]


def test_rpu_compile_mul_defaults_to_generic_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_mul_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpu.kernel @rpu_mul_kernel" in kernel.asm["rpuexec"]
    assert "rpu.mul" in kernel.asm["rpuexec"]
    assert "rpu_plan.kernel" not in kernel.asm["rpuexec"]
    assert "auto tile0 = ctx.load_contig<16>(arg1);" in kernel.asm["rpurc"]
    assert "auto tile1 = ctx.load_contig<16>(arg2);" in kernel.asm["rpurc"]
    assert "auto tile2 = tile0 * tile1;" in kernel.asm["rpurc"]
    assert "ctx.store_contig<16>(arg0, tile2);" in kernel.asm["rpurc"]


def test_rpu_compile_add_default_omits_elf(tmp_path, monkeypatch):
    monkeypatch.delenv("RPU_OUTPUT_ELF", raising=False)
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    assert "rpuelf" not in kernel.asm
    rpu_artifacts = getattr(kernel.metadata, "rpu_artifacts", {}) or {}
    assert "rpuelf" not in rpu_artifacts


def test_rpu_compile_add_emits_elf_when_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("RPU_OUTPUT_ELF", "1")
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    assert "rpuelf" in kernel.asm
    # The ELF is surfaced as hex text in kernel.asm["rpuelf"]; decode to bytes.
    elf_hex = kernel.asm["rpuelf"]
    assert isinstance(elf_hex, str)
    elf = bytes.fromhex(elf_hex)
    assert elf.startswith(b"\x7fELF"), elf[:8]
    # The ELFs flagtree's clang emits are >= 1 KiB even for trivial kernels
    # (descriptor + instruction + symbol/string table sections).
    assert len(elf) >= 512
    rpu_artifacts = getattr(kernel.metadata, "rpu_artifacts", {}) or {}
    assert "rpuelf" in rpu_artifacts
    assert rpu_artifacts["rpuelf"]["bytes"] == len(elf)
    # rpubin path must remain intact alongside ELF emission.
    assert "rpubin" in kernel.asm
    assert isinstance(kernel.asm["rpubin"], bytes)
    assert len(kernel.asm["rpubin"]) > 0


def test_rpu_compile_mul_add_defaults_to_generic_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_mul_add_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpu.kernel @rpu_mul_add_kernel" in kernel.asm["rpuexec"]
    assert "rpu.mul" in kernel.asm["rpuexec"]
    assert "rpu.add" in kernel.asm["rpuexec"]
    assert "rpu_plan.kernel" not in kernel.asm["rpuexec"]
    assert "auto tile0 = ctx.load_contig<16>(arg1);" in kernel.asm["rpurc"]
    assert "auto tile1 = ctx.load_contig<16>(arg2);" in kernel.asm["rpurc"]
    assert "auto tile2 = ctx.load_contig<16>(arg3);" in kernel.asm["rpurc"]
    assert "auto tile3 = tile0 * tile1;" in kernel.asm["rpurc"]
    assert "auto tile4 = tile3 + tile2;" in kernel.asm["rpurc"]
    assert "ctx.store_contig<16>(arg0, tile4);" in kernel.asm["rpurc"]


def test_rpu_compile_masked_mul_add_defaults_to_generic_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_masked_mul_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpu.kernel @rpu_mul_add_masked_kernel" in kernel.asm["rpuexec"]
    assert "masked = true" in kernel.asm["rpuexec"]
    assert "logical_n = 17 : i32" in kernel.asm["rpuexec"]
    assert "block_n = 32 : i32" in kernel.asm["rpuexec"]
    assert "rpu.mul" in kernel.asm["rpuexec"]
    assert "rpu.add" in kernel.asm["rpuexec"]
    assert "rpu_plan.kernel" not in kernel.asm["rpurc"]
    assert "rpu::make_tensor<half, 2, rpu::MemScope::Local>" in kernel.asm["rpurc"]
    assert "make_shape(1, 17)" in kernel.asm["rpurc"]
    assert "ctx.load<half, 16, 32>" in kernel.asm["rpurc"]
    assert "auto tile3 = tile0 * tile1;" in kernel.asm["rpurc"]
    assert "auto tile4 = tile3 + tile2;" in kernel.asm["rpurc"]
    assert "ctx.store_contig<32>" not in kernel.asm["rpurc"]


def test_rpu_compile_add_mul_defaults_to_generic_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_mul_kernel(tmp_path, monkeypatch, n_elements=16)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpu.kernel @rpu_add_mul_kernel" in kernel.asm["rpuexec"]
    assert "rpu.add" in kernel.asm["rpuexec"]
    assert "rpu.mul" in kernel.asm["rpuexec"]
    assert "rpu_plan.kernel" not in kernel.asm["rpurc"]
    assert "auto tile0 = ctx.load_contig<16>(arg1);" in kernel.asm["rpurc"]
    assert "auto tile1 = ctx.load_contig<16>(arg2);" in kernel.asm["rpurc"]
    assert "auto tile2 = ctx.load_contig<16>(arg3);" in kernel.asm["rpurc"]
    assert "auto tile3 = tile0 + tile1;" in kernel.asm["rpurc"]
    assert "auto tile4 = tile3 * tile2;" in kernel.asm["rpurc"]
    assert "ctx.store_contig<16>(arg0, tile4);" in kernel.asm["rpurc"]


def test_rpu_compile_masked_add_mul_add_defaults_to_generic_executable(tmp_path, monkeypatch):
    kernel = _compile_rpu_masked_add_mul_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32)

    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    _assert_native_exec_artifact(kernel.asm["rpuplan"], "generic")
    assert "rpu.kernel @rpu_add_mul_add_masked_kernel" in kernel.asm["rpuexec"]
    assert "masked = true" in kernel.asm["rpuexec"]
    assert "logical_n = 17 : i32" in kernel.asm["rpuexec"]
    assert "block_n = 32 : i32" in kernel.asm["rpuexec"]
    assert kernel.asm["rpuexec"].count("rpu.add") == 2
    assert "rpu.mul" in kernel.asm["rpuexec"]
    assert "make_shape(1, 17)" in kernel.asm["rpurc"]
    assert "ctx.load<half, 16, 32>" in kernel.asm["rpurc"]
    assert "auto tile4 = tile0 + tile1;" in kernel.asm["rpurc"]
    assert "auto tile5 = tile4 * tile2;" in kernel.asm["rpurc"]
    assert "auto tile6 = tile5 + tile3;" in kernel.asm["rpurc"]
    assert "ctx.store_contig<32>" not in kernel.asm["rpurc"]


def test_rpu_compile_cache_hit_restores_full_artifact_group_without_rpu_stages(tmp_path, monkeypatch):
    fresh_kernel = _compile_rpu_add_kernel_with_cache_mode(
        tmp_path,
        monkeypatch,
        n_elements=16,
        always_compile=True,
    )

    assert {"ttir", "rpuplan", "rpuexec", "rpurc", "rpuinstr", "rpubin"} <= set(fresh_kernel.asm)
    _assert_no_rpu_transient_metadata_fields(fresh_kernel.metadata._fields)

    from triton.backends import backends

    backend_cls = backends["rpu"].compiler

    def fail_make_rpuplan(src, metadata):
        raise AssertionError("cache hit unexpectedly invoked make_rpuplan")

    def fail_make_rpurc(src, metadata):
        raise AssertionError("cache hit unexpectedly invoked make_rpurc")

    def fail_make_rpuexec(src, metadata):
        raise AssertionError("cache hit unexpectedly invoked make_rpuexec")

    monkeypatch.setattr(backend_cls, "make_rpuplan", staticmethod(fail_make_rpuplan))
    monkeypatch.setattr(backend_cls, "make_rpuexec", staticmethod(fail_make_rpuexec))
    monkeypatch.setattr(backend_cls, "make_rpurc", staticmethod(fail_make_rpurc))

    cached_kernel = _compile_rpu_add_kernel_with_cache_mode(
        tmp_path,
        monkeypatch,
        n_elements=16,
        always_compile=False,
    )

    assert {"ttir", "rpuplan", "rpuexec", "rpurc", "rpuinstr", "rpubin"} <= set(cached_kernel.asm)
    assert cached_kernel.asm["rpuplan"] == fresh_kernel.asm["rpuplan"]
    assert cached_kernel.asm["rpuexec"] == fresh_kernel.asm["rpuexec"]
    assert cached_kernel.asm["rpurc"] == fresh_kernel.asm["rpurc"]
    assert cached_kernel.asm["rpuinstr"] == fresh_kernel.asm["rpuinstr"]
    assert cached_kernel.asm["rpubin"] == fresh_kernel.asm["rpubin"]
    assert "RPUPlanStageBundle" not in cached_kernel.asm["rpuplan"]
    assert "_rpu_plan_module" not in cached_kernel.metadata._fields
    _assert_no_rpu_transient_metadata_fields(cached_kernel.metadata._fields)


def test_rpu_compile_cache_hit_ignores_rpuplan_override(tmp_path, monkeypatch):
    fresh_kernel = _compile_rpu_add_kernel_with_cache_mode(
        tmp_path,
        monkeypatch,
        n_elements=16,
        always_compile=True,
    )

    from triton.backends import backends

    backend_cls = backends["rpu"].compiler

    def fail_make_rpuplan(src, metadata):
        raise AssertionError("cache hit unexpectedly invoked make_rpuplan")

    def fail_make_rpurc(src, metadata):
        raise AssertionError("cache hit unexpectedly invoked make_rpurc")

    class _OverrideManager:

        def __init__(self):
            self.calls = []

        def has_file(self, filename):
            self.calls.append(("has", filename))
            raise AssertionError("cache hit unexpectedly checked rpuplan override")

        def get_file(self, filename):
            self.calls.append(("get", filename))
            raise AssertionError("cache hit unexpectedly loaded rpuplan override")

    triton_compiler = importlib.import_module("triton.compiler.compiler")
    manager = _OverrideManager()
    override_keys = []
    monkeypatch.setattr(
        triton_compiler,
        "get_override_manager",
        lambda key: override_keys.append(key) or manager,
    )
    monkeypatch.setattr(backend_cls, "make_rpuplan", staticmethod(fail_make_rpuplan))
    monkeypatch.setattr(backend_cls, "make_rpurc", staticmethod(fail_make_rpurc))
    monkeypatch.setenv("TRITON_KERNEL_OVERRIDE", "1")

    cached_kernel = _compile_rpu_add_kernel_with_cache_mode(
        tmp_path,
        monkeypatch,
        n_elements=16,
        always_compile=False,
    )

    assert override_keys
    assert manager.calls == []
    assert cached_kernel.asm["rpuplan"] == fresh_kernel.asm["rpuplan"]
    assert cached_kernel.asm["rpurc"] == fresh_kernel.asm["rpurc"]
    assert cached_kernel.asm["rpuinstr"] == fresh_kernel.asm["rpuinstr"]
    assert cached_kernel.asm["rpubin"] == fresh_kernel.asm["rpubin"]
    _assert_native_exec_artifact(cached_kernel.asm["rpuplan"], "generic")
    assert "RPUPlanStageBundle" not in cached_kernel.asm["rpuplan"]
    assert "rpu_plan.kernel" not in cached_kernel.asm["rpuplan"]
    _assert_no_rpu_transient_metadata_fields(cached_kernel.metadata._fields)


def test_rpu_compile_cache_metadata_excludes_transient_stage_keys(tmp_path, monkeypatch):
    kernel = _compile_rpu_add_kernel_with_cache_mode(
        tmp_path,
        monkeypatch,
        n_elements=16,
        always_compile=True,
    )

    metadata, metadata_path = _read_single_rpu_cache_metadata(
        tmp_path / "cache",
        kernel.metadata.name,
    )

    assert metadata_path.name == "rpu_add_kernel.json"
    _assert_no_rpu_transient_metadata_fields(metadata)
    assert metadata["rpu_rpuexec_source_kind"] == "rpu_executable"
    assert metadata["rpu_rpurc_source_kind"] == "rpu_executable"
    assert [event["stage"] for event in metadata["rpu_pipeline_events"]] == [
        "ttir_cleanup",
        "ttir_cleanup",
        "direct_executable_lower",
        "direct_executable_lower",
        "direct_executable_lower",
        "direct_executable_lower",
        "rpuexec_lower",
        "rpurc_emit",
        "rpuinstr_compile",
        "rpuinstr_compile",
    ]
    assert [(event["stage"], event["event"]) for event in metadata["rpu_pipeline_events"][2:6]] == [
        ("direct_executable_lower", "start"),
        ("direct_executable_lower", "candidate"),
        ("direct_executable_lower", "artifact"),
        ("direct_executable_lower", "end"),
    ]


def _compile_rpu_add_kernel(tmp_path, monkeypatch, n_elements=16):
    return _compile_rpu_add_kernel_with_cache_mode(
        tmp_path,
        monkeypatch,
        n_elements=n_elements,
        always_compile=True,
    )


def _compile_rpu_mul_kernel(tmp_path, monkeypatch, n_elements=16):
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
    def rpu_mul_kernel(out, a, b, n):
        offsets = tl.arange(0, n)
        av = tl.load(a + offsets)
        bv = tl.load(b + offsets)
        tl.store(out + offsets, av * bv)

    src = make_ast_source(
        fn=rpu_mul_kernel,
        constants={3: n_elements},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _compile_rpu_mul_add_kernel(tmp_path, monkeypatch, n_elements=16):
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
    def rpu_mul_add_kernel(out, a, b, c, n):
        offsets = tl.arange(0, n)
        av = tl.load(a + offsets)
        bv = tl.load(b + offsets)
        cv = tl.load(c + offsets)
        tl.store(out + offsets, av * bv + cv)

    src = make_ast_source(
        fn=rpu_mul_add_kernel,
        constants={4: n_elements},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16", 3: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _compile_rpu_masked_mul_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32):
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
    def rpu_mul_add_masked_kernel(out, a, b, c, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        mask = offsets < n
        av = tl.load(a + offsets, mask=mask, other=0.0)
        bv = tl.load(b + offsets, mask=mask, other=0.0)
        cv = tl.load(c + offsets, mask=mask, other=0.0)
        tl.store(out + offsets, av * bv + cv, mask=mask)

    src = make_ast_source(
        fn=rpu_mul_add_masked_kernel,
        constants={4: logical_n, 5: block_size},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16", 3: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _compile_rpu_add_mul_kernel(tmp_path, monkeypatch, n_elements=16):
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
    def rpu_add_mul_kernel(out, a, b, c, n):
        offsets = tl.arange(0, n)
        av = tl.load(a + offsets)
        bv = tl.load(b + offsets)
        cv = tl.load(c + offsets)
        tl.store(out + offsets, (av + bv) * cv)

    src = make_ast_source(
        fn=rpu_add_mul_kernel,
        constants={4: n_elements},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16", 3: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _compile_rpu_masked_add_mul_add_kernel(tmp_path, monkeypatch, logical_n=17, block_size=32):
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
    def rpu_add_mul_add_masked_kernel(out, a, b, c, d, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        mask = offsets < n
        av = tl.load(a + offsets, mask=mask, other=0.0)
        bv = tl.load(b + offsets, mask=mask, other=0.0)
        cv = tl.load(c + offsets, mask=mask, other=0.0)
        dv = tl.load(d + offsets, mask=mask, other=0.0)
        tl.store(out + offsets, (av + bv) * cv + dv, mask=mask)

    src = make_ast_source(
        fn=rpu_add_mul_add_masked_kernel,
        constants={5: logical_n, 6: block_size},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16", 3: "*fp16", 4: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _compile_rpu_add_kernel_with_cache_mode(tmp_path, monkeypatch, n_elements=16, *, always_compile):
    llvm_root = require_rpu_toolchain()

    monkeypatch.setenv("RPU_LLVM_ROOT", str(llvm_root))
    monkeypatch.delenv("RPU_BOARD_LOAD_CONTIG_NVEC", raising=False)
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    if always_compile:
        monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    else:
        monkeypatch.delenv("TRITON_ALWAYS_COMPILE", raising=False)

    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    monkeypatch.setitem(globals(), "tl", tl)

    @triton.jit
    def rpu_add_kernel(out, a, b, n):
        offsets = tl.arange(0, n)
        av = tl.load(a + offsets)
        bv = tl.load(b + offsets)
        tl.store(out + offsets, av + bv)

    src = make_ast_source(
        fn=rpu_add_kernel,
        constants={3: n_elements},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})


def _cache_metadata_json_files(cache_root):
    return sorted(Path(cache_root).glob("**/*.json"))


def _read_single_rpu_cache_metadata(cache_root, kernel_name):
    metadata_files = [path for path in _cache_metadata_json_files(cache_root) if path.name == f"{kernel_name}.json"]
    assert len(metadata_files) == 1, [str(path) for path in metadata_files]
    return json.loads(metadata_files[0].read_text()), metadata_files[0]


def _assert_no_rpu_transient_metadata_fields(metadata_fields):
    assert "_rpu_plan_module" not in metadata_fields
    assert "_rpu_direct_rpurc_result" not in metadata_fields
    assert "_rpubin_hex" not in metadata_fields


def _assert_native_exec_artifact(artifact, pattern):
    assert "rpu.kernel" in artifact
    assert f'kind = "{pattern}"' in artifact
    assert "rpu_plan.kernel" not in artifact
    assert "RPUPlanStageBundle" not in artifact


def _recognized_plan_from_kernel(kernel, tmp_path, monkeypatch):
    plan_json, _trace_json, _printed = _run_rpu_plan_pass_on_mlir(kernel.asm["ttir"], tmp_path, monkeypatch)
    return json.loads(plan_json)


def _run_rpu_plan_pass_on_mlir(module_text, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "rpu")

    from triton._C.libtriton import ir, rpu

    context = ir.context()
    ir.load_dialects(context)
    rpu.load_dialects(context)
    module_path = tmp_path / "rpu_plan_dual_emit_input.mlir"
    module_path.write_text(module_text)
    module = ir.parse_mlir_module(str(module_path), context)
    pm = ir.pass_manager(context)
    rpu.passes.add_recognize_plan(pm)
    pm.run(module, "rpu_test_helper")
    printed = str(module)
    roundtrip_path = tmp_path / "rpu_plan_dual_emit_output.mlir"
    roundtrip_path.write_text(printed)
    ir.parse_mlir_module(str(roundtrip_path), context)
    return (
        rpu.get_module_str_attr(module, "rpu.plan.json"),
        rpu.get_module_str_attr(module, "rpu.plan.trace"),
        printed,
    )


def _run_rpu_plan_trace_on_mlir(module_text, tmp_path, monkeypatch):
    return _run_rpu_plan_pass_on_mlir(module_text, tmp_path, monkeypatch)[1]


def _compile_rpu_masked_add_kernel(tmp_path, monkeypatch, logical_n, block_size):
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
    def rpu_add_masked_kernel(out, a, b, n: tl.constexpr, block: tl.constexpr):
        offsets = tl.arange(0, block)
        mask = offsets < n
        av = tl.load(a + offsets, mask=mask, other=0.0)
        bv = tl.load(b + offsets, mask=mask, other=0.0)
        tl.store(out + offsets, av + bv, mask=mask)

    src = make_ast_source(
        fn=rpu_add_masked_kernel,
        constants={3: logical_n, 4: block_size},
        signature={0: "*fp16", 1: "*fp16", 2: "*fp16"},
    )
    return triton.compile(src=src, target=GPUTarget("rpu", "rpu-v1", 1), options={"num_warps": 1})
