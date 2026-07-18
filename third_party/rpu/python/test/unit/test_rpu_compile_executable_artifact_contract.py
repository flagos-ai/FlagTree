import json
from pathlib import Path

from test_rpu_compile_add import _compile_rpu_add_kernel
from test_rpu_compile_conv import (
    _compile_rpu_convkxk_kernel,
    _compile_rpu_resnet50_bottleneck_kernel,
    _compile_rpu_resnet_block_kernel,
)
from test_rpu_compile_gemm import _compile_rpu_gemm_kernel
from test_rpu_compile_softmax import _compile_rpu_softmax_kernel


def _parse_executable_summary(artifact, tmp_path, monkeypatch, label):
    monkeypatch.setenv("TRITON_RPU_ACTIVE", "1")
    monkeypatch.setenv("RPU_ARCH", "rpu")
    monkeypatch.setenv("RPU_LOGICAL_LANES", "16")
    monkeypatch.setenv("TRITON_CODEGEN_BACKENDS", "rpu")

    from triton._C.libtriton import ir, rpu

    context = ir.context()
    ir.load_dialects(context)
    rpu.load_dialects(context)
    module_path = tmp_path / f"{label}.mlir"
    module_path.write_text(artifact)
    module = ir.parse_mlir_module(str(module_path), context)
    summary = rpu._get_rpuexec_kernel_summary(module)

    roundtrip_path = tmp_path / f"{label}.roundtrip.mlir"
    roundtrip_path.write_text(str(module))
    roundtrip = ir.parse_mlir_module(str(roundtrip_path), context)
    assert rpu._get_rpuexec_kernel_summary(roundtrip) == summary
    return summary


def _assert_executable_artifact_summary(
    artifact,
    *,
    tmp_path,
    monkeypatch,
    label,
    kernel_name,
    pattern,
):
    assert "rpu_plan.kernel" not in artifact
    assert "RPUPlanStageBundle" not in artifact
    summary = _parse_executable_summary(artifact, tmp_path, monkeypatch, label)
    assert summary == {"kernel_name": kernel_name, "pattern": pattern}


def _primary_rpuexec_artifact(kernel):
    assert kernel.metadata.rpu_rpuexec_source_kind == "rpu_executable"
    assert kernel.metadata.rpu_rpurc_source_kind == "rpu_executable"
    executable_artifact = kernel.asm["rpuexec"]
    assert executable_artifact
    return executable_artifact


def _assert_compat_rpuplan_mirrors_rpuexec(kernel, executable_artifact):
    assert kernel.asm["rpuplan"] == kernel.asm["rpuexec"]
    assert kernel.asm["rpuplan"] == executable_artifact


def _assert_debug_rpuexec_sidecar(
    kernel,
    tmp_path,
    monkeypatch,
    label,
    kernel_name,
    pattern,
    executable_artifact,
):
    debug_bundle = kernel.metadata.rpu_debug_bundle
    manifest = json.loads(Path(debug_bundle["manifest"]).read_text())
    assert "rpuexec_mlir" in debug_bundle["files"]
    sidecar_info = debug_bundle["files"]["rpuexec_mlir"]
    assert manifest["files"]["rpuexec_mlir"] == sidecar_info
    sidecar = Path(sidecar_info["path"])
    assert sidecar.name == "rpu.exec.mlir"
    sidecar_text = sidecar.read_text()
    assert sidecar_text == kernel.asm["rpuexec"]
    assert sidecar_text == executable_artifact
    _assert_executable_artifact_summary(
        sidecar_text,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        label=f"{label}_debug_sidecar",
        kernel_name=kernel_name,
        pattern=pattern,
    )


def test_rpu_compile_supported_executable_artifacts_parse_to_summary(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    cases = [
        (
            "add",
            "rpu_add_kernel",
            "generic",
            lambda case_tmp: _compile_rpu_add_kernel(case_tmp, monkeypatch),
        ),
        (
            "gemm",
            "rpu_gemm_kernel",
            "gemm",
            lambda case_tmp: _compile_rpu_gemm_kernel(case_tmp, monkeypatch, m=16, n=16, k=16),
        ),
        (
            "softmax",
            "rpu_softmax_kernel",
            "softmax",
            lambda case_tmp: _compile_rpu_softmax_kernel(case_tmp, monkeypatch, n_elements=16),
        ),
        (
            "convkxk",
            "rpu_convkxk_kernel",
            "convkxk",
            lambda case_tmp: _compile_rpu_convkxk_kernel(case_tmp, monkeypatch, kernel_size=3),
        ),
        (
            "resnet_block",
            "rpu_resnet_block_kernel",
            "resnet_block",
            lambda case_tmp: _compile_rpu_resnet_block_kernel(case_tmp, monkeypatch, pixels=16, channels=16,
                                                              hidden_channels=16),
        ),
        (
            "resnet50_bottleneck",
            "rpu_resnet50_bottleneck_kernel",
            "resnet50_bottleneck",
            lambda case_tmp: _compile_rpu_resnet50_bottleneck_kernel(case_tmp, monkeypatch),
        ),
    ]

    for label, kernel_name, pattern, compile_case in cases:
        case_tmp = tmp_path / label
        case_tmp.mkdir()
        kernel = compile_case(case_tmp)

        executable_artifact = _primary_rpuexec_artifact(kernel)
        _assert_executable_artifact_summary(
            executable_artifact,
            tmp_path=case_tmp,
            monkeypatch=monkeypatch,
            label=f"{label}_rpuexec",
            kernel_name=kernel_name,
            pattern=pattern,
        )
        _assert_compat_rpuplan_mirrors_rpuexec(kernel, executable_artifact)
        _assert_executable_artifact_summary(
            kernel.asm["rpuplan"],
            tmp_path=case_tmp,
            monkeypatch=monkeypatch,
            label=f"{label}_rpuplan_compat",
            kernel_name=kernel_name,
            pattern=pattern,
        )
        _assert_debug_rpuexec_sidecar(
            kernel,
            case_tmp,
            monkeypatch,
            label,
            kernel_name,
            pattern,
            executable_artifact,
        )


def test_rpu_compile_rpuexec_is_primary_executable_artifact_identity(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("RPU_DEBUG_ARTIFACT_DIR", str(tmp_path / "debug"))
    kernel = _compile_rpu_add_kernel(tmp_path, monkeypatch)

    executable_artifact = _primary_rpuexec_artifact(kernel)
    _assert_executable_artifact_summary(
        executable_artifact,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        label="d151_add_primary_rpuexec",
        kernel_name="rpu_add_kernel",
        pattern="generic",
    )
    _assert_compat_rpuplan_mirrors_rpuexec(kernel, executable_artifact)
    _assert_debug_rpuexec_sidecar(
        kernel,
        tmp_path,
        monkeypatch,
        "d151_add",
        "rpu_add_kernel",
        "generic",
        executable_artifact,
    )
