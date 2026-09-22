"""Unit tests for TLE-Raw TOPS backend (TOPSJITFunction & TOPSMLIRJITFunction)."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from triton.experimental.tle.raw.tops.runtime import (
    TOPSJITFunction,
    _find_tops_include_dir,
    _get_topscc_path,
    _get_gcu_arch,
)
from triton.experimental.tle.raw.tops.mlir_runtime import TOPSMLIRJITFunction

# ---------------------------------------------------------------------------
# TOPSJITFunction registration tests
# ---------------------------------------------------------------------------


class TestTOPSDialectRegistration:

    def test_tops_in_registry(self):
        from triton.experimental.tle.raw.runtime import registry
        assert "tops" in registry
        assert registry["tops"] is TOPSJITFunction

    def test_tops_mlir_in_registry(self):
        from triton.experimental.tle.raw.runtime import registry
        assert "tops_mlir" in registry
        assert registry["tops_mlir"] is TOPSMLIRJITFunction

    def test_cuda_still_in_registry(self):
        from triton.experimental.tle.raw.runtime import registry
        from triton.experimental.tle.raw.cuda.runtime import CUDAJITFunction
        assert "cuda" in registry
        assert registry["cuda"] is CUDAJITFunction

    def test_mlir_still_in_registry(self):
        from triton.experimental.tle.raw.runtime import registry
        from triton.experimental.tle.raw.mlir.runtime import MLIRJITFunction
        assert "mlir" in registry
        assert registry["mlir"] is MLIRJITFunction


# ---------------------------------------------------------------------------
# TOPSJITFunction constructor tests
# ---------------------------------------------------------------------------


class TestTOPSJITFunctionInit:

    def test_init_with_file(self, tmp_path):
        kernel_code = '__device__ void foo(float *a) { a[0] = 1.0f; }'
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text(kernel_code)

        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file)

        assert jit.code == kernel_code
        assert jit.file_name == str(kernel_file)
        assert jit.__triton_builtin__ is True
        assert jit.fn is fn

    def test_init_with_custom_arch(self, tmp_path):
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("__device__ void foo() {}")

        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file, arch="gcu300")
        assert jit.arch == "gcu300"

    def test_init_default_arch(self, tmp_path):
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("")
        fn = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCU_ARCH", None)
            jit = TOPSJITFunction(fn, file=kernel_file)
            assert jit.arch == "gcu400"

    def test_init_without_file(self):
        fn = MagicMock()
        jit = TOPSJITFunction(fn)
        assert jit.code == ""
        assert jit.file_name == "<inline>"

    def test_init_extra_flags(self, tmp_path):
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("")
        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file, extra_flags=["-DFOO=1", "-O3"])
        assert jit.extra_flags == ["-DFOO=1", "-O3"]

    def test_init_extra_flags_default(self, tmp_path):
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("")
        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file)
        assert jit.extra_flags == []

    def test_triton_builtin_flag(self):
        fn = MagicMock()
        jit = TOPSJITFunction(fn)
        assert hasattr(jit, "__triton_builtin__")
        assert jit.__triton_builtin__ is True


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:

    def test_get_gcu_arch_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GCU_ARCH", None)
            assert _get_gcu_arch() == "gcu400"

    def test_get_gcu_arch_env_gcu300(self):
        with patch.dict(os.environ, {"GCU_ARCH": "gcu300"}):
            assert _get_gcu_arch() == "gcu300"

    def test_get_gcu_arch_env_gcu410(self):
        with patch.dict(os.environ, {"GCU_ARCH": "gcu410"}):
            assert _get_gcu_arch() == "gcu410"

    def test_find_tops_include_dir_from_env(self, tmp_path):
        include_dir = tmp_path / "include"
        include_dir.mkdir()
        with patch.dict(os.environ, {"TOPS_INCLUDE_DIR": str(include_dir)}):
            result = _find_tops_include_dir()
            assert result == str(include_dir)

    def test_find_tops_include_dir_env_nonexistent(self, tmp_path):
        with patch.dict(os.environ, {"TOPS_INCLUDE_DIR": "/nonexistent/path"}):
            result = _find_tops_include_dir()
            assert result != "/nonexistent/path"

    def test_get_topscc_path_from_env(self, tmp_path):
        topscc = tmp_path / "topscc"
        topscc.touch()
        with patch.dict(os.environ, {"TOPSCC": str(topscc)}):
            assert _get_topscc_path() == str(topscc)

    def test_get_topscc_path_env_nonexistent(self):
        with patch.dict(os.environ, {"TOPSCC": "/nonexistent/topscc"}):
            result = _get_topscc_path()
            assert result != "/nonexistent/topscc"

    def test_get_topscc_path_from_caps(self, tmp_path):
        caps = tmp_path / "caps"
        (caps / "bin").mkdir(parents=True)
        topscc = caps / "bin" / "topscc"
        topscc.touch()
        with patch.dict(os.environ, {"CAPS_PATH": str(caps)}):
            os.environ.pop("TOPSCC", None)
            assert _get_topscc_path() == str(topscc)

    def test_get_topscc_fallback(self):
        with patch.dict(os.environ, {"CAPS_PATH": "/nonexistent/caps"}):
            os.environ.pop("TOPSCC", None)
            assert _get_topscc_path() == "topscc"


# ---------------------------------------------------------------------------
# @dialect decorator tests
# ---------------------------------------------------------------------------


class TestDialectDecorator:

    def test_dialect_creates_tops_jit(self, tmp_path):
        from triton.experimental.tle.raw import dialect

        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("__device__ void foo() {}")

        @dialect(name="tops", file=kernel_file)
        def edsl(*args, **kwargs):
            ...

        assert isinstance(edsl, TOPSJITFunction)
        assert edsl.__triton_builtin__ is True
        assert edsl.code == "__device__ void foo() {}"

    def test_dialect_creates_tops_mlir_jit(self):
        from triton.experimental.tle.raw import dialect

        @dialect(name="tops_mlir")
        def edsl(*args, **kwargs):
            ...

        assert isinstance(edsl, TOPSMLIRJITFunction)
        assert edsl.__triton_builtin__ is True

    def test_dialect_tops_with_arch(self, tmp_path):
        from triton.experimental.tle.raw import dialect

        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("")

        @dialect(name="tops", file=kernel_file, arch="gcu300")
        def edsl(*args, **kwargs):
            ...

        assert edsl.arch == "gcu300"

    def test_dialect_invalid_name_raises(self):
        from triton.experimental.tle.raw import dialect
        with pytest.raises(KeyError):

            @dialect(name="nonexistent_backend")
            def edsl(*args, **kwargs):
                ...


# ---------------------------------------------------------------------------
# TOPSMLIRJITFunction tests
# ---------------------------------------------------------------------------


class TestTOPSMLIRJITFunction:

    def test_init_defaults(self):
        fn = MagicMock()
        fn.__name__ = "test_fn"
        fn.__globals__ = {"a": 1}
        jit = TOPSMLIRJITFunction(fn)
        assert jit.arch == "gcu400"
        assert jit.__triton_builtin__ is True
        assert "cse" in jit.pipeline
        assert "convert-nvvm-to-llvm" not in jit.pipeline

    def test_init_custom_arch(self):
        fn = MagicMock()
        jit = TOPSMLIRJITFunction(fn, arch="gcu300")
        assert jit.arch == "gcu300"

    def test_init_custom_pipeline(self):
        fn = MagicMock()
        pipeline = ["convert-scf-to-cf", "cse"]
        jit = TOPSMLIRJITFunction(fn, pipeline=pipeline)
        assert jit.pipeline == pipeline

    def test_pipeline_no_nvvm(self):
        """TOPS MLIR pipeline should NOT include convert-nvvm-to-llvm."""
        fn = MagicMock()
        jit = TOPSMLIRJITFunction(fn)
        assert "convert-nvvm-to-llvm" not in jit.pipeline

    def test_deepcopy(self):
        import copy
        fn = MagicMock()
        fn.__name__ = "test_fn"
        fn.__globals__ = {}
        jit = TOPSMLIRJITFunction(fn, arch="gcu300")
        jit_copy = copy.deepcopy(jit)
        assert jit_copy.arch == "gcu300"
        assert jit_copy is not jit


# ---------------------------------------------------------------------------
# TOPSJITFunction compile flow tests (mock topscc)
# ---------------------------------------------------------------------------


class TestTOPSJITFunctionCompile:

    def test_compile_command_structure_new_style(self, tmp_path):
        """Verify topscc is called with correct flags (new-style topscc)."""
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("__device__ void foo() {}")

        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file, arch="gcu400")

        import subprocess

        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "mock error"
            return result

        with patch.object(TOPSJITFunction, "_detect_topscc_style", return_value="new"):
            with patch.object(subprocess, "run", side_effect=mock_run):
                with pytest.raises(RuntimeError, match="topscc compilation failed"):
                    jit._compile_tops_to_llvm_ir()

        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]
        assert "-x" in cmd
        assert "c++" in cmd
        assert "--device-only" in cmd
        assert "-emit-llvm" in cmd
        assert "-S" in cmd
        assert "-std=c++17" in cmd
        assert "-O2" in cmd
        assert any("--target=dtu-enflame-tops--gcu400" in c for c in cmd)
        assert any("--gcu-arch=gcu400" in c for c in cmd)

    def test_compile_command_structure_legacy_style(self, tmp_path):
        """Verify topscc is called with correct flags (legacy-style topscc)."""
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("__device__ void foo() {}")

        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file, arch="gcu400")

        import subprocess

        captured_cmds = []

        def mock_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "mock error"
            return result

        with patch.object(TOPSJITFunction, "_detect_topscc_style", return_value="legacy"):
            with patch.object(subprocess, "run", side_effect=mock_run):
                with pytest.raises(RuntimeError, match="topscc compilation failed"):
                    jit._compile_tops_to_llvm_ir()

        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]
        assert "-x" in cmd
        assert "tops" in cmd
        assert "--cuda-device-only" in cmd
        assert "-emit-llvm" in cmd
        assert "-S" in cmd
        assert "-std=c++17" in cmd
        assert any("--cuda-gpu-arch=gcu400" in c for c in cmd)

    def test_compile_success_returns_ir(self, tmp_path):
        """Verify successful compilation returns LLVM IR text."""
        kernel_file = tmp_path / "test.tops"
        kernel_file.write_text("__device__ void foo() {}")

        fn = MagicMock()
        jit = TOPSJITFunction(fn, file=kernel_file, arch="gcu400")

        mock_ir = "; ModuleID = 'test'\ntarget triple = \"dtu-enflame-tops--gcu400\"\n"

        import subprocess

        def mock_run(cmd, **kwargs):
            result = MagicMock()
            if "-S" in cmd:
                result.returncode = 0
                result.stdout = mock_ir
                result.stderr = ""
            else:
                result.returncode = 1
                result.stderr = "fallback error"
            return result

        with patch.object(TOPSJITFunction, "_detect_topscc_style", return_value="new"):
            with patch.object(subprocess, "run", side_effect=mock_run):
                ir_text = jit._compile_tops_to_llvm_ir()
                assert "ModuleID" in ir_text
                assert "dtu-enflame-tops--gcu400" in ir_text
