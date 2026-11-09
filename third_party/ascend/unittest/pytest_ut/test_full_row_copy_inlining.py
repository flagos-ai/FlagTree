"""Only compiler-versioned A5 SIMD row copies opt into libcall inlining."""
import subprocess
from types import SimpleNamespace

import pytest

from triton.backends.ascend import compiler


@pytest.mark.parametrize("marked,arch,mode,parallel,expected", [
    (True, "Ascend950PR_9579", "aiv", "simd", True),
    (False, "Ascend950PR_9579", "aiv", "simd", False),
    (True, "Ascend910B4", "aiv", "simd", False),
    (True, "Ascend950PR_9579", "mix", "mix_simd_simt", False),
    (True, "Ascend950PR_9579", "aiv", "simt", False),
])
def test_full_row_copy_inlining_scope(monkeypatch, marked, arch, mode, parallel, expected):
    monkeypatch.setattr(compiler, "_supports_full_row_copy_inlining", lambda _path: True)
    metadata = dict(has_full_row_copy=marked, target=SimpleNamespace(arch=arch), mix_mode=mode, parallel_mode=parallel)
    assert compiler._full_row_copy_compile_options(
        metadata, "/compiler") == (["--enable-lib-call-no-inline=false"] if expected else [])


@pytest.mark.parametrize("status,output,supported", [(0, "--enable-lib-call-no-inline", True), (0, "", False),
                                                     (1, "--enable-lib-call-no-inline", False)])
def test_compiler_option_probe(monkeypatch, status, output, supported):
    compiler._supports_full_row_copy_inlining.cache_clear()
    monkeypatch.setattr(compiler.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=status, stdout=output))
    assert compiler._supports_full_row_copy_inlining("/compiler") == supported
    compiler._supports_full_row_copy_inlining.cache_clear()


def test_probe_failure_keeps_default(monkeypatch):
    compiler._supports_full_row_copy_inlining.cache_clear()

    def fail(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("/compiler", 10)

    monkeypatch.setattr(compiler.subprocess, "run", fail)
    metadata = dict(has_full_row_copy=True, target=SimpleNamespace(arch="Ascend950PR"), mix_mode="aiv",
                    parallel_mode="simd")
    assert compiler._full_row_copy_compile_options(metadata, "/compiler") == []
    compiler._supports_full_row_copy_inlining.cache_clear()


@pytest.mark.parametrize("attrs,expected", [("tt.full_row_copy", True), ("tt.full_row_copy_extra", False), ("", False)])
def test_marker_is_derived_from_ir(attrs, expected):
    linalg = ('module attributes {' + attrs + '} { func.func @kernel() '
              'attributes {mix_mode = "aiv", parallel_mode = "simd"} { return } }')
    _, metadata = compiler._parse_linalg_metadata(linalg, {"has_full_row_copy": not expected})
    assert metadata["has_full_row_copy"] == expected
