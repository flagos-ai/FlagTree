# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""The cached host allocator must be bound to its SDK and build ABI."""

from pathlib import Path
from types import SimpleNamespace
import tempfile

import pytest

from triton.experimental.tle.language import communication


@pytest.fixture
def allocator_environment(monkeypatch, tmp_path):
    sdk = tmp_path / "sdk"
    include = sdk / "include"
    include.mkdir(parents=True)
    (sdk / "libflagcx.so").write_bytes(b"current host SDK")
    (include / "flagcx.h").write_text("current SDK declarations")
    compiled, loaded = [], []

    def build(*, name, build_directory, extra_ldflags, **_kwargs):
        compiled.append((Path(build_directory), tuple(extra_ldflags)))
        Path(build_directory).mkdir(parents=True, exist_ok=True)
        (Path(build_directory) / f"{name}.so").write_bytes(b"current allocator")

    def load(path, *_args):
        loaded.append(Path(path))
        return SimpleNamespace(allocator=lambda: object())

    torch = SimpleNamespace(
        __version__="test-torch-1",
        version=SimpleNamespace(cuda="12.8"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
        distributed=SimpleNamespace(is_available=lambda: False),
    )
    for name, value in {
            "tempfile": tempfile,
            "torch": torch,
            "FLAGCX_LIB_PATH": str(sdk),
            "FLAGCX_INCLUDE_PATH": str(include),
            "flagcx_allocator_source": "allocator source",
            "load_inline": build,
            "CUDAPluggableAllocator": load,
            "_allocator": None,
            "_allocator_wrapper": None,
            "_flagcx_allocator_failed_to_compile": False,
    }.items():
        monkeypatch.setattr(communication, name, value, raising=False)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_RANK", "0")
    return SimpleNamespace(sdk=sdk, include=include, compiled=compiled, loaded=loaded, torch=torch, temporary=tmp_path)


def test_existing_legacy_allocator_cannot_select_a_different_sdk(allocator_environment):
    env = allocator_environment
    legacy = env.temporary / "flagcx_allocator.so"
    legacy.write_bytes(b"allocator linked to a different SDK")

    communication.compile_flagcx_allocator()

    assert len(env.compiled) == 1
    assert env.loaded[0] != legacy
    assert env.loaded[0].read_bytes() == b"current allocator"
    assert legacy.read_bytes() == b"allocator linked to a different SDK"
    assert f"-Wl,-rpath,{env.sdk}" in env.compiled[0][1]


def test_same_allocator_inputs_reuse_the_compilation(allocator_environment):
    env = allocator_environment
    communication.compile_flagcx_allocator()
    communication.compile_flagcx_allocator()
    assert len(env.compiled) == 1
    assert env.loaded[0] == env.loaded[1]


@pytest.mark.parametrize("changed_input", ["host_library", "header", "source", "torch", "cuda"])
def test_changed_allocator_input_cannot_reuse_old_library(allocator_environment, monkeypatch, changed_input):
    env = allocator_environment
    communication.compile_flagcx_allocator()
    if changed_input == "host_library":
        (env.sdk / "libflagcx.so").write_bytes(b"changed SDK ABI")
    elif changed_input == "header":
        (env.include / "flagcx.h").write_text("changed SDK declarations")
    elif changed_input == "source":
        monkeypatch.setattr(communication, "flagcx_allocator_source", "changed source")
    elif changed_input == "torch":
        env.torch.__version__ = "test-torch-2"
    else:
        env.torch.version.cuda = "13.0"
    communication.compile_flagcx_allocator()
    assert len(env.compiled) == 2
    assert env.loaded[0] != env.loaded[1]


def test_identical_sdk_bytes_at_another_path_rebuild_the_runtime_link(allocator_environment, monkeypatch):
    env = allocator_environment
    communication.compile_flagcx_allocator()
    other = env.temporary / "other-sdk"
    other.mkdir()
    (other / "libflagcx.so").write_bytes((env.sdk / "libflagcx.so").read_bytes())
    monkeypatch.setattr(communication, "FLAGCX_LIB_PATH", str(other))
    communication.compile_flagcx_allocator()
    assert len(env.compiled) == 2
    assert env.loaded[0] != env.loaded[1]
    assert f"-Wl,-rpath,{other}" in env.compiled[1][1]
