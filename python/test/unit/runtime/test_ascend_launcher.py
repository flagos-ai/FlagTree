import functools
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture(params=["torch_npu", "mindspore"])
def launcher_backend(request, monkeypatch):
    driver = pytest.importorskip("triton.backends.ascend.driver")
    from triton.backends.ascend.backend_register import backend_strategy_registry

    # Launcher generation should not need an NPU or initialize either framework.
    monkeypatch.setattr(driver, "NPUUtils",
                        lambda: SimpleNamespace(get_aicore_num=lambda: 24, get_aivector_core_num=lambda: 48))
    monkeypatch.setattr(driver, "get_ascend_arch_from_env", lambda: "Ascend910B4")
    monkeypatch.setattr(driver, "is_ffts_supported", lambda arch: False)
    monkeypatch.setattr(driver, "get_backend_func",
                        functools.partial(backend_strategy_registry.execute_func, request.param))
    monkeypatch.setenv("TRITON_DEVICE_PRINT", "false")
    monkeypatch.setenv("TRITON_GRID_WARN_PRINT", "false")
    return driver


def make_launcher(driver, workspace_size, auto_blockify):
    metadata = SimpleNamespace(
        workspace_size=workspace_size,
        mix_mode="aiv",
        parallel_mode="aiv",
        compile_on_910_95=False,
        force_simt_only=False,
        enable_auto_blockify=auto_blockify,
    )
    return driver.make_launcher({}, {}, metadata)


@pytest.mark.parametrize("taskqueue", [False, True])
@pytest.mark.parametrize("auto_blockify", [False, True])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_launcher_workspace_arithmetic(launcher_backend, monkeypatch, tmp_path, taskqueue, auto_blockify, optimization):
    """Compile the generated arithmetic, without allocating multi-GiB workspaces."""
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("workspace arithmetic regression requires a C++ compiler")
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue).lower())
    boundary = (1 << 32) // 147456
    cases = [(147456, (0, 1, 1)), (147456, (1, 1, 1)), (147456, (16384, 1, 1)),  # 2.25 GiB, below the overflow boundary
             (147456, (boundary, 1, 1)), (147456, (boundary + 1, 1, 1)),
             (131072, (32768, 1, 1)),  # Exactly 4 GiB must not wrap to zero
             (147456, (32768, 1, 1)),  # 4.5 GiB, reported in issue #1194
             (196608, (32768, 1, 1)),  # 6 GiB
             (262144, (32768, 1, 1)),  # 8 GiB, another multiple of 2**32
             (147456, (128, 256, 1)), (147456, (32, 32, 32)),
             (1 << 32, (1, 1, 1)),  # A workspace literal already wider than int
             ]
    functions, calls, expected = [], [], []
    for index, (workspace_size, grid) in enumerate(cases):
        launcher = make_launcher(launcher_backend, workspace_size, auto_blockify)
        blocks = re.search(r"uint32_t blockNum4Workspace = [^;]+;", launcher)
        size = re.search(r"uint64_t totalWorkSpaceSize = [^;]+;", launcher)
        assert blocks is not None and size is not None
        allocation = launcher_backend.get_backend_func("allocate_memory", "totalWorkSpaceSize", "stream")
        assert allocation in launcher
        functions.append(f"""
uint64_t workspace_bytes_{index}(int gridX, int gridY, int gridZ) {{
    {blocks.group()}
    {size.group()}
    return totalWorkSpaceSize;
}}
""")
        calls.append(f"std::cout << workspace_bytes_{index}({', '.join(map(str, grid))}) << '\\n';")
        expected.append(workspace_size * grid[0] * grid[1] * grid[2])
    source = tmp_path / "workspace.cpp"
    executable = tmp_path / "workspace"
    source.write_text("#include <cstdint>\n#include <iostream>\n" + "\n".join(functions) + "\nint main() {\n" +
                      "\n".join(calls) + "\n}\n")
    subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", optimization,
                    str(source), "-o",
                    str(executable)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    actual = [int(line) for line in result.stdout.splitlines()]
    assert actual == expected


@pytest.mark.parametrize("workspace_size", [-1, 0])
def test_launcher_without_workspace(launcher_backend, monkeypatch, workspace_size):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", "true")
    launcher = make_launcher(launcher_backend, workspace_size, auto_blockify=False)
    assert "totalWorkSpaceSize" not in launcher
    assert "static_cast<void*>(workspace_addr_ptr)," not in launcher
