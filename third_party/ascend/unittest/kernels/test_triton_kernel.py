import importlib
import os
import urllib.request
from pathlib import Path

import pytest

import test_common


def discover_kernels():
    kernels = []
    kernels_root_path = Path(__file__).parents[0]
    for p in kernels_root_path.rglob("*.py"):
        if not p.is_file():
            continue
        if p.parent == kernels_root_path:
            continue
        rel = p.relative_to(kernels_root_path)
        if len(rel.parts) == 1 or p.name == "__init__.py":
            continue
        module_path = ".".join(rel.with_suffix("").parts)
        kernels.append((module_path, p.stem))
    return sorted(kernels, key=lambda x: x[1])


KERNEL_ITEMS = discover_kernels()


@pytest.mark.parametrize("module_path, kernel_name", KERNEL_ITEMS)
def test_triton_kernel(module_path, kernel_name, pytestconfig):
    selected = pytestconfig.getoption("kernel")
    if selected:
        if kernel_name not in selected:
            pytest.skip(f"skip {kernel_name} due to --kernel filter")
    base_url = "https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com"
    rel = module_path
    parts = rel.split(".") if rel else []
    pt_url = f"{base_url}/test/kernels/{parts[0]}_pt/{kernel_name}.pt"
    local_pt = Path(__file__).parent / f"{kernel_name}.pt"
    downloaded = False
    if not local_pt.exists():
        try:
            urllib.request.urlretrieve(pt_url, local_pt)
            downloaded = True
        except Exception as e:
            pytest.fail(
                f"Failed to download the {kernel_name}.pt file. Please check whether the {kernel_name}.pt file has been uploaded to the OBS bucket: {e}"
            )
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        pytest.fail(f"import {module_path} failed: {e}")

    if hasattr(mod, kernel_name):
        kernel_attr = kernel_name
    else:
        candidates = [a for a in dir(mod) if a.endswith("_kernel")]
        kernel_attr = candidates[0] if candidates else None

    if not kernel_attr:
        pytest.fail(f"No kernel callable found in {module_path}")

    kernel_callable = getattr(mod, kernel_attr)

    def runner(input_data, grid):
        kernel_callable[grid](**input_data)

    try:
        test_common.run_and_compare_ptfile(str(local_pt), runner, device_type='npu')
    finally:
        if downloaded and local_pt.exists():
            local_pt.unlink()
