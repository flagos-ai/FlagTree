import os
import subprocess
from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--ptfile_path",
        type=str,
        default=None,
        help="the test-file path (pt file)"
    )


@pytest.fixture(scope="function")
def ptfile_path(request):
    filepath = request.config.getoption("--ptfile_path")
    if not filepath:
        # fatch default ptfile
        test_case_name = request.node.name
        remote_default_ptfile_url = f"https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/test/SGLang/{test_case_name}/default.pt"
        local_path = Path(f"default_{test_case_name}.pt")

        # download ptfile
        try:
            subprocess.run(
                ["curl", "-f", "-s", "-S", "-L", "-o", str(local_path), remote_default_ptfile_url],
                check=True,
                capture_output=True
            )
            print(f"default ptfile is saved to: {local_path}")
        except subprocess.CalledProcessError as e:
            pytest.fail("download default ptfile error!")

        if not local_path.exists():
            pytest.fail(f"the {local_path} does not exist!")

        return str(local_path)

    if not os.path.exists(filepath):
        pytest.fail(f"the {filepath} does not exist!")

    return filepath
