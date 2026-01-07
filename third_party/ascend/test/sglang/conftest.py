# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
