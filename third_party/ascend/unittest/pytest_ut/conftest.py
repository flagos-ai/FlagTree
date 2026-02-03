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

import pytest


@pytest.fixture(scope="module", autouse=True)
def assign_npu(request, worker_id):
    marker = request.node.get_closest_marker("backend")
    if marker:
        backend = marker.args[0]
    else:
        backend = "torch_npu"
    if backend == "torch_npu":
        import torch
        npu_count = torch.npu.device_count()
        if worker_id == "master":
            npu_id = 0
        else:
            idx = int(worker_id.replace("gw", ""))
            npu_id = idx % npu_count
        torch.npu.set_device(npu_id)
    elif backend == "mindspore":
        import mindspore
        npu_count = mindspore.device_context.ascend.device_count()
        if worker_id == "master":
            npu_id = 0
        else:
            idx = int(worker_id.replace("gw", ""))
            npu_id = idx % npu_count
        mindspore.set_device("Ascend", npu_id)
