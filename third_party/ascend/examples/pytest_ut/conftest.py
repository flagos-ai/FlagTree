import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def assign_npu(worker_id):
    npu_count = torch.npu.device_count()
    if worker_id == "master":
        npu_id = 0
    else:
        idx = int(worker_id.replace("gw", ""))
        npu_id = idx % npu_count
    torch.npu.set_device(npu_id)

