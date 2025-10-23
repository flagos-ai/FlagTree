import os
import pytest


def pytest_configure(config):
    # 仅在工作节点设置设备ID
    if hasattr(config, "workerinput"):
        worker_id = config.workerinput["workerid"]
        device_id = int(worker_id.replace("gw", ""))
        os.environ["ASCEND_DEVICE_ID"] = str(device_id)
        print(f"\n>> Worker {worker_id} using NPU device {device_id}")


# 可选：设备初始化逻辑
@pytest.fixture(scope="session", autouse=True)
def init_npu_device():
    if "ASCEND_DEVICE_ID" in os.environ:
        device_id = os.environ["ASCEND_DEVICE_ID"]
        # 在此处添加设备初始化代码
        print(f"Initializing NPU device {device_id}")