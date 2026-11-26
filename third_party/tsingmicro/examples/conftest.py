import pytest
import triton


@pytest.fixture
def device():
    return triton.runtime.driver.active.get_active_torch_device()
