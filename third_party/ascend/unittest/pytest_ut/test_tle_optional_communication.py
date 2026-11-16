"""Non-communication TLE users must not require the optional shmem package."""
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(scope="module", autouse=True)
def assign_npu():
    """These subprocess checks inspect imports without initializing a device."""
    yield


@pytest.mark.parametrize("entry", ["import_only", "init_communicator", "create_dist_tensor", "cleanup_communicator"])
def test_shmem_is_required_only_by_communication(entry):
    code = textwrap.dedent("""
        import importlib
        import sys

        class MissingShmem:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "shmem" or fullname.startswith("shmem."):
                    raise ModuleNotFoundError("shmem deliberately unavailable", name="shmem")

        for name in list(sys.modules):
            if name == "shmem" or name.startswith("shmem."):
                del sys.modules[name]
        sys.meta_path.insert(0, MissingShmem())
        module = importlib.import_module("triton.experimental.tle.language.dsa.ascend.communication")
        entry = sys.argv[1]
        if entry != "import_only":
            import torch
            args = (torch.zeros(1),) if entry == "create_dist_tensor" else (None,) if entry == "cleanup_communicator" else ()
            try:
                getattr(module, entry)(*args)
            except ModuleNotFoundError as error:
                assert error.name == "shmem", error
            else:
                raise AssertionError("Communication must still require shmem")
        print("optional-shmem check passed")
    """)
    result = subprocess.run([sys.executable, "-c", code, entry], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "optional-shmem check passed" in result.stdout
