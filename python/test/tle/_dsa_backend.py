# flagtree tle
"""Backend selector for the portable TLE DSA integration tests.

The DSA tests were originally written for TsingMicro (``txda``).  Ascend
exposes the same public ``tle.dsa`` surface but uses the 3.5 CommonIR
implementation, a different torch device name and UB (instead of SPM) as the
default on-chip buffer space.  Keep that mapping in one place so the test
kernels stay backend-agnostic.
"""

import pytest
import triton


def get_dsa_backend():
    """Return ``(backend, torch_device, on_chip_space)`` or skip the module.

    ``backend`` is one of ``"txda"``, ``"npu"`` or ``"ascend"``.
    ``torch_device`` is the string accepted by ``torch`` tensor constructors.
    ``on_chip_space`` is the value passed to ``tle.dsa.to_buffer`` and, on
    Ascend, ``tle.dsa.alloc(..., mem_addr_space=...)``.
    """
    torch = pytest.importorskip("torch")
    import triton.experimental.tle.language as tle

    try:
        target = triton.runtime.driver.active.get_current_target()
        backend = getattr(target, "backend", None)
    except Exception:
        backend = None

    if backend == "txda":
        pytest.importorskip("torch_txda", reason="TLE DSA txda tests require torch_txda")
        return "txda", "txda", tle.dsa.tsingmicro.SPM

    if backend in ("npu", "ascend"):
        pytest.importorskip("torch_npu", reason="TLE DSA Ascend tests require torch_npu")
        return backend, "npu", tle.dsa.ascend.UB

    # Fallback probes for environments where get_current_target() is not
    # decisive yet (for example before the driver has been initialized).
    try:
        pytest.importorskip("torch_npu", reason="no torch_npu")
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "ascend", "npu", tle.dsa.ascend.UB
    except pytest.skip.Exception:
        pass
    except Exception:
        pass

    try:
        pytest.importorskip("torch_txda", reason="no torch_txda")
        return "txda", "txda", tle.dsa.tsingmicro.SPM
    except pytest.skip.Exception:
        pass
    except Exception:
        pass

    pytest.skip("TLE DSA tests require Ascend (npu/torch_npu) or TsingMicro (txda/torch_txda) backend",
                allow_module_level=True)
