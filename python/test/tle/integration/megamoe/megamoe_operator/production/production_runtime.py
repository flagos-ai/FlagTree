"""Small runtime shim shared by the production-shape MegaMoE candidates."""

from __future__ import annotations

import fcntl
import os
import site
import subprocess
import time
from pathlib import Path
from typing import Any


MEGA_SITE = Path(
    os.environ.get(
        "MEGAMOE_TORCH_SITE_PACKAGES",
        "/workspace/megakernel/.mega-venv/lib/python3.10/site-packages",
    )
)
NVSHMEM_HOME = Path(
    os.environ.get("NVSHMEM_HOME", str(MEGA_SITE / "nvidia" / "nvshmem"))
)

cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8")
os.environ.setdefault("CUDA_HOME", cuda_home)
os.environ["CPATH"] = (
    f"{cuda_home}/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
)
os.environ["LD_LIBRARY_PATH"] = (
    f"{NVSHMEM_HOME / 'lib'}:{cuda_home}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
)


def _import_env() -> dict[str, Any]:
    import triton
    import triton.language as tl
    import triton.experimental.tle.language as tle

    site.addsitedir(str(MEGA_SITE))
    import torch

    return {"torch": torch, "triton": triton, "tl": tl, "tle": tle}


def _compile_nvshmem_host_so(host_src: Path) -> Path:
    """Build the tiny host-side NVSHMEM wrapper once across all MPI ranks."""

    so_path = host_src.with_suffix(".so")
    lock_path = so_path.with_suffix(".so.lock")
    with lock_path.open("w") as lock_file:
        while True:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                break
            except BlockingIOError:
                time.sleep(0.1)
        if so_path.exists() and so_path.stat().st_mtime_ns >= host_src.stat().st_mtime_ns:
            return so_path
        cmd = [
            "nvcc",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-rdc=true",
            "-arch=sm_90a",
            f"-I{NVSHMEM_HOME / 'include'}",
            f"-L{NVSHMEM_HOME / 'lib'}",
            "-lnvshmem_host",
            "-lnvshmem_device",
            "-Xlinker",
            "-rpath",
            "-Xlinker",
            str(NVSHMEM_HOME / "lib"),
            "-o",
            str(so_path),
            str(host_src),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    return so_path
