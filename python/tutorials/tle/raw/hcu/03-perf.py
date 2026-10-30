"""Compare deferred tle.raw SGEMM with the same HIP source launched directly."""

import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch

HERE = Path(__file__).parent
RAW = SourceFileLoader("hcu_raw_sgemm", str(HERE / "02-sgemm.py")).load_module()


def _pure_ms(m: int, n: int, k: int, iters: int) -> float:
    launcher = HERE / "03-pure-sgemm.hip"
    binary = Path("/tmp/hcu-pure-sgemm")
    # hipcc pulls in the device libraries (ockl) that threadIdx/blockIdx need.
    # -nogpulib is only for the Triton device-IR path, which links those
    # libraries itself.
    command = [
        "/opt/dtk/bin/hipcc",
        "-O3",
        "--offload-arch=gfx936",
        "-fno-exceptions",
        "-fno-rtti",
        "-I/opt/dtk/include",
        "-I/opt/dtk/hip/include",
        str(launcher),
        "-o",
        str(binary),
    ]
    build = subprocess.run(command, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"pure HIP build failed:\n{build.stderr}")
    run = subprocess.run([str(binary), str(m), str(n), str(k), str(iters)], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"pure HIP run failed:\n{run.stdout}\n{run.stderr}")
    for line in run.stdout.splitlines():
        if line.startswith("pure_ms"):
            return float(line.split()[1])
    raise RuntimeError(f"pure HIP did not report timing:\n{run.stdout}")


def _raw_ms(m: int, n: int, k: int, iters: int) -> float:
    a = torch.randn((m, k), device="cuda")
    b = torch.randn((k, n), device="cuda")
    for _ in range(3):
        RAW.sgemm(a, b)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        RAW.sgemm(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    m = n = k = 1024
    iters = 10
    # Correctness on a smaller shape is checked by 02-sgemm.py.
    pure = _pure_ms(m, n, k, iters)
    raw = _raw_ms(m, n, k, iters)
    ratio = raw / pure
    print("pure_ms", pure)
    print("raw_ms", raw)
    print("raw_over_pure", ratio)
    # Launch overhead on the Triton side is real; the device work should still
    # stay in the same band as the HIP kernel that only calls this source.
    if ratio > 1.25:
        raise SystemExit(f"raw is too slow versus pure HIP: {ratio:.3f}x")


if __name__ == "__main__":
    main()
