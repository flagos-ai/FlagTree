"""Compare deferred tle.raw simple_shift with the same HIP function.

Both sides launch one wave (64 threads). Outside mpirun this script starts
the two jobs; under mpirun it only times the raw kernel.
"""

import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).parent
ITERS = 1000


def _mpi_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    env["DUSHMEM_BOOTSTRAP"] = "MPI"
    env["OMPI_MCA_coll"] = "^hcoll"
    env["LD_LIBRARY_PATH"] = "/opt/dtk/lib/dushmem:" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("TRITON_CACHE_DIR", "/tmp/hcu-dushmem-perf-cache")
    return env


def _build_pure() -> Path:
    binary = Path("/tmp/dushmem-simple-shift-perf")
    command = [
        "/opt/dtk/bin/hipcc",
        "-fgpu-rdc",
        "--offload-arch=gfx936",
        "-O3",
        "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
        "-mcode-object-version=4",
        "-I/opt/dtk/include",
        "-I/opt/dtk/include/dushmem",
        "-I/opt/mpi/include",
        f"-I{HERE}",
        "-L/opt/dtk/lib/dushmem",
        "-L/opt/mpi/lib",
        str(HERE / "simple-shift-perf.hip"),
        "-o",
        str(binary),
        "-ldushmem_host",
        "-ldushmem_device",
        "-lmpi",
    ]
    build = subprocess.run(command, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"pure HIP build failed:\n{build.stderr}")
    return binary


def _mpirun(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["mpirun", "--allow-run-as-root", "-n", "2", *argv],
        capture_output=True,
        text=True,
        env=env,
    )


def _parse_ms(stdout: str, key: str) -> float:
    for line in stdout.splitlines():
        if line.startswith(key):
            return float(line.split()[1])
    raise RuntimeError(f"did not find {key} in:\n{stdout}")


def _raw_ms() -> None:
    import importlib.machinery
    simple_shift = importlib.machinery.SourceFileLoader(
        "hcu_dushmem_simple_shift", str(HERE / "simple-shift.py")).load_module()

    host = simple_shift._load_host()
    host, stream, stream_ptr, destination, destination_ptr, mype, npes = simple_shift.run_once(host, iters=10)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    with torch.cuda.stream(stream):
        for _ in range(ITERS):
            simple_shift.simple_shift_kernel[(1, )](destination, num_warps=1)
    end.record(stream)
    end.synchronize()
    if mype.value == 0:
        print("raw_ms", start.elapsed_time(end) / ITERS, flush=True)
    result = host.simple_shift_after_launch(stream_ptr, destination_ptr, mype, npes)
    if result != 0:
        raise SystemExit(f"PE {mype.value}: shift mismatch")
    host.simple_shift_finalize()


def main() -> None:
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        _raw_ms()
        return
    env = _mpi_env()
    binary = _build_pure()
    pure = _mpirun([str(binary), str(ITERS)], env)
    if pure.returncode != 0 and "pure_ms" not in pure.stdout:
        raise RuntimeError(f"pure run failed ({pure.returncode}):\n{pure.stdout}\n{pure.stderr}")
    raw = _mpirun(["python3", str(HERE / "perf.py")], env)
    if "raw_ms" not in raw.stdout:
        raise RuntimeError(f"raw run failed ({raw.returncode}):\n{raw.stdout}\n{raw.stderr}")
    pure_ms = _parse_ms(pure.stdout, "pure_ms")
    raw_ms = _parse_ms(raw.stdout, "raw_ms")
    ratio = raw_ms / pure_ms
    print("pure_ms", pure_ms)
    print("raw_ms", raw_ms)
    print("raw_over_pure", ratio)
    if ratio > 1.25:
        raise SystemExit(f"raw is too slow versus pure HIP: {ratio:.3f}x")


if __name__ == "__main__":
    main()
