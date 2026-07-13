"""Minimal FlagTree debugger level 2 full-dump abs example for Ascend."""

from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import debugger

OUTPUT_DIR = Path("/tmp/flagtree_debugger_level2_example")

debugger.configure(
    output_dir=OUTPUT_DIR,
    record_capacity=4096,
    export_raw_records=False,
)
triton.enable_debug(level=2, addr_level=2)


@triton.jit
def debug_abs_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    tl.debug_collect_start(level=2, addr_level=2)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.abs(x)
    z = y + 1.0
    tl.store(y_ptr + offsets, z, mask=mask)
    tl.debug_collect_end()


def main():
    device = "npu"
    n = 16
    block = 16
    x = torch.linspace(-8, 7, n, dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    debug_abs_kernel[(1, )](x, y, n, BLOCK_SIZE=block)
    torch_npu.npu.synchronize()

    expected = torch.abs(x) + 1.0
    ok = torch.allclose(y.cpu(), expected.cpu())
    runs = debugger.take_exported_runs()

    print(f"output_allclose={ok}")
    print(f"exported_runs={len(runs)}")
    for run in runs:
        report_path = run.get("report_path")
        artifacts_dir = (Path(report_path).with_suffix("").as_posix() + "_artifacts" if report_path else None)
        print(f"report_path={report_path}")
        print(f"artifacts_dir={artifacts_dir}")
        print(f"meta={run.get('meta')}")


if __name__ == "__main__":
    main()
