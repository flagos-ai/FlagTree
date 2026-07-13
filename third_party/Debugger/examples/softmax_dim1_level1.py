"""Softmax dim1 level 1 example derived from Debugger sample 295."""

from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import debugger

OUTPUT_DIR = Path("/tmp/flagtree_debugger_softmax_level1_example")

debugger.configure(
    output_dir=OUTPUT_DIR,
    record_capacity=4096,
    export_raw_records=False,
)
triton.enable_debug(level=1, addr_level=1)


@triton.jit
def debug_softmax_dim1_kernel(x_ptr, y_ptr, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    tl.debug_collect_start(level=1, addr_level=1)
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    shifted = x - tl.max(x, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator
    tl.store(y_ptr + offsets, y, mask=mask)
    tl.debug_collect_end()


def main():
    device = "npu"
    rows = 4
    cols = 8
    block = 8
    torch.manual_seed(0)
    torch_npu.npu.manual_seed_all(0)

    x = torch.randn((rows, cols), dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    debug_softmax_dim1_kernel[(rows, )](x, y, cols, BLOCK_SIZE=block)
    torch_npu.npu.synchronize()

    expected = torch.nn.functional.softmax(x, dim=1)
    ok = torch.allclose(y.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)
    runs = debugger.take_exported_runs()

    print("sample=third_party/Debugger/samples/295_softmax")
    print(f"output_allclose={ok}")
    print(f"exported_runs={len(runs)}")
    for run in runs:
        print(f"report_path={run.get('report_path')}")
        print(f"meta={run.get('meta')}")


if __name__ == "__main__":
    main()
