#!/usr/bin/env python3
"""End-to-end CANN validation driver for real NPU + external msprof."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/proton_cann_real")
    parser.add_argument("--msprof", default="msprof")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--aic-metrics",
        default="MemoryAccess",
        help="msprof --aic-metrics value. MemoryAccess exports per-op memory counters used for bandwidth.",
    )
    parser.add_argument(
        "--sys-hardware-mem-freq",
        type=int,
        default=100,
        help="Sampling frequency for msprof --sys-hardware-mem.",
    )
    parser.add_argument(
        "--disable-bandwidth-capture",
        action="store_true",
        help="Do not add msprof memory/bandwidth collection flags.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove --out before running.",
    )
    return parser


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = _make_arg_parser().parse_args()
    out = pathlib.Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    msprof_out = out / "msprof"
    msprof_out.mkdir(parents=True, exist_ok=True)
    # CANN msprof rejects --output directories that are writable by group/other.
    os.chmod(out, 0o700)
    os.chmod(msprof_out, 0o700)

    script_dir = pathlib.Path(__file__).resolve().parent
    workload = script_dir / "cann_real_npu_workload.py"
    post_import = script_dir / "cann_post_import_msprof.py"

    profile_base = out / "profile_run"
    post_import_base = out / "post_import"

    msprof_cmd = [
        args.msprof,
        "--msproftx=on",
        "--ai-core=on",
        f"--output={msprof_out}",
    ]
    if not args.disable_bandwidth_capture:
        msprof_cmd.extend([
            f"--aic-metrics={args.aic_metrics}",
            "--task-memory=on",
            "--sys-hardware-mem=on",
            f"--sys-hardware-mem-freq={args.sys_hardware_mem_freq}",
        ])

    msprof_cmd.extend([
        sys.executable,
        str(workload),
        "--name",
        str(profile_base),
        "--vendor-output",
        str(msprof_out),
        "--device",
        str(args.device),
        "--size",
        str(args.size),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
    ])
    _run(msprof_cmd)

    csv_files = sorted(msprof_out.rglob("*.csv"))
    print("exported_csv_count", len(csv_files))
    for path in csv_files[:40]:
        print("exported_csv", path)

    _run([
        sys.executable,
        str(post_import),
        "--base",
        str(post_import_base),
        "--msprof-output",
        str(msprof_out),
    ])

    print("DONE")
    print("profile_vendor_json", profile_base.with_suffix(".vendor.json"))
    print("post_import_vendor_json", post_import_base.with_suffix(".vendor.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
