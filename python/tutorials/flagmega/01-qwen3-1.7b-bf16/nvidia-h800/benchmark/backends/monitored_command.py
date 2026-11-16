#!/usr/bin/env python3
"""Retain physical-GPU telemetry around a real isolated backend subprocess."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--monitor-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.monitor_file.exists() or args.receipt.exists():
        raise ValueError("Use new immutable monitoring output paths")
    command = json.loads(args.command_file.read_text())["argv"]
    query = ["nvidia-smi", "--id", str(args.physical_gpu),
             "--query-gpu=timestamp,index,uuid,name,driver_version,pstate,temperature.gpu,power.draw,"
             "power.limit,clocks.sm,clocks.mem,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "--loop-ms=500"]
    started = datetime.now(timezone.utc).isoformat()
    start_ns = time.monotonic_ns()
    with args.monitor_file.open("xb") as output:
        monitor = subprocess.Popen(query, stdout=output)
        try:
            result = subprocess.run(command, check=False)
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
    report = {"schema": "flagmega.backend-monitor/v1", "started_utc": started,
              "ended_utc": datetime.now(timezone.utc).isoformat(), "start_monotonic_ns": start_ns,
              "end_monotonic_ns": time.monotonic_ns(), "argv": command, "exit_code": result.returncode,
              "physical_gpu": args.physical_gpu, "monitor_argv": query, "monitor_pid": monitor.pid,
              "monitor_exit_code": monitor.returncode, "monitor_sha256": sha(args.monitor_file),
              "monitor_bytes": args.monitor_file.stat().st_size, "poll_interval_ms": 500,
              "command_file_sha256": sha(args.command_file), "wrapper_sha256": sha(__file__),
              "scope": "includes initialization/warmup/measurement; raw telemetry, not an isolation or overhead pass"}
    with args.receipt.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
