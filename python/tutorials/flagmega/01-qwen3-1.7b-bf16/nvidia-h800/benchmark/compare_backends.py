#!/usr/bin/env python3
"""Run pinned backend workers in rotated rounds, retaining failures and gaps.

Each config entry provides the backend's own interpreter and actual adapter argv.
The placeholders {output}, {round_index}, and {measurement_id} are expanded by
this coordinator. Results stay diagnostic until independently audited; missing
backends never become reported performance values or implicit target passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from benchmark_model import sha, write_new


BACKENDS = ("flagmega", "vllm", "sglang", "tilefoundry", "llama.cpp", "flashrt")


def expand(argument, values):
    for name, value in values.items():
        argument = argument.replace("{" + name + "}", str(value))
    return argument


def aggregate(records):
    totals = {}
    for path in records:
        result = json.loads(path.read_text())
        if result.get("status") != "measured":
            continue
        backend = result["backend"]
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend {backend}")
        for scenario in result["scenarios"]:
            length = scenario["prompt_length"]
            for boundary, raw_key in [("host", "decode_step_ms"), ("device", "device_decode_step_ms")]:
                values = [v for run in scenario["runs"] for v in run.get(raw_key, [])]
                if not values:
                    continue
                key = (backend, length, boundary)
                item = totals.setdefault(key, {"tokens": 0, "milliseconds": 0, "measurement_ids": [], "sources": []})
                item["tokens"] += len(values)
                item["milliseconds"] += sum(values)
                item["measurement_ids"].append(result["measurement_id"])
                item["sources"].append({"path": str(path.resolve()), "sha256": sha(path)})
    return [{"backend": k[0], "prompt_length": k[1], "boundary": k[2],
             "tokens_per_second": v["tokens"]*1000/v["milliseconds"], **v} for k,v in totals.items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON with backend{name,argv,source_pins} entries and frozen workload/environment")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.rounds < 3:
        raise ValueError("Matched comparison requires at least3 rotated rounds")
    if args.results_dir.exists():
        raise ValueError("Use a new comparison cohort directory")
    config = json.loads(args.config.read_text())
    entries = config.get("backends", [])
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names) or any(name not in BACKENDS for name in names):
        raise ValueError("Unique registered backend names required")
    for entry in entries:
        if not entry.get("argv") or not entry.get("source_pins"):
            raise ValueError(f"{entry['name']} needs an actual executable argv and immutable source pins")
        if not Path(entry["argv"][0]).is_file():
            raise ValueError(f"Interpreter/worker binary does not exist: {entry['argv'][0]}")
        for pin in entry["source_pins"]:
            path = Path(pin["path"])
            if not path.is_file() or sha(path) != pin["sha256"]:
                raise ValueError(f"Changed/missing executable input: {path}")
    args.results_dir.mkdir(parents=True)
    report = {"schema": "flagmega.backend-cohort/v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": {"path": str(args.config.resolve()), "sha256": sha(args.config)},
              "workload": config["workload"], "environment": config["environment"],
              "rounds": args.rounds, "status": "preflight", "attempts": [],
              "missing_backends": [name for name in BACKENDS if name not in names],
              "acceptance": "pending independent boundary/state/source/reproduction/signature review"}
    if args.preflight_only:
        write_new(args.results_dir / "preflight.json", report)
        return 0
    records = []
    for round_index in range(args.rounds):
        order = entries[round_index % len(entries):] + entries[:round_index % len(entries)]
        for entry in order:
            name = entry["name"]
            stem = f"r{round_index:03d}-{name.replace('.', '_')}"
            output = args.results_dir / (stem + ".json")
            values = {"output": output.resolve(), "round_index": round_index, "measurement_id": stem}
            argv = [expand(value, values) for value in entry["argv"]]
            attempt = {"backend": name, "round_index": round_index, "argv": argv,
                       "source_pins": entry["source_pins"], "started_utc": datetime.now(timezone.utc).isoformat()}
            with (args.results_dir / (stem + ".stdout.log")).open("wb") as out, \
                    (args.results_dir / (stem + ".stderr.log")).open("wb") as err:
                attempt["exit_code"] = subprocess.run(argv, stdout=out, stderr=err, check=False).returncode
            attempt["ended_utc"] = datetime.now(timezone.utc).isoformat()
            if output.is_file():
                attempt["result"] = {"path": str(output.resolve()), "sha256": sha(output)}
                result = json.loads(output.read_text())
                if result.get("backend") != name or result.get("measurement_id") != stem:
                    raise ValueError("Worker result identity differs from dispatched actual request")
                if result.get("status") == "measured":
                    records.append(output)
            else:
                attempt["result"] = None
                failed = output.with_suffix(".failed.json")
                if failed.is_file():
                    attempt["failure"] = {"path": str(failed.resolve()), "sha256": sha(failed)}
            report["attempts"].append(attempt)
            write_new(args.results_dir / (stem + ".receipt.json"), attempt)
    report["aggregate"] = aggregate(records)
    report["status"] = "collected_incomplete" if report["missing_backends"] or any(
        attempt["exit_code"] != 0 or attempt["result"] is None for attempt in report["attempts"]) else "collected_pending_review"
    # Numerical compatibility, native-vs-graph boundaries and device isolation are
    # deliberately not decided by this raw-data coordinator.
    write_new(args.results_dir / "cohort.json", report)
    return 0 if report["status"] == "collected_pending_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
