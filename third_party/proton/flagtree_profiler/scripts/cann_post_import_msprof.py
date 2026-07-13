#!/usr/bin/env python3
"""Import already-exported CANN msprof CSV files into a Proton vendor artifact."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter

import triton.profiler as proton


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base path for output artifacts.")
    parser.add_argument(
        "--msprof-output",
        required=True,
        help="Directory containing PROF_*/mindstudio_profiler_output/*.csv.",
    )
    parser.add_argument("--metrics", default="aicore,bandwidth")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success even if no CSV files were imported.",
    )
    return parser


def main() -> int:
    args = _make_arg_parser().parse_args()
    base = pathlib.Path(args.base)
    msprof_output = pathlib.Path(args.msprof_output)
    base.parent.mkdir(parents=True, exist_ok=True)

    mode = ("runtime_base:"
            f"vendor_metrics={args.metrics}:"
            f"msprof_import_path={msprof_output}:"
            "runtime_host_timing_fallback=false:"
            "aclprof_runtime_enabled=false:"
            "aclprof_auto_export=false:"
            "mstx_enabled=false:"
            "aclprof_msproftx_enabled=false")
    session_id = proton.start(
        name=str(base),
        context="shadow",
        data="tree",
        hook="triton",
        backend="cann",
        mode=mode,
    )
    proton.finalize(session_id)

    vendor_path = base.with_suffix(".vendor.json")
    meta_path = base.with_suffix(".meta.json")
    vendor = json.loads(vendor_path.read_text())
    meta = json.loads(meta_path.read_text())
    source_counts = Counter(assoc.get("source", "") for assoc in vendor.get("associations", []))

    print("post_import_vendor_json", vendor_path)
    print("post_import_meta_json", meta_path)
    print("raw_inputs", len(vendor.get("raw_inputs", [])))
    for path in vendor.get("raw_inputs", [])[:20]:
        print("raw_input", path)
    print("association_sources", dict(source_counts))
    print("meta_degrade_reasons")
    for reason in meta.get("degrade_reasons", []):
        print("-", reason)

    if not args.allow_empty and not vendor.get("raw_inputs"):
        print("ERROR: no msprof CSV files were imported")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
