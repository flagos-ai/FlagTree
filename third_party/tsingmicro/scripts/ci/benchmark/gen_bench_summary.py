#!/usr/bin/env python3
"""
Parse split benchmark logs and generate a per-operator summary JSON.

Usage:
  python gen_bench_summary.py <log_dir>

Output:
  <log_dir>/bench_summary.csv  — one entry per (op, dtype, shape), averaged across cards
"""

import os
import re
import sys
from typing import Dict, List, Optional


# Column-name → record-key mapping (substring match on header text)
_COL_MAP = [
    ("Torch Latency", "torch_lat"),
    ("Gems Latency",  "gems_lat"),
    ("Gems Speedup",  "speedup"),
    ("Kernel",        "kernel_time"),
    ("AP",            "ap_time"),
    ("TFLOPS",        "tflops"),
]


def _parse_header_columns(header_line: str) -> Dict[str, int]:
    """Parse the table header to map column-name → token-index in data lines.

    The header uses multi-word column names ("Torch Latency (ms)") so we
    locate each marker by character position, sort by position, and assign
    sequential indices that match the data line's split tokens.
    """
    markers = []
    for marker, key in _COL_MAP:
        pos = header_line.find(marker)
        if pos >= 0:
            markers.append((pos, key))
    markers.sort()
    # The Status column is first in header but not numeric — data tokens
    # start after it, so our 0 maps to the first numeric column in data.
    return {key: idx for idx, (_, key) in enumerate(markers)}


def _parse_dict_shape(dict_str: str) -> str:
    """Extract shape from a dict-format parameter block at end of line.

    Examples:
        {'end': 1073741824}             -> 'end=1073741824'
        {'size': [1073741824]}          -> '1073741824'
        {'size': [64, 512, 512]}        -> '64x512x512'
        {'n': 64}                       -> 'n=64'
    """
    # size: [N, M, ...]
    size_match = re.search(r"'size':\s*\[([^\]]*)\]", dict_str)
    if size_match:
        dims = size_match.group(1).replace(' ', '')
        return 'x'.join(dims.split(',')) if dims else "scalar"
    # end: N
    end_match = re.search(r"'end':\s*(\d+)", dict_str)
    if end_match:
        return f"end={end_match.group(1)}"
    # n: N
    n_match = re.search(r"'n':\s*(\d+)", dict_str)
    if n_match:
        return f"n={n_match.group(1)}"
    return "unknown"


def parse_single_log(filepath: str) -> List[dict]:
    """Parse a benchmark log file (per-op or per-card)."""
    records = []
    mode = "default"

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return records

    # Detect mode from header
    for line in lines:
        if 'Kernel (ms)' in line and 'AP (ms)' in line:
            mode = "show_all"
            break
        if '--host-time' in line:
            mode = "host_time"
            break

    # Extract pytest node ID from the log (e.g. "test_special_perf.py::test_perf_diag")
    test_func = ""
    for line in lines:
        m = re.search(r'(\S+\.py::\S+)', line)
        if m:
            test_func = m.group(1).rstrip('.').rstrip(',')
            break

    # Extract operator name from filename (e.g. "addmm_card11.log" → "addmm")
    basename = os.path.basename(filepath)
    file_op_name = re.sub(r'_card\d+\.log$', '', basename)

    card = None
    card_match = re.search(r'_card(\d+)', basename)
    if card_match:
        card = card_match.group(1)
    if card is None:
        for line in lines:
            m = re.match(r'=== card (\d+) op \S+ ===', line)
            if m:
                card = m.group(1)
                break

    current_op = None
    current_dtype = None
    col_index: Dict[str, int] = {}
    seen_operators = set()  # operator names seen in Operator: headers

    # --- Pre-check: detect pytest result ---
    # If all tests failed (0 passed, >0 failed), skip table parsing and
    # return a minimal FAILED record per dtype for statistics.
    # If no pytest summary found at all (crash), also treat as failed.
    all_failed = False
    has_summary = False
    for line in lines:
        m = re.search(r'=+\s+(\d+)\s+failed.*?(\d+)\s+passed.*?=+', line)
        if m:
            has_summary = True
            failed_n = int(m.group(1))
            passed_n = int(m.group(2))
            if failed_n > 0 and passed_n == 0:
                all_failed = True
            break
        # Also match "all failed, no passed" format: "=+ X failed, Y deselected =+"
        m2 = re.match(r'=+\s+(\d+)\s+failed,\s+\d+\s+deselected.*?=+', line)
        if m2:
            has_summary = True
            passed_m = re.search(r'(\d+)\s+passed', line)
            if not passed_m:
                all_failed = True
            break
        # Match success summary: "=+ X passed, Y deselected =+"
        m3 = re.match(r'=+\s+(\d+)\s+passed.*?=+', line)
        if m3:
            has_summary = True
            break

    if all_failed or not has_summary:
        # Collect operator names and dtypes from Operator: headers only
        for line in lines:
            op_match = re.match(
                r'Operator:\s+(.+?)\s{2,}Performance Test \(dtype=([^,]+),',
                line
            )
            if op_match:
                log_op = op_match.group(1)
                dtype = op_match.group(2).strip()
                records.append(dict(
                    op_name=file_op_name,
                    dtype=dtype,
                    shape="",
                    card=card,
                    mode="",
                    status="FAILED",
                    test_func=test_func,
                    _file=basename,
                    _raw="",
                    torch_lat=None, gems_lat=None, speedup=None,
                    kernel_time=None, ap_time=None, tflops=None,
                ))
        if not records:
            # No Operator: headers found either — crash before any output
            records.append(dict(
                op_name=file_op_name,
                dtype="",
                shape="",
                card=card,
                mode="",
                status="FAILED",
                test_func=test_func,
                _file=basename,
                _raw="",
                torch_lat=None, gems_lat=None, speedup=None,
                kernel_time=None, ap_time=None, tflops=None,
            ))
        return records
    # --- end pre-check ---

    for line in lines:
        op_match = re.match(
            r'Operator:\s+(.+?)\s{2,}Performance Test \(dtype=([^,]+),',
            line
        )
        if op_match:
            current_op = op_match.group(1)
            current_dtype = op_match.group(2).strip()
            col_index = {}
            seen_operators.add(current_op)
            continue

        # Table header: "Status    Torch Latency ..."
        if 'Status' in line and 'Torch Latency' in line:
            col_index = _parse_header_columns(line)
            continue

        if line.startswith('---'):
            continue
        if not line.strip():
            # Blank line ends the current table; reset column index
            col_index = {}
            continue

        if col_index and ('SUCCESS' in line or 'FAILED' in line):
            parts = line.split()

            # Extract shape(s): [torch.Size([...])] or [..., dim]
            shapes = re.findall(r'torch\.Size\(\[([^\]]*)\]\)', line)
            if shapes:
                shape = 'x'.join(s.strip().replace(' ', '') for s in shapes)
                dim_match = re.search(r'\[torch\.Size\(\[[^\]]*\]\),\s*(\d+)\]', line)
                if dim_match:
                    shape += f',dim={dim_match.group(1)}'
            else:
                # Fallback: try to extract shape from dict format
                # e.g. {'end': 1073741824, ...}  or {'size': [N, M], ...}  or {'n': 64, ...}
                dict_match = re.search(r"\{[^}]+\}\s*$", line)
                if dict_match:
                    shape = _parse_dict_shape(dict_match.group(0))
                else:
                    shape = "unknown"

            # Parse values by column name
            def _val(key):
                i = col_index.get(key)
                if i is None:
                    return None
                i += 1  # +1: skip Status column (parts[0] = "SUCCESS"/"FAILED")
                if i >= len(parts):
                    return None
                v = parts[i]
                if v == "N/A":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            rec = dict(
                op_name=file_op_name,
                dtype=current_dtype,
                shape=shape,
                card=card,
                mode=mode,
                status=parts[0],       # SUCCESS or FAILED
                test_func=test_func,    # pytest node ID
                _file=basename,        # source log file
                _raw=line.strip() if shape == "unknown" else "",
                torch_lat=_val("torch_lat"),
                gems_lat=_val("gems_lat"),
                speedup=_val("speedup"),
                kernel_time=_val("kernel_time"),
                ap_time=_val("ap_time"),
                tflops=_val("tflops"),
            )
            records.append(rec)

    # Warn if we saw Operator: headers but produced 0 records (likely format change)
    if seen_operators and not records:
        print(f"[WARN] {os.path.basename(filepath)}: "
              f"Operator(s) {sorted(seen_operators)} found but 0 records parsed — "
              f"unexpected data format, parser may need updating", file=sys.stderr)

    return records




def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_dir>", file=sys.stderr)
        sys.exit(1)

    log_dir = sys.argv[1]

    # Find log files: try ops/ (new), flag_gems/ (old), cards/ (oldest)
    log_files = []
    for subdir in ("ops", "flag_gems", "cards"):
        d = os.path.join(log_dir, subdir)
        if os.path.isdir(d):
            log_files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.log'))
            if log_files:
                break

    if not log_files:
        print(f"ERROR: no benchmark logs found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    all_records = []
    op_set = set()
    files_with_zero_records = []
    for fpath in log_files:
        recs = parse_single_log(fpath)
        if not recs:
            files_with_zero_records.append(os.path.basename(fpath))
        all_records.extend(recs)
        for r in recs:
            op_set.add(r['op_name'])

    # --- anomaly summary ---
    failed_records = [r for r in all_records if r.get('status') == 'FAILED']
    unknown_shape_records = [r for r in all_records if r.get('shape') == 'unknown']

    if unknown_shape_records:
        unknown_ops = sorted(set(r['op_name'] for r in unknown_shape_records))
        # per-operator record count (some ops may have partial parse failures)
        op_counts = {}
        for r in unknown_shape_records:
            op_counts[r['op_name']] = op_counts.get(r['op_name'], 0) + 1
        print(f"[WARN] {len(unknown_ops)} operator(s) with shape='unknown' "
              f"({len(unknown_shape_records)} records total, parser can't extract shape):")
        # Group by file for detail
        from collections import OrderedDict
        file_groups = OrderedDict()
        for r in unknown_shape_records:
            fname = r.get('_file', '?')
            if fname not in file_groups:
                file_groups[fname] = []
            file_groups[fname].append(r)
        for op in unknown_ops:
            print(f"       {op} ({op_counts[op]} records)")
        for fname, recs in file_groups.items():
            print(f"         {fname}:")
            for r in recs[:5]:
                dtype = r.get('dtype', '?')
                raw = r.get('_raw', '')[:120]
                print(f"           [{dtype}] {raw}")
            if len(recs) > 5:
                print(f"           ... and {len(recs) - 5} more")

    if failed_records:
        failed_ops = sorted(set(r['op_name'] for r in failed_records))
        print(f"[INFO] {len(failed_records)} FAILED record(s) across {len(failed_ops)} operator(s):")
        for op in failed_ops:
            files = sorted(set(r.get('_file', '?') for r in failed_records if r['op_name'] == op))
            print(f"       {op}  ({', '.join(files)})")
    # -----------------------

    # Keep records in original log order (no sort, no aggregation)

    # Determine which columns have data (dynamic based on mode)
    def _has_data(key):
        return any(r.get(key) is not None for r in all_records)

    _col_defs = [
        ("torch_lat",    "torch_lat(ms)"),   # only if torch baseline ran
        ("gems_lat",     "gems_lat(ms)"),    # always present
        ("speedup",      "speedup"),         # only if torch baseline ran
        ("kernel_time",  "kernel(ms)"),      # show_all only
        ("ap_time",      "ap(ms)"),          # show_all only
        ("tflops",       "tflops"),          # may or may not be present
    ]
    cols_display = ["op_name", "dtype", "shape", "test_func", "card", "status"]
    cols_key     = ["op_name", "dtype", "shape", "test_func", "card", "status"]
    for key, display in _col_defs:
        if _has_data(key):
            cols_display.append(display)
            cols_key.append(key)

    import csv
    out_dir = log_dir
    if not os.access(out_dir, os.W_OK):
        out_dir = os.getcwd()
        print(f"Note: log dir not writable, saving to {out_dir}/")
    out_path = os.path.join(out_dir, 'bench_summary.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols_display)
        for r in all_records:
            # Skip operator-level failure markers (shape="", no benchmark data)
            if not r.get('shape'):
                continue
            w.writerow([r.get(k, '') if r.get(k) is not None else '' for k in cols_key])

    # --- final summary ---
    total_files = len(log_files)
    # Count test_func per log file, no global dedup
    func_count = len(set((r.get('_file', ''), r.get('test_func', ''))
                         for r in all_records if r.get('test_func')))

    # Operator-level failures: operators whose entire log failed (no table data)
    op_failed_set = set(r['op_name'] for r in all_records if r.get('status') == 'FAILED')
    succ_ops = len(op_set) - len(op_failed_set)

    # Case-level: only count records with actual benchmark data (shape non-empty)
    case_records = [r for r in all_records if r.get('shape')]
    succ_cases = sum(1 for r in case_records if r.get('status') == 'SUCCESS')
    failed_cases = sum(1 for r in case_records if r.get('status') == 'FAILED')
    unknown_count = len(unknown_shape_records)
    total_cases = len(case_records)

    print(f"")
    print(f"=== Summary ===")
    print(f"Log files : {total_files}")
    print(f"Operators: {len(op_set)} (succ: {succ_ops}, failed: {len(op_failed_set)})")
    print(f"Functions: {func_count}")
    print(f"Cases    : {total_cases} (succ: {succ_cases}, failed: {failed_cases}, unsupported: {unknown_count})")
    print(f"Output   : {out_path}")


if __name__ == '__main__':
    main()
