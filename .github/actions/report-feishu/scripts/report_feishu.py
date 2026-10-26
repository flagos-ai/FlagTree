#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
report_feishu.py - Push flat JSON records to a Feishu Bitable table

Generic on purpose: it takes a flat JSON object (or a JSONL file) and writes one
row per object, mapping each key to the field of the same name. Any workflow
that can produce such a file can report through it; nothing here knows about
FlagGems.

Keys with no matching field in the table are skipped rather than failing, so a
producer may emit more than the table currently shows.

Exits 0 even when reporting fails, unless --strict: this runs inside a PR gate,
and losing a dashboard row must not fail someone's PR.

Usage:
    FEISHU_APP_ID=... FEISHU_APP_SECRET=... FEISHU_BITABLE_APP_TOKEN=... \\
    python report_feishu.py --table-name flaggems_runs \\
        --record /root/flaggems_ci/PR/1400/record.json --dedup-field record_id
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import feishu_client as fc


def load_rows(record: Optional[Path], jsonl: Optional[Path]) -> List[Dict]:
    """Read rows from a single JSON object/array and/or a JSONL file."""
    rows: List[Dict] = []

    if record is not None:
        with open(record, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows.extend(payload if isinstance(payload, list) else [payload])

    if jsonl is not None:
        with open(jsonl, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] {jsonl}:{lineno} is not valid JSON ({e}), skipped")

    return [r for r in rows if isinstance(r, dict)]


def schema_types(path: Optional[Path]) -> Dict[str, int]:
    """Field name -> type code from a schema JSON, for offline coercion."""
    if path is None or not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return {
            f["field_name"]: f["type"]
            for f in schema.get("fields", [])
            if f.get("field_name") and isinstance(f.get("type"), int)
        }
    except (OSError, ValueError) as e:
        print(f"[WARN] Could not read schema {path}: {e}")
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Credentials come from the environment, never argv (argv is visible in `ps` on a
shared self-hosted runner):

  FEISHU_APP_ID
  FEISHU_APP_SECRET
  FEISHU_BITABLE_APP_TOKEN   may be used instead of --app-token

When the credentials are absent the script reports nothing and exits 0, so it
can be wired into CI before the secrets exist.

Examples:
  # One record, deduplicated on record_id
  python report_feishu.py --table-name flaggems_runs \\
      --record /root/flaggems_ci/PR/1400/record.json --dedup-field record_id

  # Backfill a whole ledger
  python report_feishu.py --table-name flaggems_runs \\
      --jsonl /root/flaggems_ci/Ledger/runs.jsonl --dedup-field record_id

  # Check the field mapping offline, no credentials needed
  python report_feishu.py --record record.json --dry-run \\
      --schema flaggems_runs.schema.json
        """,
    )
    parser.add_argument("--record", type=Path, help="Path to a flat JSON object")
    parser.add_argument("--jsonl", type=Path, help="Path to a JSONL file of objects")
    parser.add_argument(
        "--app-token",
        default=None,
        help="Bitable app token (default: $FEISHU_BITABLE_APP_TOKEN)",
    )
    parser.add_argument(
        "--table-name",
        default=None,
        help="Data table name; resolved to a table_id via the API",
    )
    parser.add_argument(
        "--table-id",
        default=None,
        help="Data table id, used directly instead of --table-name",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Schema JSON, used only by --dry-run to coerce values offline",
    )
    parser.add_argument(
        "--dedup-field",
        default=None,
        help=("Field to deduplicate on. An existing row with the same value is "
              "updated instead of a second row being added."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rows that would be written, without calling the API",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on failure (default: always exit 0)",
    )
    return parser.parse_args()


def resolve_table(args: argparse.Namespace, token: str, app_token: str) -> str:
    """Return the table_id to write to."""
    if args.table_id:
        return args.table_id
    if not args.table_name:
        raise fc.FeishuError("--table-id or --table-name is required")
    table_id = fc.find_table(token, app_token, args.table_name)
    if not table_id:
        raise fc.FeishuError(f"No table named '{args.table_name}'. Run bootstrap_feishu_table.py first.")
    return table_id


def report(args: argparse.Namespace, rows: List[Dict]) -> None:
    """Resolve the table, coerce rows to its field types, and write them."""
    app_token = args.app_token or os.environ.get("FEISHU_BITABLE_APP_TOKEN", "").strip()
    if not app_token:
        raise fc.FeishuError("--app-token or FEISHU_BITABLE_APP_TOKEN is required")

    token = fc.get_token()
    table_id = resolve_table(args, token, app_token)

    # Coerce against the LIVE field types rather than a local schema, so a
    # field someone changed in the UI is still written correctly.
    types = fc.field_types(token, app_token, table_id)
    if not types:
        raise fc.FeishuError(f"Table {table_id} reports no fields")

    pairs = [(row, fc.coerce_row(row, types)) for row in rows]
    pairs = [(original, fitted) for original, fitted in pairs if fitted]
    if not pairs:
        raise fc.FeishuError("No row had any field matching the table")

    created, updated = 0, 0
    to_create: List[Dict] = []

    for original, row in pairs:
        key = original.get(args.dedup_field) if args.dedup_field else None
        if not key:
            to_create.append(row)
            continue
        existing = fc.find_records(token, app_token, table_id, args.dedup_field, str(key))
        if existing:
            # Overwrite in place: a rerun of the same attempt should correct
            # its own row, not add a near-duplicate.
            for record_id in existing:
                fc.update_record(token, app_token, table_id, record_id, row)
                updated += 1
        else:
            to_create.append(row)

    if to_create:
        created = fc.create_records(token, app_token, table_id, to_create)

    print(f"[INFO] Feishu table {table_id}: {created} created, {updated} updated")


def do_dry_run(args: argparse.Namespace, rows: List[Dict]) -> int:
    """Show how rows would be coerced, without touching the API."""
    types = schema_types(args.schema)
    label = f"schema {args.schema}" if types else "no schema (values shown as-is)"
    print(f"[DRY-RUN] {len(rows)} row(s), coerced against {label}:")
    for row in rows[:3]:
        fitted = fc.coerce_row(row, types) if types else row
        print(json.dumps(fitted, ensure_ascii=False, indent=2, sort_keys=True))
        dropped = sorted(set(row) - set(fitted))
        if dropped:
            print(f"  (dropped: {', '.join(dropped)})")
    if len(rows) > 3:
        print(f"  ... and {len(rows) - 3} more")
    return 0


def main() -> int:
    args = parse_args()
    fail = 1 if args.strict else 0

    if not args.record and not args.jsonl:
        print("[ERROR] --record or --jsonl is required")
        return 1

    for path in (args.record, args.jsonl):
        if path is not None and not path.exists():
            print(f"[ERROR] Input not found: {path}")
            return fail

    try:
        rows = load_rows(args.record, args.jsonl)
    except (OSError, ValueError) as e:
        print(f"[ERROR] Failed to read input: {e}")
        return fail

    if not rows:
        print("[INFO] No rows to report")
        return 0

    if args.dry_run:
        return do_dry_run(args, rows)

    if not fc.credentials_present():
        print("[INFO] Feishu credentials not set, skipping report")
        return 0

    try:
        report(args, rows)
        return 0
    except Exception as e:
        print(f"[WARN] Failed to report to Feishu: {e}")
        return fail


if __name__ == "__main__":
    sys.exit(main())
