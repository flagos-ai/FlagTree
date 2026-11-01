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
bootstrap_feishu_table.py - Create or repair a Feishu Bitable data table

Idempotent: creates the table if it is missing, adds any field the schema
declares but the table lacks, and leaves everything else alone. Run once after
creating the Bitable document, and again whenever the schema grows.

Existing fields are never modified or deleted. A field whose type differs from
the schema is reported, not corrected, because changing a field's type can drop
data and a human should decide.

The schema is a JSON file so this script stays independent of any particular
workflow:

    {
      "table_name": "flaggems_runs",
      "dedup_field": "record_id",
      "fields": [{"field_name": "record_id", "type": 1}, ...]
    }

Prerequisites (manual, one-off):
  1. A Feishu custom app with the bitable:app scope, published.
  2. A Bitable document; its app_token is the /base/<app_token> part of the URL.
  3. The app added as a document collaborator with EDIT permission.
     This is the step most often missed, and the resulting API error does not
     say so.

Usage:
    FEISHU_APP_ID=... FEISHU_APP_SECRET=... \\
    python bootstrap_feishu_table.py --schema flaggems_runs.schema.json \\
        --app-token <app_token>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import feishu_client as fc


def load_schema(path: Path) -> Dict:
    """Read and validate a table schema JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{path}: 'fields' must be a non-empty list")

    seen = set()
    for field in fields:
        name = field.get("field_name")
        if not name:
            raise ValueError(f"{path}: every field needs a 'field_name'")
        if not isinstance(field.get("type"), int):
            raise ValueError(f"{path}: field '{name}' needs an integer 'type'")
        if name in seen:
            raise ValueError(f"{path}: duplicate field '{name}'")
        seen.add(name)

    dedup = schema.get("dedup_field")
    if dedup and dedup not in seen:
        raise ValueError(f"{path}: dedup_field '{dedup}' is not among the fields")

    return schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schema",
        required=True,
        type=Path,
        help="Path to the table schema JSON",
    )
    parser.add_argument(
        "--app-token",
        default=None,
        help="Bitable app token (default: $FEISHU_BITABLE_APP_TOKEN)",
    )
    parser.add_argument(
        "--table-name",
        default=None,
        help="Override the table name declared in the schema",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching the table",
    )
    return parser.parse_args()


def ensure_fields(token: str, app_token: str, table_id: str, wanted: List[Dict]) -> None:
    """Add missing fields; report type mismatches and extras."""
    existing = fc.field_types(token, app_token, table_id)

    missing = [f for f in wanted if f["field_name"] not in existing]
    mismatched = [f for f in wanted if f["field_name"] in existing and existing[f["field_name"]] != f["type"]]

    for field in missing:
        fc.create_field(token, app_token, table_id, field)
        print(f"[INFO] Added field {field['field_name']} (type={field['type']})")

    for field in mismatched:
        name = field["field_name"]
        print(f"[WARN] Field '{name}' is type {existing[name]}, "
              f"schema expects {field['type']}; left unchanged")

    extra = sorted(set(existing) - {f["field_name"] for f in wanted})
    if extra:
        print(f"[INFO] Fields in table but not in schema (kept): {', '.join(extra)}")

    if not missing and not mismatched:
        print("[INFO] Table already matches the schema, nothing to do")


def main() -> int:
    args = parse_args()

    try:
        schema = load_schema(args.schema)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[ERROR] {e}")
        return 1

    table_name = args.table_name or schema.get("table_name")
    if not table_name:
        print("[ERROR] No table name in the schema and none given")
        return 1

    wanted = schema["fields"]
    print(f"Schema declares {len(wanted)} fields for table '{table_name}'")

    # Checked before credentials, so the schema can be reviewed without them
    if args.dry_run:
        print("[DRY-RUN] Fields that would be ensured:")
        for field in wanted:
            print(f"  {field['field_name']:<24} type={field['type']}")
        return 0

    app_token = args.app_token or os.environ.get("FEISHU_BITABLE_APP_TOKEN", "").strip()
    if not app_token:
        print("[ERROR] --app-token or FEISHU_BITABLE_APP_TOKEN is required")
        return 1

    try:
        token = fc.get_token()
        table_id = fc.find_table(token, app_token, table_name)

        if table_id is None:
            print(f"Table '{table_name}' not found, creating it...")
            table_id = fc.create_table(token, app_token, table_name, wanted)
            print(f"[INFO] Created table {table_id} with {len(wanted)} fields")
        else:
            print(f"Table '{table_name}' exists ({table_id}), checking fields...")
            ensure_fields(token, app_token, table_id, wanted)

        print(f"\ntable_id={table_id}")
        return 0

    except fc.FeishuError as e:
        print(f"[ERROR] {e}")
        print("\nIf this is a permission error, confirm the app was added as a "
              "collaborator on the Bitable document with edit rights.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
