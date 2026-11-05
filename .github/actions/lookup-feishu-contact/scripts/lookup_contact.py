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
lookup_contact.py - Resolve a GitHub login to a Feishu user, and check skips

Two lookups against the Bitable, both of which decide whether a CI failure
turns into a direct message:

    --login <github-login>    -> open_id of the person to notify
    --workflow <workflow>     -> whether that workflow's failures are skipped

Both tables are created on first use from the schemas next to this script, so
setting this up means filling in rows, not defining columns.

Nothing is notified when a login has no row. That is deliberate: a message
about a stranger's failing PR is noise to whoever gets it, so an unmapped
login is silence plus a warning, never a fallback to a group chat.

Writes GitHub step outputs when $GITHUB_OUTPUT is set:

    open-ids    resolved ids, comma separated, empty when none resolved
    names       display names, comma separated, for log lines
    skip        'true' when this workflow should not notify
    skip-reason why, when it is skipped

Usage:
    FEISHU_APP_ID=... FEISHU_APP_SECRET=... FEISHU_BITABLE_APP_TOKEN=... \\
    python lookup_contact.py --login liuhaoyutz --workflow Nvidia3.6-Gems-Test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import feishu_client as fc

HERE = Path(__file__).resolve().parent
CONTACTS_SCHEMA = HERE / "ci_contacts.schema.json"
SKIP_SCHEMA = HERE / "ci_notify_skip.schema.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--login", default="", help="GitHub login(s) to resolve, space or comma separated")
    parser.add_argument("--workflow", default="", help="Workflow name to check against the skip list")
    parser.add_argument("--app-token", default=None, help="Bitable app token (default: $FEISHU_BITABLE_APP_TOKEN)")
    parser.add_argument("--contacts-table", default="", help="Override the contacts table name")
    parser.add_argument("--skip-table", default="", help="Override the skip list table name")
    parser.add_argument("--create-missing", action="store_true", help="Create either table when it does not exist yet")
    return parser.parse_args()


def split_logins(raw: str) -> List[str]:
    """Logins from a space or comma separated list, in order, without repeats."""
    out: List[str] = []
    for part in raw.replace(",", " ").split():
        if part not in out:
            out.append(part)
    return out


def load_schema(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_table(token: str, app_token: str, schema: Dict, name_override: str, create: bool) -> Optional[str]:
    """table_id of the table this schema describes, creating it if asked.

    Returns None when the table does not exist and creating was not asked for,
    which the caller treats as "no data", not as an error: a repository that
    has not set up notifications yet should stay quiet rather than fail runs.
    """
    name = name_override or schema["table_name"]
    table_id = fc.find_table(token, app_token, name)
    if table_id:
        return table_id
    if not create:
        print(f"[WARN] Table '{name}' does not exist. "
              f"Run once with --create-missing to create it.")
        return None
    table_id = fc.create_table(token, app_token, name, schema["fields"])
    print(f"[INFO] Created table '{name}' ({table_id}). Add rows to it to take effect.")
    return table_id


def lookup_login(token: str, app_token: str, table_id: str, login: str) -> Tuple[str, str]:
    """(open_id, name) for a login, or ('', '') when there is nothing to use.

    GitHub treats logins case-insensitively while this table holds whatever
    somebody typed, so the filter is done on an exact match first and then
    re-checked in lower case, which keeps the common path to one API call.
    """
    rows: List[Dict] = fc.search_records(token, app_token, table_id, "github_login", login)
    if not rows:
        wanted = login.strip().lower()
        for row in fc.search_records(token, app_token, table_id, "github_login", wanted):
            rows.append(row)

    for row in rows:
        stored = fc.cell_text(row.get("github_login"))
        if stored.lower() != login.strip().lower():
            continue
        # An unchecked `enabled` pauses a person without losing their mapping.
        # Absent means enabled: a row added before the column existed should
        # still work.
        enabled = row.get("enabled")
        if enabled is not None and not enabled:
            print(f"[INFO] {login} is in the table but disabled, not notifying.")
            return "", fc.cell_text(row.get("name"))
        open_id = fc.cell_text(row.get("open_id"))
        name = fc.cell_text(row.get("name"))
        if not open_id:
            print(f"[WARN] {login} has a row but no open_id, not notifying.")
            return "", name
        return open_id, name

    print(f"[INFO] {login} is not in the contacts table, not notifying. "
          f"Add a row with their open_id to enable it.")
    return "", ""


def check_skip(token: str, app_token: str, table_id: str, workflow: str) -> Tuple[bool, str]:
    """(skip, reason) for a workflow.

    An entry whose `until` has passed does NOT skip: a silenced workflow that
    nobody remembers to re-enable is worse than a noisy one, so the skip
    expires on its own and says so.
    """
    rows = fc.search_records(token, app_token, table_id, "workflow", workflow)
    for row in rows:
        if fc.cell_text(row.get("workflow")) != workflow:
            continue
        reason = fc.cell_text(row.get("reason")) or "no reason given"
        until = row.get("until")
        if not until:
            # Allowed, but never silently: "forever" should stay visible.
            print(f"[WARN] {workflow} is skipped indefinitely ({reason}). "
                  f"Set `until` so the skip expires on its own.")
            return True, reason
        try:
            expires = datetime.fromtimestamp(int(until) / 1000, tz=timezone.utc)
        except (TypeError, ValueError):
            print(f"[WARN] {workflow} has an unreadable `until` ({until!r}), treating as indefinite.")
            return True, reason
        if expires < datetime.now(timezone.utc):
            print(f"[WARN] {workflow} skip expired on {expires.date()} ({reason}). "
                  f"Notifying again -- remove the row or extend `until`.")
            return False, ""
        print(f"[INFO] {workflow} is skipped until {expires.date()} ({reason}).")
        return True, reason
    return False, ""


def emit(**outputs: str) -> None:
    """Write step outputs, when running inside GitHub Actions."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        for key, value in outputs.items():
            print(f"  {key}={value}")
        return
    with open(path, "a", encoding="utf-8") as f:
        for key, value in outputs.items():
            f.write(f"{key}={value}\n")


def main() -> int:
    args = parse_args()

    if not args.login and not args.workflow:
        print("[ERROR] Nothing to look up: pass --login and/or --workflow.")
        return 1

    app_token = args.app_token or os.environ.get("FEISHU_BITABLE_APP_TOKEN", "").strip()
    if not fc.credentials_present() or not app_token:
        # Looking up is not a gate. A repository without credentials simply
        # does not notify, and must not fail runs over it.
        print("[WARN] Feishu credentials or app token missing, cannot look up.")
        emit(**{"open-ids": "", "names": "", "skip": "false", "skip-reason": ""})
        return 0

    try:
        token = fc.get_token()

        skip, skip_reason = False, ""
        if args.workflow:
            table_id = resolve_table(token, app_token, load_schema(SKIP_SCHEMA), args.skip_table, args.create_missing)
            if table_id:
                skip, skip_reason = check_skip(token, app_token, table_id, args.workflow)

        open_ids: List[str] = []
        names: List[str] = []
        # No point resolving anyone for a notification that will not be sent.
        if args.login and not skip:
            table_id = resolve_table(token, app_token, load_schema(CONTACTS_SCHEMA), args.contacts_table,
                                     args.create_missing)
            if table_id:
                # Resolved ids are de-duplicated rather than the logins: two
                # logins can be one person (somebody merging their own PR),
                # and one inbox should get one message.
                for login in split_logins(args.login):
                    open_id, name = lookup_login(token, app_token, table_id, login)
                    if open_id and open_id not in open_ids:
                        open_ids.append(open_id)
                        names.append(name or login)
    except fc.FeishuError as e:
        print(f"[WARN] Lookup failed: {e}")
        emit(**{"open-ids": "", "names": "", "skip": "false", "skip-reason": ""})
        return 0

    if open_ids:
        print(f"[INFO] Notifying: {', '.join(names)}")
    emit(
        **{
            "open-ids": ",".join(open_ids),
            "names": ", ".join(names),
            "skip": "true" if skip else "false",
            "skip-reason": skip_reason,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
