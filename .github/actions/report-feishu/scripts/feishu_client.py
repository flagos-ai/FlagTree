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
feishu_client.py - Minimal Feishu (Lark) Bitable client

Shared by bootstrap_feishu_table.py and report_feishu.py. Uses only the
standard library, since the CI runners are not guaranteed to have requests
installed and must not need a pip install to report results.

Credentials are read from the environment, never from argv, because argv is
visible in `ps` output on a shared self-hosted runner:

    FEISHU_APP_ID
    FEISHU_APP_SECRET
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Dict, List, Optional

API_BASE = "https://open.feishu.cn/open-apis"

# Bitable field type codes.
#
# NOTE: these are the documented numeric codes; open.feishu.cn is not
# reachable from the development machine, so they were not re-verified against
# the live API. If table creation fails with an invalid-type error, check the
# code here first.
FIELD_TEXT = 1
FIELD_NUMBER = 2
FIELD_SINGLE_SELECT = 3
FIELD_MULTI_SELECT = 4
FIELD_DATETIME = 5
FIELD_CHECKBOX = 7
FIELD_URL = 15

# Feishu rejects writes larger than this per batch.
MAX_BATCH = 500


class FeishuError(RuntimeError):
    """A non-zero code came back from the Feishu API."""


# ============================================================================
# Transport
# ============================================================================


def _request(
    method: str,
    url: str,
    token: Optional[str] = None,
    payload: Optional[Dict] = None,
    timeout: int = 60,
    retries: int = 3,
) -> Dict:
    """Issue one API call, retrying transient failures.

    Retries on 5xx and on connection errors; a 4xx is a request problem and is
    raised immediately so a typo in a field name does not take 3 attempts.
    """
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") not in (0, None):
                raise FeishuError(f"{method} {url} -> code={result.get('code')} "
                                  f"msg={result.get('msg')}")
            return result
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            if e.code < 500:
                raise FeishuError(f"{method} {url} -> HTTP {e.code}: {detail}") from e
            last_error = FeishuError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = e

        if attempt < retries:
            wait = 2**attempt
            print(f"[WARN] {method} {url} failed ({last_error}), retry in {wait}s")
            time.sleep(wait)

    raise FeishuError(f"{method} {url} failed after {retries} attempts: {last_error}")


def get_token() -> str:
    """Exchange app credentials for a tenant_access_token."""
    app_id = os.environ.get("FEISHU_APP_ID", "").strip()
    app_secret = os.environ.get("FEISHU_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise FeishuError("FEISHU_APP_ID / FEISHU_APP_SECRET must be set in the environment")
    result = _request(
        "POST",
        f"{API_BASE}/auth/v3/tenant_access_token/internal",
        payload={"app_id": app_id, "app_secret": app_secret},
    )
    return result["tenant_access_token"]


def credentials_present() -> bool:
    """True when both credentials are set, so callers can skip silently."""
    return bool(os.environ.get("FEISHU_APP_ID", "").strip() and os.environ.get("FEISHU_APP_SECRET", "").strip())


# ============================================================================
# Tables and fields
# ============================================================================


def _tables_url(app_token: str) -> str:
    return f"{API_BASE}/bitable/v1/apps/{app_token}/tables"


def list_tables(token: str, app_token: str) -> List[Dict]:
    """All data tables in a Bitable app."""
    items: List[Dict] = []
    page_token = None
    while True:
        url = f"{_tables_url(app_token)}?page_size=100"
        if page_token:
            url += f"&page_token={page_token}"
        data = _request("GET", url, token).get("data", {})
        items.extend(data.get("items") or [])
        if not data.get("has_more"):
            return items
        page_token = data.get("page_token")


def find_table(token: str, app_token: str, name: str) -> Optional[str]:
    """table_id of the table with this name, or None."""
    for table in list_tables(token, app_token):
        if table.get("name") == name:
            return table.get("table_id")
    return None


def create_table(token: str, app_token: str, name: str, fields: List[Dict]) -> str:
    """Create a data table with an initial field set, returning its table_id."""
    result = _request(
        "POST",
        _tables_url(app_token),
        token,
        payload={"table": {"name": name, "fields": fields}},
    )
    return result["data"]["table_id"]


def list_fields(token: str, app_token: str, table_id: str) -> List[Dict]:
    """All fields of a table, each with field_name and type."""
    items: List[Dict] = []
    page_token = None
    base = f"{_tables_url(app_token)}/{table_id}/fields"
    while True:
        url = f"{base}?page_size=100"
        if page_token:
            url += f"&page_token={page_token}"
        data = _request("GET", url, token).get("data", {})
        items.extend(data.get("items") or [])
        if not data.get("has_more"):
            return items
        page_token = data.get("page_token")


def field_types(token: str, app_token: str, table_id: str) -> Dict[str, int]:
    """Map of field_name -> type code, used to coerce values before writing."""
    return {f["field_name"]: f.get("type") for f in list_fields(token, app_token, table_id) if f.get("field_name")}


def create_field(token: str, app_token: str, table_id: str, field: Dict) -> None:
    """Add one field to an existing table."""
    _request(
        "POST",
        f"{_tables_url(app_token)}/{table_id}/fields",
        token,
        payload=field,
    )


# ============================================================================
# Records
# ============================================================================


def _records_url(app_token: str, table_id: str) -> str:
    return f"{_tables_url(app_token)}/{table_id}/records"


def find_records(token: str, app_token: str, table_id: str, field: str, value: str) -> List[str]:
    """record_ids whose `field` equals `value`.

    Used for idempotent writes: a rerun of the same CI attempt should update
    its row rather than add a duplicate.
    """
    payload = {
        "filter": {
            "conjunction": "and",
            "conditions": [{"field_name": field, "operator": "is", "value": [str(value)]}],
        },
        "field_names": [field],
    }
    url = f"{_records_url(app_token, table_id)}/search?page_size=100"
    data = _request("POST", url, token, payload=payload).get("data", {})
    return [item["record_id"] for item in (data.get("items") or [])]


def create_records(token: str, app_token: str, table_id: str, rows: List[Dict]) -> int:
    """Batch-create rows, chunked to the API limit. Returns rows written."""
    url = f"{_records_url(app_token, table_id)}/batch_create"
    written = 0
    for start in range(0, len(rows), MAX_BATCH):
        chunk = rows[start:start + MAX_BATCH]
        _request(
            "POST",
            url,
            token,
            payload={"records": [{"fields": row} for row in chunk]},
        )
        written += len(chunk)
        if start + MAX_BATCH < len(rows):
            time.sleep(0.5)  # stay under the write rate limit
    return written


def update_record(token: str, app_token: str, table_id: str, record_id: str, row: Dict) -> None:
    """Overwrite the fields of one existing row."""
    _request(
        "PUT",
        f"{_records_url(app_token, table_id)}/{record_id}",
        token,
        payload={"fields": row},
    )


# ============================================================================
# Value coercion
# ============================================================================


def to_millis(value) -> Optional[int]:
    """Convert an ISO 8601 string (or epoch number) to ms since the epoch.

    Feishu date fields take a millisecond timestamp, not a formatted string.
    Naive timestamps are read as local time, matching how ci_run_tests.py
    writes summary.json.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Treat values that are clearly seconds as seconds
        return int(value * 1000) if value < 1e11 else int(value)
    try:
        text = str(value).replace("Z", "+00:00")
        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except (ValueError, TypeError):
        print(f"[WARN] Not a timestamp, dropping: {value!r}")
        return None


def coerce(value, field_type: Optional[int]):
    """Fit a plain Python value to what a Bitable field type accepts.

    Returns None for values that should be omitted; callers drop those keys,
    since several field types reject an explicit null.
    """
    if value is None or value == "":
        return None

    if field_type == FIELD_DATETIME:
        return to_millis(value)

    if field_type == FIELD_NUMBER:
        if isinstance(value, bool):
            return int(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            print(f"[WARN] Not a number, dropping: {value!r}")
            return None

    if field_type == FIELD_CHECKBOX:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    if field_type == FIELD_URL:
        text = str(value)
        return {"link": text, "text": text}

    if field_type == FIELD_MULTI_SELECT:
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return [p.strip() for p in str(value).split(",") if p.strip()]

    # Text, single select and anything unrecognised: send a string, so an
    # unexpected type never crashes the upload.
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def coerce_row(row: Dict, types: Dict[str, int], strict: bool = False) -> Dict:
    """Coerce a flat dict into a Bitable row.

    Keys absent from the table are skipped: the ledger is allowed to carry
    more than the table shows, so adding a field to the record does not break
    reporting before the table catches up.
    """
    out: Dict = {}
    for key, value in row.items():
        if key not in types:
            if strict:
                print(f"[WARN] No such field in table, skipping: {key}")
            continue
        fitted = coerce(value, types[key])
        if fitted is not None:
            out[key] = fitted
    return out
