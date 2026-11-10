#!/usr/bin/env python3
"""Validate an optimization trace against the skill's maintained Markdown assets."""

import argparse
from pathlib import Path
import re
import sys


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "report-templates"


def headings(text, level):
    return re.findall(rf"^{'#' * level} (.+)$", text, re.MULTILINE)


def tables(text):
    """Read ordinary GFM tables, retaining column and metadata row order."""
    result = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip().startswith("|"):
            i += 1
            continue
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append([
                cell.strip()
                for cell in re.split(r"(?<!\\)\|", lines[i].strip())[1:-1]
            ])
            i += 1
        if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", c) for c in rows[1]):
            result.append((rows[0], rows[2:]))
    return result


def section(text, level, title):
    start = re.search(rf"^{'#' * level} {re.escape(title)}$", text, re.MULTILINE)
    if not start:
        return ""
    tail = text[start.end():]
    end = re.search(rf"^#{{1,{level}}} ", tail, re.MULTILINE)
    return tail[:end.start()] if end else tail


def validate(text):
    document = (ASSETS / "optimization-trace.md").read_text()
    entry = (ASSETS / "trace-entry.md").read_text()
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def has_content(body):
        return bool(re.sub(r'<a id="[^"]+"></a>', "", body).strip())

    def required_table(body, columns, scope, fields=None):
        found = tables(body)
        require(len(found) == 1, f"{scope}: expected exactly one required table")
        if len(found) != 1:
            return []
        header, rows = found[0]
        require(header == columns, f"{scope}: column names/order must be {columns!r}")
        require(bool(rows), f"{scope}: table needs at least one row")
        for i, row in enumerate(rows, 1):
            require(len(row) == len(columns), f"{scope}: row {i} has wrong column count")
            require(all(row), f"{scope}: row {i} contains empty cells")
            require(not any(c in {"null", "not_run", "not_applicable", "unavailable", "pending"} for c in row),
                    f"{scope}: row {i} needs a reason for unavailable/unmeasured values")
        if fields is not None:
            require([row[0] for row in rows if row] == fields,
                    f"{scope}: field names/order must be {fields!r}")
        return rows

    require(len(headings(text, 1)) == 1 and headings(text, 1)[0].endswith(" optimization trace"),
            "document: expected one title ending in ' optimization trace'")
    require(headings(text, 2) == headings(document, 2),
            f"document: section names/order must be {headings(document, 2)!r}")
    for name in headings(document, 2):
        require(has_content(section(text, 2, name)), f"document: {name} is empty")
    doc_tables = tables(document)
    required_table(section(text, 2, "Contract and current status"), doc_tables[0][0],
                   "document contract", [row[0] for row in doc_tables[0][1]])
    index = required_table(section(text, 2, "Iteration index"), doc_tables[1][0],
                           "iteration index")
    entry_area = section(text, 2, "Entries")
    matches = list(re.finditer(r"^### ([A-Za-z][A-Za-z0-9_.-]*) — (.+)$", entry_area, re.MULTILINE))
    require(bool(matches), "entries: no entries found")
    require(len(matches) == len(headings(text, 3)),
            "entries: every level-3 heading must use '<entry-id> — <concrete outcome>' inside Entries")
    ids = [m[1] for m in matches]
    require(len(ids) == len(set(ids)), "entries: duplicate entry ID")
    require(len(index) == len(ids), "iteration index: must cover every entry exactly once")
    entry_tables = tables(entry)
    expected_sections = headings(entry, 4)
    expected_meta = [row[0] for row in entry_tables[0][1]]
    require(len(headings(text, 4)) == len(matches) * len(expected_sections),
            "document: level-4 headings are reserved for the required entry sections")
    for i, match in enumerate(matches):
        eid = match[1]
        body = entry_area[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(entry_area)]
        require(headings(body, 4) == expected_sections,
                f"{eid}: section names/order must be {expected_sections!r}")
        preamble = body.split("\n#### ", 1)[0]
        metadata = required_table(preamble, entry_tables[0][0], f"{eid} metadata", expected_meta)
        for name in expected_sections:
            require(has_content(section(body, 4, name)), f"{eid}: {name} is empty")
        required_table(section(body, 4, expected_sections[3]), entry_tables[1][0], f"{eid} gates")
        required_table(section(body, 4, expected_sections[4]), entry_tables[2][0], f"{eid} performance")
        if i < len(index) and len(index[i]) == 7:
            row = index[i]
            require(row[0] == f"[{eid}](#{eid})", f"{eid}: index ID/link or entry order differs")
            require(bool(re.search(rf'<a id="{re.escape(eid)}"></a>', entry_area)),
                    f"{eid}: missing stable anchor for the index link")
            values = {row[0]: row[1] for row in metadata if len(row) == 2}
            require(values.get("Entry / run / UTC", "").startswith(eid + " / "),
                    f"{eid}: metadata entry ID differs")
            require(values.get("Attempt") == row[1],
                    f"{eid}: attempt differs between index and entry")
            require(values.get("Iteration / kind") == row[2] + " / " + row[3],
                    f"{eid}: iteration/kind differs between index and entry")
            require(values.get("Baseline / trial / revision / cohort") == row[4],
                    f"{eid}: baseline/trial/revision/cohort differs between index and entry")
            require(values.get("Status") == row[5], f"{eid}: status differs between index and entry")
            if row[2].isdigit():
                require(bool(re.match(r"^passed[:：]\s*\S", values.get("Numerical qualification", ""))),
                        f"{eid}: an assigned iteration requires an explicit complete numerical pass and receipt")
                if int(row[2]) > 0:
                    require(row[1].isdigit() and int(row[1]) > 0,
                            f"{eid}: a candidate iteration requires its reserved positive attempt index")
            else:
                require(bool(re.match(r"^(not_assigned|not_applicable)[:：]\s*\S", row[2])),
                        f"{eid}: an unassigned iteration needs an explicit qualification reason")

    placeholders = set(re.findall(r"<[^<>]+>", document + entry))
    placeholders.discard('<a id="<entry-id>">')
    placeholders.discard("</a>")
    for placeholder in sorted(placeholders):
        require(placeholder not in text, f"unreplaced template placeholder: {placeholder}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    try:
        errors = validate(args.trace.read_text())
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: {args.trace} conforms to the document and entry templates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
