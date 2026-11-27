#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Read the orchestrator's Markdown dashboard; never create management state."""
import argparse
import hashlib
import json
from pathlib import Path
import re

COLUMNS = ('ID', 'Time (UTC, recorded/completed)', 'Status', 'Workflow steps', 'Work description', 'Owner', 'Evidence',
           'Prerequisites / next step')
FIELDS = ('id', 'time_utc', 'status', 'flow_steps', 'description', 'owner', 'evidence', 'next_step')


def split_cells(line):
    """Split actual table delimiters, retaining escaped pipes within cells."""
    text = line.strip()
    cells, cell = [], []
    escaped = False
    for character in text:
        if character == '|' and not escaped:
            cells.append(''.join(cell).strip())
            cell = []
        else:
            cell.append(character)
        escaped = character == '\\' and not escaped
    cells.append(''.join(cell).strip())
    if text.startswith('|'):
        cells.pop(0)
    if text.endswith('|') and not escaped and cells[-1] == '':
        cells.pop()
    return cells


def parse(text):
    """Return source table/rows with raw cells and readable field values.

    Status, time, workflow steps, and Markdown links remain source values.
    raw_cells retains exact escapes; cells/fields decode only table escapes.
    The returned object is a derived view, never another flow authority.
    """
    if not isinstance(text, str):
        raise ValueError('dashboard text must be UTF-8 text')
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if tuple(split_cells(line)) == COLUMNS]
    if not starts:
        raise ValueError('dashboard management table header is missing')
    rows, identifiers, tables = [], set(), []
    for table_index, start in enumerate(starts):
        if start + 1 >= len(lines):
            raise ValueError('dashboard table separator is invalid')
        separator = split_cells(lines[start + 1])
        if len(separator) != len(COLUMNS) or not all(re.fullmatch(r':?-{3,}:?', item) for item in separator):
            raise ValueError('dashboard table separator is invalid')
        headings = []
        for line in lines[:start]:
            heading = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading:
                level = len(heading[1])
                headings = [(n, title) for n, title in headings if n < level] + [(level, heading[2])]
        section = [title for _, title in headings]
        table_rows = []
        end = start + 2
        while end < len(lines) and lines[end].lstrip().startswith('|'):
            raw = split_cells(lines[end])
            if tuple(raw) == COLUMNS:
                break
            if len(raw) != len(COLUMNS):
                raise ValueError(f'dashboard row {end + 1} has {len(raw)} cells, expected {len(COLUMNS)}')
            decoded = [re.sub(r'\\([|\\])', r'\1', cell) for cell in raw]
            row = dict(zip(FIELDS, decoded))
            if not row['id'] or row['id'] in identifiers or not row['status'] or not row['time_utc']:
                raise ValueError(f'dashboard row {end + 1} has a missing/duplicate ID, status or UTC time')
            identifiers.add(row['id'])
            row.update({
                'cells': dict(zip(COLUMNS, decoded)), 'raw_cells': raw, 'line_number': end + 1, 'section': section,
                'table_index': table_index
            })
            rows.append(row)
            table_rows.append(row['id'])
            end += 1
        tables.append({
            'section': section, 'row_ids': table_rows, 'start_line': start + 1, 'end_line': end, 'table':
            '\n'.join(lines[start:end])
        })
    if not rows:
        raise ValueError('dashboard management table has no rows')
    return {
        'schema': 'flagmega.dashboard-table/v1', 'authority': 'source_markdown_only', 'columns': list(COLUMNS), 'rows':
        rows, 'tables': tables, 'table': tables[0]['table'] if len(tables) == 1 else text.strip()
    }


def load(path):
    """Read once so the parsed table and hash bind the same actual bytes."""
    source = Path(path).read_bytes()
    return {**parse(source.decode('utf-8')), 'source_sha256': hashlib.sha256(source).hexdigest()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('path', type=Path)
    args = cli.parse_args()
    print(json.dumps(load(args.path), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
