#!/usr/bin/env python3
"""Render graph timelines, operator/backend bars and revision history as provenance-bound SVG."""
import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import math
import re
import textwrap
from pathlib import Path
import sys

from render_kernel_swimlane import COLORS, EVIDENCE

STATUS_COLORS = {'measured': COLORS['tensor'], 'qualified': COLORS['tensor'],
                 'achieved': COLORS['store'], 'no_gain': COLORS['unknown'],
                 'rejected': COLORS['wait'], 'pending': COLORS['sync']}
WIDTH, LEFT, RIGHT = 1500, 290, 55


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def milestone_gain(data, row):
    if row.get('value') is None:
        return None
    baseline, value = row['baseline_value'], row['value']
    return value / baseline - 1 if data['direction'] == 'higher' else 1 - value / baseline


def compact_history(data):
    return data['kind'] == 'series' and data.get('chart_role') in {'revision_history', 'milestone_history'}


def performance_points(data):
    """One shared curve/table list; only numerical changes create records."""
    selected, last, gaps = [], {}, set()
    for row in sorted(data['rows'], key=lambda item: (item['x'], item['label'])):
        group = (row['label'], json.dumps(row.get('cohort_id', data['cohort']), sort_keys=True))
        if row.get('value') is None or row.get('status') == 'pending':
            gaps.add(group)
            continue
        if last.get(group) == row['value']:
            continue
        point = dict(row)
        point['_gap_before'] = group in gaps
        point['point_id'] = row.get('point_id') or data.get('x_labels', {}).get(str(row['x'])) or f"{row['label']} / {row['x']}"
        selected.append(point)
        last[group] = row['value']
        gaps.discard(group)
    return selected


def revision_table_rows(data, points):
    table = data.get('revision_table')
    if table is None:
        return points
    require(data.get('chart_role') == 'revision_history' and isinstance(table, list) and table,
            'revision table belongs to revision history')
    numerical = {p['point_id']:p for p in points}
    seen, measured = set(), set()
    for row in table:
        rid = row.get('point_id')
        require(nonempty(rid) and rid not in seen, 'revision table IDs must be unique')
        seen.add(rid)
        require(nonempty(row.get('main_improvement')), 'revision table needs a main optimization')
        if row.get('value') is None:
            require(rid not in numerical and nonempty(row.get('model_status')),
                    'unmeasured table row must not hide a numerical point')
            require(not row.get('measurement_ids'), 'unmeasured table row cannot claim model samples')
        else:
            point = numerical.get(rid)
            require(point is not None and row['value'] == point['value'] and
                    row.get('measurement_ids') == point.get('measurement_ids') and
                    row['main_improvement'] == point.get('main_improvement'),
                    'revision table numeric row must match its full-model point')
            measured.add(rid)
    require(measured == set(numerical), 'revision table must cover every full-model point')
    return table


def validate_milestones(data):
    achieved_points = {(row['label'], row['x']) for row in data['rows']
                       if data['kind'] == 'series' and row.get('status') == 'achieved'}
    require(not achieved_points or 'milestones' in data, 'achieved curve points need milestone table entries')
    if data.get('x_axis') == 'milestone':
        require('milestones' in data, 'milestone history needs its improvement table')
    if 'milestones' not in data:
        return
    require(data['kind'] == 'series', 'milestone table belongs to a history series')
    require(data.get('direction') in {'higher', 'lower'}, 'milestone gains need higher/lower direction')
    require(isinstance(data['milestones'], list), 'milestones must be a list')
    points = {(row['label'], row['x']): row for row in data['rows']}
    seen, table_points, table_achieved = set(), set(), set()
    for item in data['milestones']:
        milestone = item.get('milestone_id')
        require(isinstance(milestone, str) and re.fullmatch(r'm[0-9]{3,}', milestone) and milestone not in seen, 'unique mNNN milestone ID required')
        seen.add(milestone)
        for key in ('series', 'baseline_id', 'main_improvement'):
            require(nonempty(item.get(key)), f'milestone {key} required')
        require(item.get('status') in {'achieved', 'open', 'pending', 'no_gain', 'rejected', 'improved'}, 'invalid milestone status')
        require(type(item.get('point_x')) is int and type(item.get('baseline_point_x')) is int and
                0 <= item['baseline_point_x'] < item['point_x'], 'milestone baseline must precede the candidate point')
        key = (item['series'], item['point_x'])
        require(key not in table_points, 'each milestone must have a distinct plotted point')
        table_points.add(key)
        point = points.get(key)
        baseline = points.get((item['series'], item.get('baseline_point_x')))
        require(point is not None and baseline is not None, 'milestone/baseline must reference plotted points')
        require((item['status'] == 'achieved') == (point.get('status') == 'achieved'), 'milestone table/curve achievement states disagree')
        require(number(item.get('baseline_value')) and item['baseline_value'] > 0 and baseline.get('value') == item['baseline_value'], 'milestone baseline disagrees with curve or is not positive')
        require(item.get('value') == point.get('value'), 'milestone value disagrees with plotted point')
        if item.get('value') is None:
            require(item['status'] != 'achieved' and nonempty(item.get('reason')), 'missing milestone performance cannot be achieved')
            continue
        gain = milestone_gain(data, item)
        if item['status'] == 'achieved':
            table_achieved.add(key)
            require(gain > 0 and point.get('status') == 'achieved', 'achieved milestone needs a positive gain and matching achieved point')
            require(isinstance(item.get('revision_id'), str) and re.fullmatch(r'r[0-9]{3,}', item['revision_id']), 'achieved milestone needs winning revision')
        if 'gain' in item:
            require(number(item['gain']) and math.isclose(item['gain'], gain, rel_tol=1e-9, abs_tol=1e-12), 'supplied gain disagrees with metric values')
    require(achieved_points == table_achieved, 'every achieved curve point needs its matching achieved milestone entry')


def validate(data, input_path):
    require(data.get('schema_version') == 1, 'schema_version must be 1')
    require(type(data.get('synthetic')) is bool, 'synthetic must be explicit')
    require(data.get('kind') in {'series', 'bar', 'timeline'}, 'kind must be series/bar/timeline')
    for key in ('title', 'unit', 'cohort'):
        require(nonempty(data.get(key)), f'{key} is required')
    binding = data.get('template_binding')
    if binding is not None:
        template = Path(binding['path']).resolve()
        require(template.parent.name == 'workspace-figures' and
                template.name == binding['template_id'] + '.json', 'canonical template identity mismatch')
        require(sha(template) == binding['sha256'], 'canonical template source changed')
        definition = json.loads(template.read_text())
        require(data['kind'] == definition['kind'] and data.get('chart_role') == definition.get('chart_role'),
                'figure kind/role differs from the bound canonical template')
        require(binding['renderer_sha256'] == sha(Path(__file__)), 'maintained renderer source changed')
    provenance = data.get('provenance')
    require(isinstance(provenance, list), 'provenance must be a list')
    if not data['synthetic']:
        require(provenance, 'actual figures need raw/audited-summary pins')
        for item in provenance:
            path = (input_path.parent / item['path']).resolve()
            require(path.is_file() and sha(path) == item['sha256'], 'provenance file changed or is missing')
    rows = data.get('rows')
    require(isinstance(rows, list) and rows, 'rows must be nonempty; empty templates are not measurements')
    if data.get('pending_only'):
        require(not data['synthetic'], 'pending working panels must describe actual evidence state')
        require(nonempty(data.get('pending_reason')), 'pending panel needs its actual missing-evidence reason')
        require(not data.get('window_ns') and not data.get('timebase'), 'pending panel cannot fabricate a clock or execution window')
        require(all(nonempty(row.get('label')) and nonempty(row.get('reason')) and
                    row.get('value') is None and not row.get('intervals') for row in rows),
                'pending panel cannot contain measurements or intervals')
        require(not data.get('milestones'), 'pending-only panels cannot claim achieved milestones')
        return
    if data['kind'] == 'timeline':
        recorded_callers = data.get('scope') == 'observed_TIR_callers'
        require(data['unit'] == ('raw globaltimer units' if recorded_callers else 'ns'), 'timeline unit must match its admitted clock scope')
        require(nonempty(data.get('invocation')), 'timeline invocation is required')
        clock = data.get('timebase', {})
        recorded_leaders = data.get('scope') == 'observed_role_leaders'
        if recorded_callers:
            require(not data['synthetic'] and not data.get('window_ns'), 'observed callers cannot claim the whole model window')
            capture = data.get('caller_capture', {})
            require(capture.get('health_validated') is True and capture.get('observer') == 'physical_warp_caller' and
                    type(capture.get('cta')) is int and capture['cta'] >= 0 and type(capture.get('SM')) is int, 'actual caller CTA/SM attribution required')
            require(capture.get('observed_engine_busy') is False and capture.get('full_model_window') is False, 'caller envelopes cannot claim engine busy or full model time')
            for key in ('source_pin', 'clock_pin', 'dataset_pin'):
                pin = capture.get(key, {})
                require(any(pin.get('path') == item['path'] and pin.get('sha256') == item['sha256'] for item in provenance), 'callers need pinned source, clock and admitted dataset')
            require(nonempty(clock.get('source')) and clock.get('actual_resolution_ns') is None and clock.get('host_alignment') is None and
                    clock.get('alignment') == 'pending', 'raw caller clock calibration must remain unknown')
            window = data.get('observed_window_raw_units')
            require(isinstance(window, list) and len(window) == 2 and all(type(t) is int for t in window) and 0 <= window[0] < window[1], 'callers need the actual emitted raw-unit window')
        elif recorded_leaders:
            require(not data['synthetic'] and not data.get('window_ns'), 'recorded leaders cannot claim a full model window')
            capture = data.get('leader_capture', {})
            require(capture.get('health_validated') is True and capture.get('observer') == 'role_local_leader' and
                    type(capture.get('cta')) is int and capture['cta'] >= 0, 'recorded leader health/CTA attribution required')
            require(capture.get('observed_engine_busy') is False and capture.get('full_model_window') is False,
                    'leader envelopes cannot claim engine busy or full model time')
            for key in ('source_pin', 'clock_pin'):
                pin = capture.get(key, {})
                require(any(pin.get('path') == item['path'] and pin.get('sha256') == item['sha256'] for item in provenance),
                        'recorded leaders need pinned source and nominal clock documentation')
            require(nonempty(clock.get('source')) and clock.get('nominal_ns_verified') is True and
                    (clock.get('resolution_ns') is None or type(clock.get('resolution_ns')) is int and clock['resolution_ns'] > 0),
                    'recorded leaders need documented nominal ns; unknown resolution stays null')
            require(clock.get('alignment') in {'pending', 'validated_common'}, 'invalid leader clock alignment')
            if clock.get('alignment') == 'validated_common':
                require(type(clock.get('resolution_ns')) is int and clock['resolution_ns'] > 0 and
                        capture.get('calibration_from_full_capture') is True,
                        'calibrated leader detail needs measured resolution and full-capture calibration evidence')
            window = data.get('observed_window_ns')
            require(isinstance(window, list) and len(window) == 2 and all(type(t) is int for t in window) and 0 <= window[0] < window[1],
                    'recorded leaders need their emitted observed window')
        else:
            require(nonempty(clock.get('source')) and type(clock.get('resolution_ns')) is int and clock['resolution_ns'] > 0, 'timeline needs a real clock source/resolution')
            require(clock.get('alignment') == ('synthetic' if data['synthetic'] else 'validated_common'), 'timeline lanes require a shared validated timebase')
            window = data.get('window_ns')
            require(isinstance(window, list) and len(window) == 2 and all(type(t) is int for t in window) and 0 <= window[0] < window[1], 'timeline needs the full model window_ns')
            require(data.get('scope') == 'whole_model', 'graph timeline must declare whole_model scope')
        ids = set()
        for row in rows:
            require(nonempty(row.get('label')), 'timeline lane label required')
            require(row.get('status') in {'captured', 'unavailable', 'not_used', 'not_applicable'}, 'invalid timeline lane status')
            require(isinstance(row.get('intervals'), list), 'intervals list required')
            require(row['status'] == 'captured' or not row['intervals'], 'unavailable lanes cannot contain intervals')
            for interval in row['intervals']:
                require(nonempty(interval.get('id')) and interval['id'] not in ids, 'interval IDs must be unique')
                ids.add(interval['id'])
                start_key, end_key = ('start_raw_units', 'end_raw_units') if recorded_callers else ('start_ns', 'end_ns')
                begin, end = interval.get(start_key), interval.get(end_key)
                require(type(begin) is int and type(end) is int and 0 <= begin <= end, 'intervals need ordered integer clock values')
                require(window[0] <= begin <= end <= window[1], 'operator interval leaves its observed window')
                if recorded_callers:
                    require(interval.get('cta') == capture['cta'] and interval.get('SM') == capture['SM'] and
                            interval.get('role') in {'consumer', 'producer'} and type(interval.get('warp')) is int and
                            interval['evidence'] == 'inflight_envelope', 'caller lanes require one actual CTA/SM and physical role envelopes')
                require(interval.get('kind') in COLORS and interval.get('evidence') in EVIDENCE, 'invalid interval kind/evidence')
                require(not data['synthetic'] or interval['evidence'] not in {'observed_activity', 'sampled'}, 'synthetic events cannot claim observed activity')
                if recorded_leaders:
                    require(interval.get('cta') == capture['cta'] and interval.get('role') in {'consumer', 'producer'} and
                            interval['evidence'] == 'inflight_envelope', 'recorded leader lanes need one actual CTA and role-local envelopes')
    else:
        require(nonempty(data.get('metric')), 'metric is required')
        if data['kind'] == 'bar':
            require(data.get('sort_by', 'input') in {'input', 'value_desc', 'value_asc'}, 'invalid bar sort order')
            if data.get('chart_role') == 'graph_operator_latency':
                require(data.get('sort_by') == 'value_desc' and nonempty(data.get('aggregation')), 'graph latency bars need descending time and declared aggregation')
        if data['kind'] == 'series':
            require(data.get('x_axis') in {'attempt', 'revision', 'iteration', 'milestone'}, 'series needs an explicit attempt/revision/iteration/milestone axis')
        keys = set()
        for row in rows:
            require(nonempty(row.get('label')), 'row label required')
            if data['kind'] == 'series':
                require(type(row.get('x')) is int and row['x'] >= 0, 'series x must be a nonnegative index')
                key = (row['label'], row['x'])
                require(key not in keys, 'duplicate series/index')
                keys.add(key)
            value = row.get('value')
            if value is None:
                require(nonempty(row.get('reason')), 'missing data needs a reason')
                continue
            require(number(value) and value >= 0, 'values must be finite and nonnegative')
            require(row.get('status') in STATUS_COLORS, 'numeric points need an actual status')
            if 'main_improvement' in row:
                require(nonempty(row['main_improvement']), 'point main improvement must be nonempty')
            require(data['synthetic'] or row.get('measurement_ids'), 'actual values need measurement IDs')
            if 'low' in row or 'high' in row:
                require(number(row.get('low')) and number(row.get('high')) and 0 <= row['low'] <= value <= row['high'], 'invalid uncertainty interval')
                require(nonempty(data.get('uncertainty_method')), 'uncertainty intervals need the declared method')

    validate_milestones(data)
    if 'revision_table' in data:
        revision_table_rows(data, performance_points(data))


def render(data):
    parts = []
    def text(x, y, content, size=13, fill='#334155', anchor='start'):
        parts.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{escape(str(content))}</text>')
    def line(x1, y1, x2, y2, color='#cbd5e1', width=1, extra=''):
        parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" {extra}/>')
    def rect(x, y, w, h, fill, extra='', title=None):
        attributes = f'x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" {extra}'
        parts.append(f'<rect {attributes}><title>{escape(title)}</title></rect>' if title else f'<rect {attributes}/>')
    title = data['title'] + (' [SYNTHETIC EXAMPLE]' if data['synthetic'] else '')
    text(24, 32, title, 21, '#0f172a')
    text(24, 55, data['cohort'])
    text(24, 76, data.get('aggregation', data.get('uncertainty_method', 'No uncertainty interval supplied')))
    compact = compact_history(data)
    points = performance_points(data) if compact else []
    rows = points if compact and not data.get('pending_only') else data['rows']
    plot_width = WIDTH - LEFT - RIGHT
    if data.get('pending_only'):
        height = 190 + 54 * len(rows)
        text(LEFT, 101, 'Capture pending — no measured axis or synthetic timeline', 15, '#475569')
        for index, row in enumerate(rows):
            py = 119 + index * 54
            text(LEFT - 14, py + 18, row['label'], anchor='end')
            rect(LEFT, py, plot_width, 40, '#f1f5f9')
            for line_index, value in enumerate(textwrap.wrap(row['reason'], width=140)[:2]):
                text(LEFT + 8, py + 16 + line_index * 16, value, 12)
        text(24, height - 28, data['pending_reason'], 12)
    elif data['kind'] == 'series':
        height, bottom, top = 490, 380, 110
        valid = [row for row in rows if row.get('value') is not None]
        require(valid, 'series needs at least one measured point')
        xs = [row['x'] for row in (data['rows'] if compact else rows)]
        xmax = max(1, max(xs))
        ymin = min(row.get('low', row['value']) for row in valid)
        ymax = max(row.get('high', row['value']) for row in valid)
        padding = max((ymax - ymin) * .15, abs(ymax) * .03, 1e-9)
        ymin, ymax = max(0, ymin - padding), ymax + padding
        x = lambda v: LEFT + v / xmax * plot_width
        y = lambda v: bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
        for tick in range(5):
            value = ymin + (ymax - ymin) * tick / 4
            line(LEFT, y(value), WIDTH - RIGHT, y(value))
            text(LEFT - 12, y(value) + 4, f'{value:.4g}', anchor='end')
        text(LEFT, 96, f"{data['metric']} ({data['unit']})")
        measured_x = {row['x'] for row in rows if row.get('value') is not None}
        for index in sorted(set(xs)):
            labels = data.get('x_labels', {})
            text(x(index), bottom + 24, labels.get(str(index), index), 13,
                 COLORS['unknown'] if index not in measured_x else '#334155', anchor='middle')
        text((LEFT + WIDTH - RIGHT) / 2, bottom + 46, data['x_axis'], anchor='middle')
        groups = defaultdict(list)
        for row in (data['rows'] if compact else rows):
            groups[row['label']].append(row)
        for label, group in groups.items():
            previous, gap = None, False
            for row in sorted(group, key=lambda item: item['x']):
                if row.get('value') is None:
                    gap = True
                    marker_x = x(row['x']) - 3
                    why = row.get('reason') or 'no measurement for this revision'
                    parts.append(f'<rect class="no-measurement-marker" x="{marker_x:.2f}" y="{bottom + 8:.2f}" '
                                 f'width="6" height="6" fill="none" stroke="{COLORS["unknown"]}">'
                                 f'<title>{escape(label)} / {escape(row.get("point_id") or row["x"])}: '
                                 f'{escape(why)}</title></rect>')
                    continue
                px, py = x(row['x']), y(row['value'])
                color = STATUS_COLORS[row['status']]
                if previous:
                    line(x(previous['x']), y(previous['value']), px, py, COLORS['tensor'], 1.5,
                         'stroke-dasharray="6 4"' if gap else '')
                gap = False
                if 'low' in row:
                    line(px, y(row['low']), px, y(row['high']), color, 1.5)
                    for bound in ('low', 'high'):
                        line(px - 5, y(row[bound]), px + 5, y(row[bound]), color)
                parts.append(f'<circle class="performance-point-marker" cx="{px:.2f}" cy="{py:.2f}" r="5" fill="{color}"><title>{escape(label)}: {row["value"]} {escape(data["unit"])}; {escape(row["status"])}</title></circle>')
                if row['status'] == 'achieved':
                    parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="9" fill="none" stroke="{color}"/>')
                # Keep the final series label inside the SVG view box.
                label_at_right = px > WIDTH - RIGHT - 180
                if len(groups) > 1:
                    text(px - 8 if label_at_right else px + 8,
                         py + 18 if label_at_right else py - 9, label, 11,
                         anchor='end' if label_at_right else 'start')
                previous = row
        caption = (data.get('caption') if nonempty(data.get('caption')) else None) or \
            'Points: measured revisions; bars: 95% CI. Open squares: no measurement; dashed segments bridge them.'
        text(24, 456, caption, 12)
    elif data['kind'] == 'bar':
        sort = data.get('sort_by', 'input')
        if sort != 'input':
            rows = sorted(rows, key=lambda row: (row.get('value') is None,
                          -row['value'] if row.get('value') is not None and sort == 'value_desc' else row.get('value', 0) or 0))
        maximum = max([row.get('high', row['value']) for row in rows if row.get('value') is not None] or [1]) or 1
        height = 145 + 39 * len(rows)
        text(LEFT, 96, f"{data['metric']} ({data['unit']})")
        for index, row in enumerate(rows):
            py = 113 + index * 39
            text(LEFT - 14, py + 16, row['label'], anchor='end')
            rect(LEFT, py, plot_width, 24, '#f1f5f9')
            if row.get('value') is None:
                text(LEFT + 8, py + 17, 'Unavailable: ' + row['reason'])
                continue
            value = row['value']
            rect(LEFT, py, value / maximum * (plot_width - 140), 24, STATUS_COLORS[row['status']])
            if 'low' in row:
                line(LEFT + row['low'] / maximum * (plot_width - 140), py + 12,
                     LEFT + row['high'] / maximum * (plot_width - 140), py + 12, '#0f172a', 2)
            text(LEFT + value / maximum * (plot_width - 140) + 8, py + 17, f'{value:.4g} {data["unit"]} / {row["status"]}', 11)
    else:
        intervals = [event for row in rows for event in row['intervals']]
        require(intervals, 'timeline needs real intervals; missing lanes are not idle')
        recorded_leaders = data.get('scope') == 'observed_role_leaders'
        recorded_callers = data.get('scope') == 'observed_TIR_callers'
        origin, end = data['observed_window_raw_units' if recorded_callers else 'observed_window_ns' if recorded_leaders else 'window_ns']
        start_key, end_key = ('start_raw_units', 'end_raw_units') if recorded_callers else ('start_ns', 'end_ns')
        clock_unit = 'raw globaltimer units' if recorded_callers else 'ns'
        span = max(1, end - origin)
        x = lambda value: LEFT + (value - origin) / span * plot_width
        if recorded_leaders and data['timebase'].get('alignment') == 'validated_common':
            clock_label = (f"{data['invocation']}; CTA{data['leader_capture']['cta']}; common PTX globaltimer; "
                           f"{data['timebase']['resolution_ns']} ns observed granularity; host bounds in clock receipt")
        else:
            clock_label = (f"{data['invocation']}; CTA{data['caller_capture']['cta']}/SM{data['caller_capture']['SM']}; raw clock, resolution/calibration pending" if recorded_callers else f"{data['invocation']}; CTA{data['leader_capture']['cta']} nominal globaltimer ns; resolution/calibration pending" if recorded_leaders
                           else f"{data['invocation']}; common clock; relative ns from {origin}")
        text(LEFT, 96, clock_label)
        py = 110
        for row in rows:
            tracks = []
            placed = []
            for event in sorted(row['intervals'], key=lambda item: (item[start_key], item[end_key])):
                slot = next((i for i, endpoint in enumerate(tracks) if endpoint <= event[start_key]), len(tracks))
                if slot == len(tracks):
                    tracks.append(event[end_key])
                else:
                    tracks[slot] = event[end_key]
                placed.append((event, slot))
            row_height = max(1, len(tracks)) * 24 + 8
            rect(20, py, WIDTH - 40, row_height, '#f8fafc')
            text(LEFT - 12, py + 18, row['label'], anchor='end')
            if not placed:
                text(LEFT + 8, py + 18, row['status'])
            for event, slot in placed:
                start, stop = x(event[start_key]), x(event[end_key])
                extra = {
                    'observed_activity': '', 'inflight_envelope': 'fill-opacity=".08" stroke="#475569" stroke-dasharray="5 3"',
                    'sampled': 'fill-opacity=".45" stroke="#475569"',
                    'inferred': 'fill-opacity=".35" stroke="#64748b" stroke-dasharray="2 3"',
                    'unknown': 'fill-opacity=".2" stroke="#64748b"'}[event['evidence']]
                tooltip = f"{event['id']}: {event[start_key]}..{event[end_key]} {clock_unit}; {event['evidence']}"
                if recorded_callers:
                    tooltip = f"{event.get('callsite', event['id'])} / layer {event.get('layer')} / {event.get('role')}: {event[start_key]}..{event[end_key]} {clock_unit}; {event['evidence']}; TIR {event.get('source_link')}"
                    link = event.get('source_link', {})
                    if link.get('path') and type(link.get('line')) is int:
                        parts.append('<a href="' + escape(str(link['path']) + '#L' + str(link['line']), quote=True) + '">')
                if stop == start:
                    line(start, py + slot * 24 + 3, start, py + slot * 24 + 21, COLORS[event['kind']], 1)
                else:
                    rect(start, py + slot * 24 + 3, stop - start, 18, COLORS[event['kind']], extra, tooltip)
                    if event['evidence'] in {'sampled', 'unknown'}:
                        rect(start, py + slot * 24 + 3, stop - start, 18, f"url(#{event['evidence']}-pattern)")
                if recorded_callers and event.get('source_link', {}).get('path') and type(event['source_link'].get('line')) is int:
                    parts.append('</a>')
            py += row_height + 6
        height = py + 65
        for tick in range(5):
            offset = span * tick / 4
            text(LEFT + plot_width * tick / 4, py + 19, f'{offset:.0f}', anchor='middle')
        text(24, py + 46, (f'Observed TIR caller window: {end-origin} raw units; one CTA/SM, inclusive overlapping envelopes. No cross-SM/host alignment or engine busy; full315 pending.' if recorded_callers else f'Observed role-leader window: {end - origin} nominal ns; unobserved prefix/tail and D2H excluded. No engine busy/full-warp/whole-model claim.' if recorded_leaders
                          else f'Full model window: {end - origin} ns. Missing attribution does not imply idle time; evidence styles distinguish observations/envelopes/inference.'), 12)
    if compact and points:
        optimization_column = data.get('chart_role') == 'revision_history'
        revision_coverage = 'revision_table' in data
        table_rows = revision_table_rows(data, points) if revision_coverage else points
        value_x = 230 if optimization_column else 780
        summary_x = 650 if revision_coverage else 420
        py = height + 5
        rect(24, py, WIDTH - 48, 31, '#e2e8f0')
        text(34, py + 21, 'Revision' if revision_coverage else 'Performance point', 13, '#0f172a')
        text(value_x, py + 21, 'Decode TPS / status' if revision_coverage else
             'Decode TPS' if data['unit'] == 'tokens/s' else f"{data['metric']} ({data['unit']})", 13, '#0f172a')
        if optimization_column:
            text(summary_x, py + 21, 'Main optimization', 13, '#0f172a')
        py += 31
        for point in table_rows:
            summary = point.get('main_improvement', 'optimization not recorded')
            summaries = textwrap.wrap(summary, width=64) if revision_coverage else [summary]
            row_height = max(32, 16 + 16*len(summaries))
            parts.append(f'<g class="performance-point-table-row" data-point-id="{escape(str(point["point_id"]), quote=True)}" data-model-measured="{str(point.get("value") is not None).lower()}">')
            rect(24, py, WIDTH - 48, row_height, '#f8fafc')
            text(34, py + 22, point['point_id'], 12)
            text(value_x, py + 22, f"{point['value']:.6f}" if point.get('value') is not None else point['model_status'], 12)
            if optimization_column:
                for index, summary_line in enumerate(summaries):
                    text(summary_x, py + 22 + index*16, summary_line, 12)
            parts.append('</g>')
            py += row_height
        height = py + 24
    if 'milestones' in data and not compact:
        top = height + 5
        text(24, top + 18, 'Milestone improvements', 18, '#0f172a')
        text(24, top + 39, f"Gain uses the listed baseline; {data['metric']} ({data['unit']}), {data['direction']}-is-better. Positive delta alone does not establish achievement.", 12)
        columns = [24, 215, 805, 1150, 1320, WIDTH - 24]
        py = top + 52
        rect(24, py, WIDTH - 48, 31, '#e2e8f0')
        for column, heading in zip(columns, ['Milestone / revision', 'Main improvement', 'Performance / baseline', 'Gain', 'Status']):
            text(column + 10, py + 21, heading, 12, '#0f172a')
        py += 31
        entries = data['milestones']
        if not entries:
            text(34, py + 24, 'No milestone entries yet.', 12)
            py += 36
        for item in entries:
            improvement = textwrap.wrap(item['main_improvement'], width=78) or ['']
            gain = milestone_gain(data, item)
            performance = f"{item['baseline_value']:.6g} -> {item['value']:.6g} {data['unit']}" if gain is not None else f"{item['baseline_value']:.6g} -> unavailable"
            baseline = textwrap.wrap('vs ' + item['baseline_id'], width=39)
            first = [item['milestone_id'], item.get('revision_id') or 'not selected']
            values = [first, improvement, [performance, *baseline],
                      [f'{gain:+.2%}' if gain is not None else 'unavailable'], [item['status']]]
            row_height = max(len(cell) for cell in values) * 18 + 16
            rect(24, py, WIDTH - 48, row_height, '#f8fafc')
            for column, cell in zip(columns, values):
                for index, value in enumerate(cell):
                    text(column + 10, py + 22 + index * 18, value, 12)
            line(24, py + row_height, WIDTH - 24, py + row_height)
            py += row_height
        height = py + 24
    working = data.get('working_status')
    if working and not compact:
        text(24, height + 18, 'Actual working source / validation', 16, '#0f172a')
        for index, value in enumerate(working.get('visible_context', [])):
            text(24, height + 42 + index * 18, value, 11)
        height += 64 + 18 * len(working.get('visible_context', []))
    attempts = data.get('attempts_table', [])
    if attempts and not compact:
        text(24, height + 18, 'Actual attempt states — failures/pending are not zero-performance points', 16)
        py = height + 30
        for item in attempts:
            text(34, py + 18, f"{item['id']} | {item.get('source_id', 'source unavailable')} | {item['status']} | exit {item.get('exit_code', 'unknown')}", 11)
            py += 26
        height = py + 24
    metadata = escape(json.dumps({'schema_version': 1, 'synthetic': data['synthetic'],
                                 'provenance': data['provenance'], 'template_binding': data.get('template_binding'),
                                 'working_status': None if compact else data.get('working_status'),
                                 'performance_points': points if compact else None}))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}"><metadata>{metadata}</metadata><rect width="100%" height="100%" fill="white"/><defs><pattern id="sampled-pattern" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0,6 L6,0" stroke="#334155" stroke-opacity=".4"/></pattern><pattern id="unknown-pattern" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0,0 L8,8 M0,8 L8,0" stroke="#64748b" stroke-opacity=".4"/></pattern></defs><g font-family="Inter,DejaVu Sans,sans-serif">' + ''.join(parts) + '</g></svg>\n'
    return svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.input.resolve() != args.output.resolve(), 'output cannot overwrite input data')
        data = json.loads(args.input.read_text())
        validate(data, args.input)
        svg = render(data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg)
        print(args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    sys.exit(main())
