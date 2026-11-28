#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Render normalized kernel intervals using the FlagMega swimlane style (stdlib only)."""

import argparse
from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
import sys
import textwrap

COLORS = {
    "index": "#f0e442",
    "transfer": "#56b4e9",
    "tensor": "#0072b2",
    "simt": "#e69f00",
    "wait": "#d55e00",
    "sync": "#cc79a7",
    "store": "#009e73",
    "unknown": "#9ca3af",
}
EVIDENCE = ("observed_activity", "inflight_envelope", "sampled", "inferred", "unknown")
WORK = {"index", "transfer", "tensor", "simt", "store"}
STATUSES = {"captured", "unavailable", "not_used", "not_applicable"}
CAUSES = {"starvation", "backpressure", "buffer_reuse", "synchronization", "unknown"}
MARGIN, LEFT, RIGHT, ROW, BAR = 20, 260, 30, 24, 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate(data):
    require(isinstance(data, dict), "input must be an object")
    require(integer(data.get("schema_version")) and data["schema_version"] == 1, "schema_version must be 1")
    require(nonempty(data.get("title")), "title is required")
    require(type(data.get("synthetic")) is bool, "synthetic must be explicitly true or false")
    window = data.get("window_ns")
    require(
        isinstance(window, list) and len(window) == 2 and all(integer(t) for t in window)
        and 0 <= window[0] < window[1], "window_ns must be two increasing nonnegative integer timestamps")
    require(isinstance(data.get("panels"), list) and data["panels"], "panels must be nonempty")
    for panel in data["panels"]:
        require(isinstance(panel, dict), "each panel must be an object")
        for field in ("label", "invocation", "scope"):
            require(nonempty(panel.get(field)), f"panel {field} is required")
        clock = panel.get("timebase", {})
        require(
            isinstance(clock, dict) and nonempty(clock.get("source")) and integer(clock.get("resolution_ns"))
            and clock["resolution_ns"] > 0, "timebase needs source and positive integer resolution_ns")
        allowed = {"synthetic"} if data["synthetic"] else {"validated_common", "wg_only"}
        require(clock.get("alignment") in allowed, f"timebase alignment must be one of {sorted(allowed)}")
        require(isinstance(panel.get("lanes"), list) and panel["lanes"], "lanes must be nonempty")
        lane_ids, events = set(), {}
        for lane in panel["lanes"]:
            require(isinstance(lane, dict), "each lane must be an object")
            require(nonempty(lane.get("id")) and lane["id"] not in lane_ids, "lane IDs must be unique")
            lane_ids.add(lane["id"])
            require(nonempty(lane.get("label")) and nonempty(lane.get("scope")), "lane label/scope required")
            require(lane.get("kind") in {"wg", "hardware"}, "lane kind must be wg or hardware")
            require(lane.get("status") in STATUSES, "invalid lane status")
            intervals = lane.get("intervals")
            require(isinstance(intervals, list), "each lane requires an intervals list")
            require(lane["status"] == "captured" or not intervals, "unavailable/unused lanes cannot have intervals")
            if intervals and lane["kind"] == "hardware" and not data["synthetic"]:
                require(clock["alignment"] == "validated_common",
                        "hardware/WG axis requires validated_common alignment")
            for event in intervals:
                require(isinstance(event, dict), "each interval must be an object")
                require(
                    nonempty(event.get("id")) and event["id"] not in events,
                    "interval IDs must be unique within a panel")
                events[event["id"]] = event
                require(nonempty(event.get("label")), "interval label is required")
                require(
                    event.get("kind") in COLORS and event.get("evidence") in EVIDENCE, "invalid interval kind/evidence")
                require(
                    all(integer(event.get(k)) for k in ("start_ns", "end_ns"))
                    and 0 <= event["start_ns"] <= event["end_ns"], "interval times must be ordered relative integer ns")
                if data["synthetic"]:
                    require(event["evidence"] not in {"observed_activity", "sampled"},
                            "synthetic intervals must not claim measured activity/samples")
        for phase in panel.get("phases", []):
            require(isinstance(phase, dict) and nonempty(phase.get("label")), "phase label required")
            require(
                all(integer(phase.get(k)) for k in ("start_ns", "end_ns")) and 0 <= phase["start_ns"] < phase["end_ns"],
                "phase times must be ordered relative integer ns")
        for dependency in panel.get("dependencies", []):
            require(
                isinstance(dependency, dict) and dependency.get("from") in events and dependency.get("to") in events,
                "dependency must reference existing interval IDs")
        for bubble in panel.get("bubbles", []):
            require(isinstance(bubble, dict) and bubble.get("interval") in events, "bubble must reference an interval")
            event = events[bubble["interval"]]
            require(event["kind"] == "wait" and event["end_ns"] > event["start_ns"],
                    "bubble must reference a positive wait interval")
            require(bubble.get("classification") in CAUSES, "invalid bubble classification")
            require(
                nonempty(bubble.get("reason")) and nonempty(bubble.get("evidence")),
                "bubble needs reason and evidence/provenance")
            require(
                "critical_path" in bubble
                and (bubble["critical_path"] is None or type(bubble["critical_path"]) is bool),
                "bubble critical_path must be true, false or null")


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def clipped(event, window):
    start, end = max(event["start_ns"], window[0]), min(event["end_ns"], window[1])
    return (start, end) if start <= end else None


def summarize(data):
    """Union lengths describe only the selected window, never inferred engine utilization."""
    window = data["window_ns"]
    panels = []
    for panel in data["panels"]:
        lanes = []
        for lane in panel["lanes"]:
            evidence, waits, work, recorded = defaultdict(list), defaultdict(list), [], []
            for event in lane["intervals"]:
                span = clipped(event, window)
                if span is None:
                    continue
                recorded.append(span)
                evidence[event["evidence"]].append(span)
                if event["kind"] == "wait":
                    waits[event["evidence"]].append(span)
                if event["evidence"] == "observed_activity" and event["kind"] in WORK:
                    work.append(span)
            captured = lane["status"] == "captured"
            lanes.append({
                "id": lane["id"],
                "kind": lane["kind"],
                "status": lane["status"],
                "interval_union_ns_by_evidence": {key: duration(value)
                                                  for key, value in sorted(evidence.items())},
                "wait_union_ns_by_evidence": {key: duration(value)
                                              for key, value in sorted(waits.items())},
                "direct_work_union_ns": duration(work) if work else None,
                "uncovered_ns": window[1] - window[0] - duration(recorded) if captured else None,
            })
        event_map = {event["id"]: (lane, event) for lane in panel["lanes"] for event in lane["intervals"]}
        bubbles = []
        for bubble in panel.get("bubbles", []):
            lane, event = event_map[bubble["interval"]]
            span = clipped(event, window)
            if span is not None and span[1] > span[0]:
                bubbles.append({
                    **bubble, "lane": lane["id"], "start_ns": span[0], "end_ns": span[1], "duration_ns":
                    span[1] - span[0]
                })
        panels.append({
            "label": panel["label"], "invocation": panel["invocation"], "scope": panel["scope"], "timebase":
            panel["timebase"], "lanes": lanes, "bubbles": bubbles
        })
    return {
        "schema_version": 1, "synthetic": data["synthetic"], "window_ns": window, "note":
        "Union lengths by evidence; uncovered time is unclassified, not idle. No automatic bottleneck verdict.",
        "panels": panels
    }


def tracks(intervals, window):
    """Pack overlapping intervals without changing times or inventing physical warps."""
    ends, packed = [], []
    for event in sorted(intervals, key=lambda e: (e["start_ns"], e["end_ns"], e["id"])):
        span = clipped(event, window)
        if span is None:
            continue
        start, end = span
        track = next((i for i, limit in enumerate(ends) if limit <= start), len(ends))
        if track == len(ends):
            ends.append(end)
        else:
            ends[track] = end
        packed.append((event, start, end, track))
    return packed, max(1, len(ends))


def tick_step(span):
    target = span / 8
    scale = 10**math.floor(math.log10(target))
    return next(n * scale for n in (1, 2, 5, 10) if n * scale >= target)


def us(ns):
    return f"{ns / 1000:.3f}".rstrip("0").rstrip(".")


def render(data, width=1500):
    validate(data)
    require(integer(width) and 900 <= width <= 4000, "width must be 900..4000 pixels")
    window = data["window_ns"]
    plot_width = width - LEFT - RIGHT
    x = lambda ns: LEFT + (ns - window[0]) * plot_width / (window[1] - window[0])
    elements = []

    def add(tag, attrs=None, body=None):
        attrs = attrs or {}
        attr_text = " ".join(
            f'{key.replace("_", "-")}="{escape(str(value), quote=True)}"' for key, value in attrs.items())
        elements.append(f"<{tag}{' ' if attr_text else ''}{attr_text}>" + (body or "") + f"</{tag}>")

    def text(px, py, value, cls="small", **attrs):
        add("text", {"x": px, "y": py, "class": cls, **attrs}, escape(str(value)))

    def wrapped(value, py, px=MARGIN, cls="small", available=None):
        font_size = 20 if cls == "title" else 12
        for line in textwrap.wrap(value, max(20, int((available or width - px - RIGHT) / (font_size * .58)))):
            text(px, py, line, cls)
            py += font_size + 5
        return py

    y = wrapped(data["title"], 30, cls="title")
    if data.get("subtitle"):
        y = wrapped(data["subtitle"], y + 4)
    if data["synthetic"]:
        y = wrapped("SYNTHETIC EXAMPLE — illustrative timings; hardware activity is not measured.", y + 4, cls="notice")
    y = wrapped(
        "Panels compare separate invocations on the same relative scale; concurrency is interpreted within a panel.",
        y + 4)
    for kind, color in COLORS.items():
        position = list(COLORS).index(kind)
        column, row = position % 4, position // 4
        px, py = MARGIN + column * ((width - 2 * MARGIN) / 4), y + row * 23
        add("rect", {"x": px, "y": py - 11, "width": 18, "height": 12, "fill": color, "rx": 2})
        text(px + 25, py, kind)
    y += 50
    for i, evidence in enumerate(EVIDENCE):
        px, py = MARGIN + (i % 3) * ((width - 2 * MARGIN) / 3), y + (i // 3) * 23
        add("rect", {
            "x": px, "y": py - 12, "width": 20, "height": 14, "fill": COLORS["transfer"], "class": f"event {evidence}"
        })
        if evidence in {"sampled", "unknown"}:
            add("rect", {"x": px, "y": py - 12, "width": 20, "height": 14, "fill": f"url(#{evidence}-pattern)"})
        text(px + 27, py, evidence)
    y += 50
    y = wrapped(
        "Blank time is unclassified. Thin vertical marks are points. Stacked bars show overlapping intervals, not extra warps.",
        y)
    step = tick_step(window[1] - window[0])
    ticks = []
    current = math.ceil(window[0] / step) * step
    while current <= window[1]:
        ticks.append(current)
        current += step
    for panel in data["panels"]:
        y += 25
        text(MARGIN, y, panel["label"], "panel-title")
        y = wrapped(f'{panel["invocation"]} · {panel["scope"]}', y + 20)
        clock = panel["timebase"]
        y = wrapped(
            f'Clock: {clock["source"]}; resolution: {clock["resolution_ns"]} ns; alignment: {clock["alignment"]}', y)
        if panel.get("note"):
            y = wrapped(panel["note"], y, cls="notice")
        y += 8
        lane_layout, positions, group = [], {}, None
        for lane in sorted(panel["lanes"], key=lambda item: item["kind"] == "hardware"):
            if lane["kind"] != group:
                group = lane["kind"]
                text(MARGIN, y + 13, "WG PIPELINE" if group == "wg" else "HARDWARE MODULES", "group-title")
                y += 25
            packed, count = tracks(lane["intervals"], window)
            lane_layout.append((lane, y, packed, count))
            for event, start, end, track in packed:
                positions[event["id"]] = (x(end), y + track * ROW + BAR / 2)
            y += max(38, count * ROW + 10)
        plot_top, plot_bottom = lane_layout[0][1], y
        for phase in panel.get("phases", []):
            span = clipped(phase, window)
            if span is None or span[0] == span[1]:
                continue
            start, end = span
            add(
                "rect", {
                    "x": f"{x(start):.3f}", "y": plot_top - 22, "width": f"{x(end) - x(start):.3f}", "height":
                    plot_bottom - plot_top + 22, "fill": "#64748b", "opacity": ".045"
                })
            text(x(start) + 4, plot_top - 8, phase["label"], "phase")
        for tick in ticks:
            add("line", {"x1": x(tick), "x2": x(tick), "y1": plot_top, "y2": plot_bottom, "class": "grid"})
            text(x(tick), plot_bottom + 14, us(tick), "tick", text_anchor="middle")
        text(width - RIGHT, plot_bottom + 32, "relative time (μs)", text_anchor="end")
        for lane, ly, packed, count in lane_layout:
            label = textwrap.shorten(lane["label"], 31, placeholder="…")
            text(MARGIN, ly + 13, label, "lane")
            scope = textwrap.shorten(lane["scope"], 34, placeholder="…")
            add("text", {"x": MARGIN, "y": ly + 28, "class": "scope"},
                f'<title>{escape(lane["label"] + ": " + lane["scope"])}</title>{escape(scope)}')
            add("line", {"x1": LEFT, "x2": width - RIGHT, "y1": ly + BAR, "y2": ly + BAR, "class": "lane-line"})
            if not packed:
                status = lane["status"] if lane[
                    "status"] != "captured" else "no intervals in this window; activity unclassified"
                text(LEFT + 6, ly + 12, status, "scope")
            for event, start, end, track in packed:
                ey = ly + track * ROW
                evidence = event["evidence"]
                body = "<title>" + escape(json.dumps(event, ensure_ascii=False, sort_keys=True)) + "</title>"
                common = {
                    "class": f"event {evidence}", "data-id": event["id"], "data-evidence": evidence, "data-start-ns":
                    event["start_ns"], "data-end-ns": event["end_ns"], "data-lane": lane["id"], "data-invocation":
                    panel["invocation"]
                }
                if event["start_ns"] == event["end_ns"]:
                    add(
                        "line", {
                            **common, "x1": x(start), "x2": x(start), "y1": ey - 2, "y2": ey + BAR + 2, "stroke":
                            COLORS[event["kind"]]
                        }, body)
                    continue
                if end <= start:
                    continue
                bw = x(end) - x(start)
                attrs = {
                    **common, "x": f"{x(start):.3f}", "y": ey, "width": f"{bw:.3f}", "height": BAR, "rx": 2, "fill":
                    COLORS[event["kind"]]
                }
                add("rect", attrs, body)
                if evidence in {"sampled", "unknown"}:
                    add(
                        "rect", {
                            "x": attrs["x"], "y": ey, "width": attrs["width"], "height": BAR, "fill":
                            f"url(#{evidence}-pattern)"
                        })
                caption = event["label"]
                if event.get("tile") is not None:
                    caption = f'{event["tile"]} {caption}'
                if bw >= 65:
                    limit = max(8, int((bw - 8) / 6))
                    color = "#fff" if evidence == "observed_activity" and event["kind"] == "tensor" else "#1f2937"
                    text(
                        x(start) + 4, ey + 12, textwrap.shorten(caption, limit, placeholder="…"), "bar-label",
                        fill=color)
        for dependency in panel.get("dependencies", []):
            if dependency["from"] not in positions or dependency["to"] not in positions:
                continue
            sx, sy = positions[dependency["from"]]
            tx, ty = positions[dependency["to"]]
            # Do not turn a cropped end into a claimed completion point.
            event_map = {event["id"]: event for lane in panel["lanes"] for event in lane["intervals"]}
            if any(not window[0] <= event_map[key]["end_ns"] <= window[1]
                   for key in (dependency["from"], dependency["to"])):
                continue
            add("path",
                {"d": f"M {sx:.3f} {sy:.3f} L {tx:.3f} {ty:.3f}", "class": "dependency", "marker-end": "url(#arrow)"},
                f'<title>{escape(json.dumps(dependency, ensure_ascii=False))}</title>')
        y = plot_bottom + 48
        event_map = {event["id"]: (lane, event) for lane in panel["lanes"] for event in lane["intervals"]}
        for number, bubble in enumerate(panel.get("bubbles", []), 1):
            lane, event = event_map[bubble["interval"]]
            span = clipped(event, window)
            if span is None or span[0] == span[1]:
                continue
            px, py = positions[event["id"]]
            add("circle", {"cx": px, "cy": py - 7, "r": 8, "fill": "#fff", "stroke": COLORS["wait"]})
            text(px, py - 4, str(number), "bubble-number", text_anchor="middle")
            critical = {True: "yes", False: "no", None: "unknown"}[bubble["critical_path"]]
            y = wrapped(
                f'B{number} · {lane["label"]} · {us(span[0])}–{us(span[1])} μs · {us(span[1] - span[0])} μs · '
                f'{bubble["classification"]} · critical path: {critical}', y, cls="bubble")
            y = wrapped(f'{bubble["reason"]} Evidence: {bubble["evidence"]}', y, px=MARGIN + 12)
            y += 5
        add("line", {"x1": MARGIN, "x2": width - RIGHT, "y1": y, "y2": y, "class": "lane-line"})
    metadata = {
        "schema_version": 1, "synthetic": data["synthetic"], "window_ns": window, "panels":
        [{k: p[k]
          for k in ("label", "invocation", "scope", "timebase")}
         for p in data["panels"]]
    }
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{y + 20}" viewBox="0 0 {width} {y + 20}" role="img">
<title>{escape(data["title"])}</title>
<desc>FlagMega kernel swimlane. Color denotes stage; stroke/pattern denotes evidence. Blank time is unclassified.</desc>
<metadata>{escape(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}</metadata>
<defs>
<marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#475569"/></marker>
<pattern id="sampled-pattern" width="6" height="6" patternUnits="userSpaceOnUse"><path d="M0,6 L6,0" stroke="#334155" stroke-opacity=".4"/></pattern>
<pattern id="unknown-pattern" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M0,0 L8,8 M0,8 L8,0" stroke="#64748b" stroke-opacity=".4"/></pattern>
</defs>
<style>
text {{ font-family: Arial, "DejaVu Sans", sans-serif; fill: #1f2937; }}
.title {{ font-size: 20px; font-weight: 700; }} .panel-title {{ font-size: 17px; font-weight: 700; }}
.small, .lane {{ font-size: 12px; }} .lane {{ font-weight: 700; }}
.scope, .tick, .phase, .bar-label {{ font-size: 10px; }} .scope, .tick {{ fill: #64748b; }}
.notice, .bubble {{ font-size: 12px; fill: #9a3412; }} .group-title {{ font-size: 11px; font-weight: 700; fill: #475569; }}
.bubble-number {{ font-size: 9px; fill: #9a3412; }}
.grid {{ stroke: #e5e7eb; stroke-width: 1; }} .lane-line {{ stroke: #d1d5db; stroke-width: 1; }}
.event {{ stroke-width: 1; }} .observed_activity {{ fill-opacity: .9; }}
.inflight_envelope {{ fill-opacity: .08; stroke: #475569; stroke-dasharray: 5 3; }}
.sampled {{ fill-opacity: .45; stroke: #475569; }}
.inferred {{ fill-opacity: .35; stroke: #64748b; stroke-dasharray: 2 3; }}
.unknown {{ fill-opacity: .15; stroke: #64748b; stroke-dasharray: 2 2; }}
.dependency {{ fill: none; stroke: #475569; stroke-width: 1; stroke-dasharray: 3 3; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
''' + "\n".join(elements) + "\n</svg>\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="normalized JSON; see references/kernel-trace-method.md#renderer")
    parser.add_argument("--output", required=True, type=Path, help="SVG output path")
    parser.add_argument("--summary", type=Path, help="optional JSON interval/wait union lengths")
    parser.add_argument("--width", type=int, default=1500, help="SVG width, 900..4000 (default: 1500)")
    args = parser.parse_args()
    try:
        paths = [path.resolve() for path in (args.input, args.output, args.summary) if path is not None]
        require(len(paths) == len(set(paths)), "input, output and summary must have distinct paths")
        require(args.output.suffix.lower() == ".svg", "output must have a .svg suffix")
        data = json.loads(args.input.read_text(encoding="utf-8"))
        svg = render(data, args.width)
        summary = json.dumps(summarize(data), indent=2, ensure_ascii=False) + "\n" if args.summary else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg, encoding="utf-8")
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(summary, encoding="utf-8")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output}" + (f" and {args.summary}" if args.summary else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
