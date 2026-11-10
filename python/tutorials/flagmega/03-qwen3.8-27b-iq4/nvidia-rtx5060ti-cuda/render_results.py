"""Render the final two-series performance history and latest comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

SCENARIOS = ("synthetic32", "chinese64")
QUALITY = {"screening": 1, "matched": 2, "formal": 3}
COLORS = {"llama.cpp": "#64748b", "flagmega": "#2563eb"}


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def scenario_statistics(rows):
    result = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        if not selected:
            return None
        samples = [value for row in selected for value in row["latency_ms"]]
        latency = statistics.mean(samples)
        result[scenario] = dict(mean_latency_ms=latency, tokens_per_second=1000 / latency,
            measurement_ids=[row["measurement_id"] for row in selected],
            process_count=len(selected), sample_count=len(samples))
    return result


def build_series(data):
    if data.get("schema") != "flagmega.cpp-public-history/v1":
        raise ValueError("Unknown history schema")
    ids, candidates, references, coverage = set(), defaultdict(list), defaultdict(list), defaultdict(set)
    for row in data["records"]:
        if row["measurement_id"] in ids:
            raise ValueError("Duplicate measurement ID")
        ids.add(row["measurement_id"])
        samples = row.get("latency_ms", [])
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("Invalid latency sample")
        if row.get("mean_ms") is not None and (not samples or not math.isclose(
                row["mean_ms"], statistics.mean(samples), rel_tol=1e-10)):
            raise ValueError("Published statistic does not match its samples")
        if (not samples or not row.get("measurement_valid") or row.get("scenario") not in SCENARIOS
                or row.get("kind") not in QUALITY):
            continue
        cohort, kind = row["cohort"], row["kind"]
        if row["variant"] == "reference":
            if kind in ("formal", "matched"):
                references[(cohort, kind)].append(row)
            continue
        if row["variant"] not in ("baseline", "candidate"):
            continue
        iteration = row.get("iteration_index")
        if iteration is None:
            continue
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Invalid iteration index")
        if row.get("numerical_status") != "passed" or not row.get("qualification_receipt"):
            raise ValueError("Iteration lacks complete numerical qualification")
        candidates[(iteration, cohort, kind)].append(row)
        if row["variant"] == "candidate":
            coverage[cohort].add(iteration)

    choices = defaultdict(list)
    for (iteration, cohort, kind), rows in candidates.items():
        stats = scenario_statistics(rows)
        if stats is not None:
            # Evidence strength and cumulative coverage take priority over recency.
            # Never choose by measured speed or pool separate cohorts.
            rank = (QUALITY[kind], len(coverage[cohort]), max(row.get("utc_end", "") for row in rows), cohort)
            choices[iteration].append((rank, dict(iteration=iteration, cohort=cohort, kind=kind,
                scenarios=stats, tokens_per_second=geometric_mean(
                    [value["tokens_per_second"] for value in stats.values()]))))
    points = [max(choices[index], key=lambda item: item[0])[1] for index in sorted(choices)]
    if not points:
        raise ValueError("No complete two-scenario iteration data")
    final = points[-1]
    reference_choices = []
    for (cohort, kind), rows in references.items():
        stats = scenario_statistics(rows)
        if stats is not None:
            rank = (cohort == final["cohort"], QUALITY[kind], max(row.get("utc_end", "") for row in rows), cohort)
            reference_choices.append((rank, dict(cohort=cohort, kind=kind, scenarios=stats,
                tokens_per_second=geometric_mean([value["tokens_per_second"] for value in stats.values()]))))
    if not reference_choices:
        raise ValueError("No complete two-scenario llama.cpp reference")
    reference = max(reference_choices, key=lambda item: item[0])[1]
    return dict(schema="flagmega.final-performance-series/v1", scenarios=list(SCENARIOS),
        aggregation="Geometric mean of per-scenario throughput; each throughput is 1000 / pooled mean latency_ms.",
        selection="Per iteration: formal > matched > screening; then greatest cumulative candidate-iteration coverage; then latest cohort. No speed-based selection or cross-cohort pooling.",
        reference_policy="Horizontal line uses the latest selected candidate's complete matched reference when available. Historical points are not contemporaneous matched comparisons.",
        points=points, reference=reference, final=final,
        final_same_cohort=reference["cohort"] == final["cohort"],
        full_microprofile_accepted=data.get("full_microprofile_accepted", False),
        source_record_count=len(data["records"]))


def render(data_path, output_dir, series_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data_path, output = Path(data_path), Path(output_dir)
    raw = data_path.read_bytes()
    series = build_series(json.loads(raw))
    series["source"] = dict(path=str(data_path.resolve()), nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "svg.fonttype": "none", "svg.hashsalt": "flagmega-final-performance", "axes.titleweight": "semibold"})

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")
        ax.tick_params(colors="#475569", length=0, pad=8)
        ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
        ax.set_axisbelow(True)

    def save_figure(fig, name):
        fig.savefig(output / (name + ".svg"), metadata={"Date": None}, bbox_inches="tight", pad_inches=.16)
        fig.savefig(output / (name + ".png"), dpi=160, bbox_inches="tight", pad_inches=.16)
        plt.close(fig)

    indices = [point["iteration"] for point in series["points"]]
    values = [point["tokens_per_second"] for point in series["points"]]
    reference = series["reference"]["tokens_per_second"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    style(ax)
    llama, = ax.plot(indices, [reference] * len(indices), color=COLORS["llama.cpp"],
                    linewidth=2, linestyle="--", label="llama.cpp")
    flagship, = ax.plot(indices, values, color=COLORS["flagmega"], linewidth=2.4,
                       marker="o", markersize=3.5, label="flagmega")
    llama.set_gid("series-llamacpp")
    flagship.set_gid("series-flagmega")
    assert len(ax.lines) == 2
    ax.set_title("Decode performance by iteration", loc="left", pad=18)
    ax.set_xlabel("Iteration", labelpad=10)
    ax.set_ylabel("Decode tokens/s", labelpad=10)
    ax.set_ylim(0, math.ceil(max(reference, *values) / 5) * 5)
    ax.set_xlim(min(indices) - .4, max(indices) + .4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.legend(loc="lower right", frameon=False, ncols=2)
    fig.text(.12, .015, "Geometric mean: synthetic32 + chinese64.  llama.cpp: latest matched reference.",
             fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, .065, 1, 1))
    save_figure(fig, "performance-vs-iteration")

    final = series["final"]
    labels = ["Synthetic32", "Chinese64", "Geometric mean"]
    candidate_values = [final["scenarios"][key]["tokens_per_second"] for key in SCENARIOS] + [final["tokens_per_second"]]
    reference_values = [series["reference"]["scenarios"][key]["tokens_per_second"] for key in SCENARIOS] + [reference]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style(ax)
    for backend, offset, heights in [("llama.cpp", -.18, reference_values), ("flagmega", .18, candidate_values)]:
        bars = ax.bar([index + offset for index in range(3)], heights, width=.32, color=COLORS[backend], label=backend)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Decode tokens/s", labelpad=10)
    ax.set_ylim(0, max(reference_values + candidate_values) * 1.16)
    ax.set_title("Final decode comparison", loc="left", pad=18)
    ax.legend(loc="upper left", frameon=False, ncols=2, bbox_to_anchor=(0, 1.01))
    caption = "Latest matched reproduction" if series["final_same_cohort"] else "Latest available measurements; cohorts differ"
    fig.text(.12, .015, f"Iteration {final['iteration']} · {caption}. Higher is better.", fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, .065, 1, 1))
    save_figure(fig, "performance-comparison")
    if series_path is not None:
        series_path = Path(series_path)
        series_path.parent.mkdir(parents=True, exist_ok=True)
        series_path.write_text(json.dumps(series, ensure_ascii=False, indent=2) + "\n")
    return dict(records=series["source_record_count"], iteration_points=len(indices), line_series=2,
                final_iteration=final["iteration"], final_flagmega_tps=final["tokens_per_second"],
                llama_cpp_tps=reference, source=str(data_path), output=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
    parser.add_argument("--series-data", type=Path,
                        default=Path(__file__).resolve().parent / ".local/results/final-performance/series.json")
    args = parser.parse_args()
    print(json.dumps(render(args.data, args.output_dir, args.series_data), indent=2))


if __name__ == "__main__":
    main()
