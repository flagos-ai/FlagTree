import argparse
from collections import namedtuple
import json
import pandas as pd
import re

try:
    import hatchet as ht
    from hatchet.query import NegationQuery
except ImportError:
    raise ImportError("Failed to import hatchet. `pip install llnl-hatchet` to get the correct version.")
import numpy as np
from triton.profiler.hooks.launch import COMPUTE_METADATA_SCOPE_NAME, LaunchHook
from triton.profiler import specs


def match_available_metrics(metrics, inclusive_metrics, exclusive_metrics):
    ret = []
    if not isinstance(metrics, list):
        metrics = [metrics]
    if metrics:
        for metric in metrics:
            metric = metric.lower()
            for raw_metric in inclusive_metrics + exclusive_metrics:
                suffix = " (inc)" if raw_metric in inclusive_metrics else ""
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                if metric in (raw_metric.lower(), raw_metric_no_unit):
                    ret.append(raw_metric + suffix)
                    break
    if len(ret) == 0:
        raise RuntimeError(f"Metric {metric} is not found. Use the --list flag to list available metrics")
    return ret


def _metric_base_name(metric):
    return metric.split("(")[0].strip()


def match_raw_metric(metric, inclusive_metrics, exclusive_metrics):
    metric = metric.lower()
    for raw_metric in inclusive_metrics + exclusive_metrics:
        raw_metric_no_unit = _metric_base_name(raw_metric).lower()
        if metric in (raw_metric.lower(), raw_metric_no_unit):
            return raw_metric
    raise RuntimeError(f"Metric {metric} is not found. Use the --list flag to list available metrics")


def is_aggregable_metric(metric):
    name = _metric_base_name(metric).lower()
    if name in {"cann.op_summary_begin_time_us", "cann.op_summary_finish_time_us"}:
        return False
    if any(token in name for token in ("bandwidth", "throughput", "variance", "util")):
        return False
    if name.endswith(("_ratio", ".ratio", "_rate", ".rate")):
        return False
    if re.search(r"(^|[._])(?:t|g)flop\d*_s$", name):
        return False
    return True


def clear_internal_values_for_nonaggregable_metric(gf, metric):
    raw_metric = metric.removesuffix(" (inc)")
    if is_aggregable_metric(raw_metric) or raw_metric not in gf.dataframe:
        return
    internal_nodes = [bool(node.children) for node in gf.dataframe.index]
    gf.dataframe.loc[internal_nodes, raw_metric] = np.nan


def remove_frames(database: json):
    # We first fine frames that match either one of the two conditions:
    # 1. The frame name is COMPUTE_METADATA_SCOPE_NAME
    # 2. The frame has no metrics and no children
    # Then we go up from the located nodes and remove the parents if all children were
    # metadata nodes
    def remove_frame_helper(node):
        if "frame" not in node:
            return node
        if node["frame"]["name"] == COMPUTE_METADATA_SCOPE_NAME:
            return None
        if len(node["metrics"]) == 0 and len(node["children"]) == 0:
            return None
        children = node.get("children", [])
        new_children = []
        for child in children:
            new_child = remove_frame_helper(child)
            if new_child is not None:
                new_children.append(new_child)
        if len(new_children) > 0 or len(children) == 0:
            node["children"] = new_children
            return node
        return None

    new_database = []
    for node in database:
        new_node = remove_frame_helper(node)
        if new_node is not None:
            new_database.append(new_node)
    return new_database


def get_raw_metrics(file):
    database = json.load(file)
    database = remove_frames(database)
    device_info = database.pop(1)
    gf = ht.GraphFrame.from_literal(database)
    inclusive_metrics = gf.show_metric_columns()
    exclusive_metrics = [metric for metric in gf.dataframe.columns if metric not in inclusive_metrics]
    return gf, inclusive_metrics, exclusive_metrics, device_info


def get_min_time_flops(df, device_info):
    min_time_flops = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            arch = device_info[device_type][device_index]["arch"]
            num_sms = device_info[device_type][device_index]["num_sms"]
            clock_rate = device_info[device_type][device_index]["clock_rate"]
            for width in LaunchHook.flops_width:
                idx = df["device_id"] == device_index
                device_frames = df[idx]
                if f"flops{width}" not in device_frames.columns:
                    continue
                max_flops = specs.max_flops(device_type, arch, width, num_sms, clock_rate)
                min_time_flops.loc[idx, "min_time"] += device_frames[f"flops{width}"].fillna(0) / max_flops
    return min_time_flops


def get_min_time_bytes(df, device_info):
    min_time_bytes = pd.DataFrame(0.0, index=df.index, columns=["min_time"])
    for device_type in device_info:
        for device_index in device_info[device_type]:
            idx = df["device_id"] == device_index
            device_frames = df[idx]
            device = device_info[device_type][device_index]
            memory_clock_rate = device["memory_clock_rate"]  # in khz
            bus_width = device["bus_width"]  # in bits
            peak_bandwidth = specs.max_bps(device_type, device['arch'], bus_width, memory_clock_rate)
            min_time_bytes.loc[idx, "min_time"] += device_frames["bytes"] / peak_bandwidth
    return min_time_bytes


FactorDict = namedtuple("FactorDict", ["name", "factor"])
time_factor_dict = FactorDict("time", {"time/s": 1, "time/ms": 1e-3, "time/us": 1e-6, "time/ns": 1e-9})
avg_time_factor_dict = FactorDict("avg_time", {f"avg_{key}": value for key, value in time_factor_dict.factor.items()})
cpu_time_factor_dict = FactorDict("cpu_time",
                                  {"cpu_time/s": 1, "cpu_time/ms": 1e-3, "cpu_time/us": 1e-6, "cpu_time/ns": 1e-9})
avg_cpu_time_factor_dict = FactorDict("avg_cpu_time",
                                      {f"avg_{key}": value
                                       for key, value in cpu_time_factor_dict.factor.items()})
bytes_factor_dict = FactorDict("bytes", {"byte/s": 1, "gbyte/s": 1e9, "tbyte/s": 1e12})

derivable_metrics = {
    **{key: bytes_factor_dict
       for key in bytes_factor_dict.factor.keys()},
}

# FLOPS have a specific width to their metric
default_flop_factor_dict = {"flop/s": 1, "gflop/s": 1e9, "tflop/s": 1e12}
derivable_metrics.update(
    {key: FactorDict("flops", default_flop_factor_dict)
     for key in default_flop_factor_dict.keys()})
for width in LaunchHook.flops_width:
    factor_name = f"flops{width}"
    factor_dict = {f"flop{width}/s": 1, f"gflop{width}/s": 1e9, f"tflop{width}/s": 1e12}
    derivable_metrics.update({key: FactorDict(factor_name, factor_dict) for key in factor_dict.keys()})


def cann_derived_metric_spec(metric):
    if metric == "cann.tflop_s":
        return "flops", 1e12
    match = re.fullmatch(r"cann\.tflop(8|16|32|64)_s", metric)
    if match:
        return f"flops{match.group(1)}", 1e12
    if metric == "cann.estimated_bandwidth_gb_s":
        return "bytes", 1e9
    return None


def derive_cann_metric(gf, metric, inclusive_metrics, exclusive_metrics):
    spec = cann_derived_metric_spec(metric)
    if spec is None:
        return None
    numerator_name, scale = spec
    try:
        numerator_metric = match_raw_metric(numerator_name, inclusive_metrics, exclusive_metrics)
        duration_metric = match_raw_metric("cann.task_duration_us", inclusive_metrics, exclusive_metrics)
    except RuntimeError:
        return None

    def subtree_sum(metric_name):
        values = gf.dataframe[metric_name].astype(float)
        sums = {}

        def visit(node):
            value = values.loc[node]
            total = 0.0 if pd.isna(value) else value
            for child in node.children:
                total += visit(child)
            sums[node] = total
            return total

        for root in gf.graph.roots:
            visit(root)
        return pd.Series(
            [sums.get(node, np.nan) for node in gf.dataframe.index],
            index=gf.dataframe.index,
        )

    numerator = subtree_sum(numerator_metric)
    duration_us = subtree_sum(duration_metric)
    gf.dataframe[metric] = np.where(
        (numerator > 0.0) & (duration_us > 0.0),
        numerator / (duration_us * 1.0e-6) / scale,
        np.nan,
    )
    return metric


def get_cann_scope_elapsed_seconds(gf, inclusive_metrics, exclusive_metrics):
    try:
        begin_metric = match_raw_metric("cann.op_summary_begin_time_us", inclusive_metrics, exclusive_metrics)
        finish_metric = match_raw_metric("cann.op_summary_finish_time_us", inclusive_metrics, exclusive_metrics)
    except RuntimeError:
        return None

    begin_values = gf.dataframe[begin_metric].astype(float)
    finish_values = gf.dataframe[finish_metric].astype(float)
    elapsed_us = {}

    def visit(node):
        min_begin = None
        max_finish = None
        if not node.children:
            begin = begin_values.loc[node]
            finish = finish_values.loc[node]
            min_begin = None if pd.isna(begin) else begin
            max_finish = None if pd.isna(finish) else finish
        for child in node.children:
            child_begin, child_finish = visit(child)
            if child_begin is not None:
                min_begin = child_begin if min_begin is None else min(min_begin, child_begin)
            if child_finish is not None:
                max_finish = child_finish if max_finish is None else max(max_finish, child_finish)
        elapsed_us[node] = (max_finish - min_begin
                            if min_begin is not None and max_finish is not None and max_finish >= min_begin else np.nan)
        return min_begin, max_finish

    for root in gf.graph.roots:
        visit(root)
    return pd.Series(
        [elapsed_us.get(node, np.nan) * 1.0e-6 for node in gf.dataframe.index],
        index=gf.dataframe.index,
    )


def derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info):
    derived_metrics = []
    cann_scope_time_seconds = get_cann_scope_elapsed_seconds(gf, inclusive_metrics, exclusive_metrics)

    def get_time_seconds(df, metric, factor_dict):
        if metric == "time" and cann_scope_time_seconds is not None:
            return cann_scope_time_seconds
        time_metric_name = match_available_metrics(metric, inclusive_metrics, exclusive_metrics)[0]
        time_unit = factor_dict.name + "/" + time_metric_name.split("(")[1].split(")")[0]
        return df[time_metric_name] * factor_dict.factor[time_unit]

    for metric in metrics:
        cann_metric = derive_cann_metric(gf, metric, inclusive_metrics, exclusive_metrics)
        if cann_metric is not None:
            derived_metrics.append(cann_metric)
        elif metric == "util":  # exclusive
            min_time_bytes = get_min_time_bytes(gf.dataframe, device_info)
            min_time_flops = get_min_time_flops(gf.dataframe, device_info)
            time_sec = get_time_seconds(gf.dataframe, "time", time_factor_dict)
            internal_frame_indices = gf.dataframe["device_id"].isna()
            gf.dataframe["util"] = min_time_flops["min_time"].combine(min_time_bytes["min_time"], max) / time_sec
            gf.dataframe.loc[internal_frame_indices, "util"] = np.nan
            derived_metrics.append("util")
        elif metric in derivable_metrics:  # flop<width>/s, <t/g>byte/s, inclusive
            derivable_metric = derivable_metrics[metric]
            metric_name = derivable_metric.name
            metric_factor_dict = derivable_metric.factor
            matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
            gf.dataframe[f"{metric} (inc)"] = (gf.dataframe[matched_metric_name] /
                                               (get_time_seconds(gf.dataframe, "time", time_factor_dict)) /
                                               metric_factor_dict[metric])
            derived_metrics.append(f"{metric} (inc)")
        elif (metric in time_factor_dict.factor or metric in cpu_time_factor_dict.factor
              or metric in avg_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor):  # inclusive
            is_cpu = metric in cpu_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor
            is_avg = metric in avg_time_factor_dict.factor or metric in avg_cpu_time_factor_dict.factor

            factor_dict = ((avg_cpu_time_factor_dict if is_avg else cpu_time_factor_dict) if is_cpu else
                           (avg_time_factor_dict if is_avg else time_factor_dict))
            metric_name = "cpu_time" if is_cpu else "time"
            metric_time_unit = factor_dict.name + "/" + metric.split("/")[1]

            time_value = get_time_seconds(gf.dataframe, metric_name, factor_dict)
            if is_avg:
                time_value = time_value / gf.dataframe["count (inc)"]

            output_metric = (f"{metric} (cann elapsed)"
                             if not is_cpu and cann_scope_time_seconds is not None else f"{metric} (inc)")
            gf.dataframe[output_metric] = time_value / factor_dict.factor[metric_time_unit]
            derived_metrics.append(output_metric)
        else:
            metric_name_and_unit = metric.split("/")
            metric_name = metric_name_and_unit[0]
            if len(metric_name_and_unit) > 1:  # percentage, exclusive or inclusive
                metric_unit = metric_name_and_unit[1]
                if metric_unit != "%":
                    raise ValueError(f"Unsupported unit {metric_unit}")
                matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
                single_frame = gf.dataframe[matched_metric_name]
                suffix = ""
                if "(inc)" in matched_metric_name:
                    suffix = " (inc)"
                    total = gf.dataframe[matched_metric_name].iloc[0]
                else:
                    total = gf.dataframe[matched_metric_name].sum()
                gf.dataframe[metric + suffix] = (single_frame / total) * 100.0
                derived_metrics.append(metric + suffix)
            else:
                matched_metric_name = match_available_metrics(metric_name, inclusive_metrics, exclusive_metrics)[0]
                clear_internal_values_for_nonaggregable_metric(gf, matched_metric_name)
                derived_metrics.append(matched_metric_name)

    # Update derived metrics to the graph frame
    for derived_metric in derived_metrics:
        if derived_metric.endswith("(inc)"):
            gf.inc_metrics.append(derived_metric)
        else:
            gf.exc_metrics.append(derived_metric)

    return derived_metrics


def format_frames(gf, format):
    if format == "file_function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split("/")[-1])
    elif format == "function_line":
        gf.dataframe["name"] = gf.dataframe["name"].apply(lambda x: x.split(":")[-1])
    elif format == "file_function":
        gf.dataframe["name"] = gf.dataframe["name"].apply(
            lambda x: f"{x.split('/')[-1].split(':')[0]}@{x.split('@')[-1].split(':')[0]}")
    return gf


def filter_frames(gf, include=None, exclude=None, threshold=None, metric=None):
    if include:
        query = f"""
MATCH ("*")->(".", p)->("*")
WHERE p."name" =~ "{include}"
"""
        gf = gf.filter(query, squash=True)
    if exclude:
        inclusion_query = f"""
MATCH (".", p)->("*")
WHERE p."name" =~ "{exclude}"
"""
        query = NegationQuery(inclusion_query)
        gf = gf.filter(query, squash=True)
    if threshold:
        query = ["*", {metric: f">= {threshold}"}]
        gf = gf.filter(query, squash=True)
    return gf


def emit_warnings(gf, metrics):
    if "bytes (inc)" in metrics:
        byte_values = gf.dataframe["bytes (inc)"].values
        min_byte_value = np.nanmin(byte_values)
        if min_byte_value < 0:
            print("Warning: Negative byte values detected, this is usually the result of a datatype overflow\n")


def print_tree(gf, metrics, depth=100, format=None, print_sorted=False):
    gf = format_frames(gf, format)
    print("Metrics: " + " | ".join(metrics))
    print(gf.tree(metric_column=metrics, expand_name=True, depth=depth, render_header=False))

    if print_sorted:
        print("Sorted kernels by metric " + metrics[0])
        sorted_df = gf.dataframe.sort_values(by=[metrics[0]], ascending=False)
        for row in range(1, len(sorted_df)):
            kernel_name = (sorted_df.iloc[row]["name"][:100] +
                           "..." if len(sorted_df.iloc[row]["name"]) > 100 else sorted_df.iloc[row]["name"])
            print("{:105} {:.4}".format(kernel_name, sorted_df.iloc[row][metrics[0]]))
    emit_warnings(gf, metrics)


def read(filename):
    with open(filename, "r") as f:
        gf, inclusive_metrics, exclusive_metrics, device_info = get_raw_metrics(f)
        assert len(inclusive_metrics + exclusive_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        return gf, inclusive_metrics, exclusive_metrics, device_info


def parse(metrics, filename, include=None, exclude=None, threshold=None):
    gf, inclusive_metrics, exclusive_metrics, device_info = read(filename)
    metrics = derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info)
    # TODO: generalize to support multiple metrics, not just the first one
    gf = filter_frames(gf, include, exclude, threshold, metrics[0])
    return gf, metrics


def show_metrics(file_name):
    with open(file_name, "r") as f:
        _, inclusive_metrics, exclusive_metrics, _ = get_raw_metrics(f)
        print("Available inclusive metrics:")
        if inclusive_metrics:
            for raw_metric in inclusive_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")
        print("Available exclusive metrics:")
        if exclusive_metrics:
            for raw_metric in exclusive_metrics:
                raw_metric_no_unit = raw_metric.split("(")[0].strip().lower()
                print(f"- {raw_metric_no_unit}")


def main():
    argparser = argparse.ArgumentParser(
        description="Performance data viewer for proton profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    argparser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="""List available metrics. Metric names are case insensitive and ignore units.
Derived metrics can be created when source metrics are available.
- time/s, time/ms, time/us, time/ns: time
- avg_time/s, avg_time/ms, avg_time/us, avg_time/ns: time / count
- flop[<8/16/32/64>]/s, gflop[<8/16/32/64>]/s, tflop[<8/16/32/64>]/s: flops / time
- byte/s, gbyte/s, tbyte/s: bytes / time
- util: max(sum(flops<width>) / peak_flops<width>_time, sum(bytes) / peak_bandwidth_time)
- <metric>/%%: frame(metric) / sum(metric). Only available for inclusive metrics (e.g. time)
""",
    )
    argparser.add_argument(
        "-m",
        "--metrics",
        type=str,
        default=None,
        help="""At maximum two metrics can be specified, separated by comma.
There are two modes:
1) Choose the output metric to display. It's case insensitive and ignore units.
2) Derive a new metric from existing metrics.
""",
    )
    argparser.add_argument(
        "-i",
        "--include",
        type=str,
        default=None,
        help=
        """Find frames that match the given regular expression and return all nodes in the paths that pass through the matching frames.
For example, the following command will display all paths that contain frames that contains "test":
```
proton-viewer -i ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-e",
        "--exclude",
        type=str,
        default=None,
        help="""Exclude frames that match the given regular expression and their children.
For example, the following command will exclude all paths starting from frames that contains "test":
```
proton-viewer -e ".*test.*" path/to/file.json
```
""",
    )
    argparser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=
        "Exclude frames(kernels) whose metrics are below the given threshold. This filter only applies on the first metric.",
    )
    argparser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=100,
        help="The depth of the tree to display",
    )
    argparser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["full", "file_function_line", "function_line", "file_function"],
        default="full",
        help="""Formatting the frame name.
- full: include the path, file name, function name and line number.
- file_function_line: include the file name, function name and line number.
- function_line: include the function name and line number.
- file_function: include the file name and function name.
""",
    )
    argparser.add_argument(
        "--print-sorted",
        action="store_true",
        default=False,
        help="Sort output by metric value instead of chronologically",
    )
    argparser.add_argument(
        "--diff-profile",
        "-diff",
        type=str,
        default=None,
        help="Compare two profiles. When used as 'proton-viewer -m time -diff file1.log file2.log', "
        "computes the difference: file2['time'] - file1['time']",
    )

    args, target_args = argparser.parse_known_args()
    assert len(target_args) == 1, "Must specify a file to read"

    file_name = target_args[0]
    metrics = args.metrics.split(",") if args.metrics else None
    include = args.include
    exclude = args.exclude
    threshold = args.threshold
    depth = args.depth
    format = args.format
    diff = args.diff_profile
    print_sorted = args.print_sorted
    if include and exclude:
        raise ValueError("Cannot specify both include and exclude")
    if args.list:
        show_metrics(file_name)
    elif metrics:
        gf, derived_metrics = parse(metrics, file_name, include, exclude, threshold)
        if diff:
            gf2, _ = parse(metrics, diff, include, exclude, threshold)
            gf = gf.sub(gf2)
        print_tree(gf, derived_metrics, depth, format, print_sorted)


if __name__ == "__main__":
    main()
