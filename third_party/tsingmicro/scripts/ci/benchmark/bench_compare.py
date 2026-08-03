#!/usr/bin/env python3
"""
Benchmark performance comparison script.
Compares two bench_summary.csv files (baseline vs optimized),
computes per-case speedup, type-level average speedup, and op-level average speedup,
outputs results to CSV files with statistical conclusions.

Usage:
  python3 bench_compare.py <baseline_csv> <optimized_csv> [output_csv]
"""

import csv
import sys
import os
import re
from collections import defaultdict, OrderedDict


def clean_op_name(raw_name):
    """
    Clean op_name by removing _cardN-rerun.log suffix.
    Examples:
      'abs_card6-rerun.log' -> 'abs'
      'normal_tensor_tensor_card6-rerun.log' -> 'normal_tensor_tensor'
      'randperm_card6-rerun.log' -> 'randperm'
      'add' -> 'add'
    """
    # Match _card followed by digits, then -rerun.log
    cleaned = re.sub(r'_card\d+-rerun\.log$', '', raw_name)
    return cleaned


def clean_shape_for_key(op_clean, shape):
    """
    For gather and scatter ops, only use the first input's shape for matching.
    Shapes like '1024,1024x754,327' -> '1024,1024'
    Other ops: return shape as-is.
    """
    if op_clean in ('gather', 'scatter', 'index_add'):
        # Split by 'x' and take the first part
        shape = shape.split('x')[0].strip()
    return shape


def parse_csv(filepath):
    """
    Parse a bench_summary.csv file.
    Returns (data_list, data_dict).
      data_list: all rows in original file order (no dedup)
      data_dict: { (clean_op_name, dtype, cleaned_shape): [row_index_in_list, ...] }
    """
    data_list = []
    data_dict = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 7:
                print(f"[Warning]parse_csv: format error in {filepath}, skip, row content: {row}")
                continue
            op_raw = row[0]
            dtype = row[1]
            shape = row[2]
            status = row[5]
            lat_str = row[6]

            op_clean = clean_op_name(op_raw)
            shape_for_key = clean_shape_for_key(op_clean, shape)

            # Try to parse latency
            try:
                lat = float(lat_str)
            except ValueError:
                lat = None
                print(f"[Warning]parse_csv: lat_str2float error in {filepath}, set none, row content: {row}")

            key = (op_clean, dtype, shape_for_key)

            one_case_perf = {
                'op_raw': op_raw,
                'op_clean': op_clean,
                'dtype': dtype,
                'shape': shape,
                'shape_for_key': shape_for_key,
                'status': status,
                'lat': lat,
            }
            idx = len(data_list)
            data_list.append(one_case_perf)
            data_dict[key].append(idx)

    return data_list, data_dict


def compute_speedup(baseline_lat, optimized_lat):
    """
    Compute speedup = baseline / optimized.
    Returns None if either is None or optimized is 0.
    """
    if baseline_lat is None or optimized_lat is None:
        return None
    if optimized_lat == 0:
        return None
    return baseline_lat / optimized_lat


def write_detailed_csv(output_path, rows):
    """Write the detailed comparison CSV."""
    headers = [
        "op_name(baseline)", "dtype(baseline)", "shape(baseline)", "gems_lat_ms(baseline)",
        "op_name(optimized)", "dtype(optimized)", "shape(optimized)", "gems_lat_ms(optimized)",
        "op_match", "dtype_match", "shape_match",
        "case_speedup", "type_avg_speedup", "op_avg_speedup",
    ]

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            b_op_clean = clean_op_name(row['b_op_raw']) if row['b_op_raw'] else ''
            o_op_clean = clean_op_name(row['o_op_raw']) if row['o_op_raw'] else ''
            match_op = (b_op_clean == o_op_clean)

            # Speedup formatting
            if row['speedup'] is not None:
                speedup_str = f"{row['speedup']:.6f}"
            else:
                speedup_str = 'N/A'

            # Type avg formatting
            if row['type_avg'] is not None:
                type_avg_str = f"{row['type_avg']:.4f}"
            else:
                type_avg_str = 'N/A'

            # Op avg formatting
            if row['op_avg'] is not None:
                op_avg_str = f"{row['op_avg']:.4f}"
            else:
                op_avg_str = 'N/A'

            # Latency formatting
            b_lat_str = f"{row['b_lat']:.3f}" if row['b_lat'] is not None else 'N/A'
            o_lat_str = f"{row['o_lat']:.3f}" if row['o_lat'] is not None else 'N/A'

            writer.writerow([
                clean_op_name(row['b_op_raw']),
                row['dtype'],
                row['shape'],
                b_lat_str,
                clean_op_name(row['o_op_raw']),
                row['dtype'],
                row['o_shape'],
                o_lat_str,
                str(match_op),
                'True',
                'True',
                speedup_str,
                type_avg_str,
                op_avg_str,
            ])

    print(f"Detailed CSV saved to: {output_path}")


def write_summary_csv(output_path, op_summaries, rows, common_count, only_baseline, only_optimized,
                      cat_high, cat_mid, cat_low, cat_na, top10, bottom10):
    """Write the summary statistics CSV."""

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Section 1: Overall statistics
        writer.writerow(["=== 总体统计 ==="])
        writer.writerow(["总算子数", len(op_summaries)])
        writer.writerow(["总输出行数", len(rows)])
        writer.writerow(["可对比case数", common_count])
        writer.writerow(["仅baseline有", len(only_baseline)])
        writer.writerow(["仅optimized有", len(only_optimized)])
        writer.writerow([])

        # Section 2: Category breakdown
        writer.writerow(["=== 加速比分类统计（按算子平均加速比） ==="])
        writer.writerow(["分类", "算子数"])

        categories = [
            ("平均加速比 >= 1.2x（显著提升）", cat_high),
            ("平均加速比 0.9x ~ 1.2x（持平）", cat_mid),
            ("平均加速比 < 0.9x（性能下降）", cat_low),
            ("无法计算", cat_na),
        ]

        for cat_name, cat_list in categories:
            writer.writerow([cat_name, len(cat_list)])

        writer.writerow([])

        # Detail for each category
        for cat_name, cat_list in categories:
            if cat_list:
                writer.writerow([f"--- {cat_name} ---"])
                writer.writerow(["算子", "平均加速比", "case数", "各dtype加速比"])
                for s in cat_list:
                    speedup_str = f"{s['avg_speedup']:.2f}x" if s['avg_speedup'] is not None else 'N/A'
                    dtype_detail = ', '.join([f"{d}: {v:.2f}x" for d, v in s['type_avgs']])
                    writer.writerow([s['op'], speedup_str, s['case_count'], dtype_detail])
                writer.writerow([])

        # Section 3: Top-10
        writer.writerow(["=== 加速比最高 Top-10 算子 ==="])
        writer.writerow(["排名", "算子", "平均加速比", "case数"])
        for rank, s in enumerate(top10, 1):
            writer.writerow([rank, s['op'], f"{s['avg_speedup']:.2f}x", s['case_count']])
        writer.writerow([])

        # Section 4: Bottom-10
        writer.writerow(["=== 加速比最低 Bottom-10 算子 ==="])
        writer.writerow(["排名", "算子", "平均加速比", "case数"])
        for rank, s in enumerate(bottom10, 1):
            writer.writerow([rank, s['op'], f"{s['avg_speedup']:.2f}x", s['case_count']])

    print(f"Summary CSV saved to: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 bench_compare.py <baseline_csv> <optimized_csv> [output_csv_prefix]")
        print("Example:")
        print("  python3 bench_compare.py baseline/bench_summary.csv optimized/bench_summary.csv result")
        print("  (produces: result_detail.csv, result_summary.csv)")
        sys.exit(1)

    baseline_path = sys.argv[1]
    optimized_path = sys.argv[2]

    # Default output prefix
    if len(sys.argv) >= 4:
        output_prefix = sys.argv[3]
        # Strip .csv or .xlsx suffix if user provided one
        if output_prefix.endswith('.csv'):
            output_prefix = output_prefix[:-4]
        elif output_prefix.endswith('.xlsx'):
            output_prefix = output_prefix[:-5]
    else:
        output_prefix = "bench_compare_result"

    detail_output = f"{output_prefix}_detail.csv"
    summary_output = f"{output_prefix}_summary.csv"

    # Validate input files
    for p in [baseline_path, optimized_path]:
        if not os.path.exists(p):
            print(f"Error: File not found: {p}")
            sys.exit(1)

    print(f"Baseline      : {baseline_path}")
    print(f"Optimized     : {optimized_path}")
    print(f"Output detail : {detail_output}")
    print(f"Output summary: {summary_output}")
    print()

    # Parse both CSVs
    print("Parsing CSV files...")
    baseline_list, baseline_dict = parse_csv(baseline_path)
    optimized_list, optimized_dict = parse_csv(optimized_path)

    print(f"  Baseline : {len(baseline_list)} test cases")
    print(f"  Optimized: {len(optimized_list)} test cases")

    # ============================================================
    # Build rows in baseline order
    # ============================================================
    # Track which optimized indices have been matched
    matched_optimized = set()

    rows = []
    only_baseline_keys = []
    common_count = 0

    for b_entry in baseline_list:
        op_clean = b_entry['op_clean']
        dtype = b_entry['dtype']
        b_shape = b_entry['shape']
        shape_for_key = b_entry['shape_for_key']

        key = (op_clean, dtype, shape_for_key)

        # Find the first unmatched optimized entry with the same key
        o_entry = None
        if key in optimized_dict:
            for o_idx in optimized_dict[key]:
                if o_idx not in matched_optimized:
                    o_entry = optimized_list[o_idx]
                    matched_optimized.add(o_idx)
                    break

        b_lat = b_entry['lat']
        o_lat = o_entry['lat'] if o_entry else None
        b_op_raw = b_entry['op_raw']
        o_op_raw = o_entry['op_raw'] if o_entry else ''
        o_shape = o_entry['shape'] if o_entry else ''

        speedup = compute_speedup(b_lat, o_lat)

        if o_entry is not None:
            common_count += 1
        else:
            only_baseline_keys.append(key)

        rows.append({
            'op_clean': op_clean,
            'dtype': dtype,
            'shape': b_shape,
            'o_shape': o_shape,
            'b_op_raw': b_op_raw,
            'b_lat': b_lat,
            'o_op_raw': o_op_raw,
            'o_lat': o_lat,
            'speedup': speedup,
        })

    # Collect only-optimized entries (unmatched)
    only_optimized_keys = []
    for o_idx, o_entry in enumerate(optimized_list):
        if o_idx not in matched_optimized:
            op_clean = o_entry['op_clean']
            dtype = o_entry['dtype']
            shape = o_entry['shape']
            shape_for_key = o_entry['shape_for_key']

            key = (op_clean, dtype, shape_for_key)
            only_optimized_keys.append(key)

            rows.append({
                'op_clean': op_clean,
                'dtype': dtype,
                'shape': '',
                'o_shape': shape,
                'b_op_raw': '',
                'b_lat': None,
                'o_op_raw': o_entry['op_raw'],
                'o_lat': o_entry['lat'],
                'speedup': None,
            })

    print(f"  Common (paired)   : {common_count}")
    print(f"  Only in baseline  : {len(only_baseline_keys)}")
    if only_baseline_keys:
        for k in only_baseline_keys:
            print(f"    - {k}")
    print(f"  Only in optimized : {len(only_optimized_keys)}")
    if only_optimized_keys:
        for k in only_optimized_keys:
            print(f"    - {k}")
    print()

    # ============================================================
    # Compute group averages
    # ============================================================
    # Type-level: group by (op_clean, dtype)
    type_groups = defaultdict(list)
    for i, row in enumerate(rows):
        type_groups[(row['op_clean'], row['dtype'])].append(i)

    type_avg_map = {}
    for group_key, indices in type_groups.items():
        speedups = [rows[i]['speedup'] for i in indices if rows[i]['speedup'] is not None]
        if speedups:
            type_avg_map[group_key] = sum(speedups) / len(speedups)
        else:
            type_avg_map[group_key] = None

    # Op-level: group by op_clean
    op_groups = defaultdict(list)
    for i, row in enumerate(rows):
        op_groups[row['op_clean']].append(i)

    op_avg_map = {}
    for op_name, indices in op_groups.items():
        speedups = [rows[i]['speedup'] for i in indices if rows[i]['speedup'] is not None]
        if speedups:
            op_avg_map[op_name] = sum(speedups) / len(speedups)
        else:
            op_avg_map[op_name] = None

    # Store group info per row
    for i, row in enumerate(rows):
        row['type_avg'] = type_avg_map.get((row['op_clean'], row['dtype']))
        row['op_avg'] = op_avg_map.get(row['op_clean'])

    # ============================================================
    # Write Detailed CSV
    # ============================================================
    print("Writing detail CSV file...")
    write_detailed_csv(detail_output, rows)

    # ============================================================
    # Build op-level summary for statistics
    # ============================================================
    op_summaries = []
    for op_name in sorted(op_avg_map.keys()):
        avg = op_avg_map[op_name]
        type_avgs = []
        case_count = 0
        for (o, d), tidx in type_groups.items():
            if o == op_name:
                ta = type_avg_map.get((o, d))
                if ta is not None:
                    type_avgs.append((d, ta))
                case_count += len(tidx)

        if avg is not None:
            if avg >= 1.2:
                category = "显著提升 (>=1.2x)"
            elif avg >= 0.9:
                category = "持平 (0.9x~1.2x)"
            else:
                category = "性能下降 (<0.9x)"
        else:
            category = "无法计算"

        op_summaries.append({
            'op': op_name,
            'avg_speedup': avg,
            'category': category,
            'case_count': case_count,
            'type_avgs': type_avgs,
        })

    # Count categories
    cat_high = [s for s in op_summaries if s['avg_speedup'] is not None and s['avg_speedup'] >= 1.2]
    cat_mid = [s for s in op_summaries if s['avg_speedup'] is not None and 0.9 <= s['avg_speedup'] < 1.2]
    cat_low = [s for s in op_summaries if s['avg_speedup'] is not None and s['avg_speedup'] < 0.9]
    cat_na = [s for s in op_summaries if s['avg_speedup'] is None]

    # Top & Bottom
    top10 = sorted([s for s in op_summaries if s['avg_speedup'] is not None],
                   key=lambda x: x['avg_speedup'], reverse=True)[:10]
    bottom10 = sorted([s for s in op_summaries if s['avg_speedup'] is not None],
                      key=lambda x: x['avg_speedup'])[:10]

    # ============================================================
    # Write Summary CSV
    # ============================================================
    print("Writing summary CSV file...")
    write_summary_csv(summary_output, op_summaries, rows, common_count,
                      only_baseline_keys, only_optimized_keys,
                      cat_high, cat_mid, cat_low, cat_na,
                      top10, bottom10)

    # ============================================================
    # Print terminal summary
    # ============================================================
    print("\n" + "=" * 80)
    print("统计结论")
    print("=" * 80)

    print(f"\n总算子数: {len(op_summaries)}")
    print(f"总输出行数: {len(rows)}")
    print(f"可对比case数: {common_count}")
    print(f"仅baseline有: {len(only_baseline_keys)}")
    print(f"仅optimized有: {len(only_optimized_keys)}")

    print(f"\n--- 按算子平均加速比分类 ---")
    print(f"平均加速比 >= 1.2x（显著提升）: {len(cat_high)} 个算子")
    if cat_high:
        for s in cat_high:
            print(f"  - {s['op']}: {s['avg_speedup']:.2f}x")

    print(f"\n平均加速比 0.9x ~ 1.2x（持平）: {len(cat_mid)} 个算子")
    if cat_mid:
        for s in cat_mid:
            print(f"  - {s['op']}: {s['avg_speedup']:.2f}x")

    print(f"\n平均加速比 < 0.9x（性能下降）: {len(cat_low)} 个算子")
    if cat_low:
        for s in cat_low:
            print(f"  - {s['op']}: {s['avg_speedup']:.2f}x")

    if cat_na:
        print(f"\n无法计算: {len(cat_na)} 个算子")
        for s in cat_na:
            print(f"  - {s['op']}")

    print(f"\n--- Top-5 加速比最高 ---")
    for s in top10[:5]:
        print(f"  {s['op']}: {s['avg_speedup']:.2f}x")

    print(f"\n--- Bottom-5 加速比最低 ---")
    for s in bottom10[:5]:
        print(f"  {s['op']}: {s['avg_speedup']:.2f}x")

    print("\nDone!")


if __name__ == '__main__':
    main()
