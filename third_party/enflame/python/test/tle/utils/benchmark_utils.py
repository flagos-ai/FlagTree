import os
import shutil

import pandas as pd


def validate_benchmark(
    current_csv,
    baseline_csv,
    dimension_fields,
    regression_threshold=0.20,
):
    if not os.path.exists(baseline_csv):
        print(f"Baseline CSV '{baseline_csv}' not found. Saving current results as baseline.")
        shutil.copy2(current_csv, baseline_csv)
        return True
    current = pd.read_csv(current_csv)
    baseline = pd.read_csv(baseline_csv)
    passed = True
    for _, cur_row in current.iterrows():
        match_cond = True
        for dim in dimension_fields:
            match_cond &= (baseline[dim] == cur_row[dim])
        base_rows = baseline[match_cond]
        if len(base_rows) == 0:
            print(f"  WARN  no baseline match for {dict((d, cur_row[d]) for d in dimension_fields)}")
            continue
        base_row = base_rows.iloc[0]
        perf_cols = [c for c in current.columns if c not in dimension_fields]
        for col in perf_cols:
            cur_val = cur_row[col]
            base_val = base_row[col]
            if pd.isna(cur_val) or pd.isna(base_val):
                continue
            ratio = cur_val / base_val
            if ratio > (1.0 + regression_threshold):
                print(f"  FAIL  {dict((d, cur_row[d]) for d in dimension_fields)} {col}: "
                      f"current={cur_val:.6f} baseline={base_val:.6f} ratio={ratio:.2f} "
                      f"(threshold={1.0 + regression_threshold:.2f})")
                passed = False
            else:
                print(f"  PASS  {dict((d, cur_row[d]) for d in dimension_fields)} {col}: "
                      f"current={cur_val:.6f} baseline={base_val:.6f} ratio={ratio:.2f}")
    return passed
