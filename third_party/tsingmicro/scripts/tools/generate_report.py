#!/usr/bin/env python3
"""
解析 flaggems perf test 日志，生成 Markdown / Excel 报告。

==== 使用方式 ====

  # 单目录 → 单报告 (每个 .log 文件一个测试用例)
  python tools/generate_report.py <dir>

  # 双目录 → 对比报告 (左右对照, 带汇总分类)
  python tools/generate_report.py <dir0> <dir1>

  # 单日志文件 → 兼容旧批跑格式 (需要时间戳匹配)
  python tools/generate_report.py <file.log>

==== 环境要求 ====

  - Python 3.8+
  - 无外部依赖 (仅标准库, Excel 需要 openpyxl)

==== 日志格式 ====

  输入日志需包含:
  1. pytest 输出: perf_test.py::test_accuracy_XXX[...] + PASSED/FAILED
  2. tsingmicro launch 日志: tsingmicro_launch:KERNEL_NAME launch card:X begin
  3. HPGR 计时记录: HPGR: record[...]: kcore_dur_time:...ns

==== 输出字段 ====

  单报告: 算子 | 参数 | 状态 | kcore_dur_time 合计 | ap_dur_time 合计
         | kernel 名称 | 调用次数 | kcore_dur_time 单次 | ap_dur_time 单次 | ap_start_time

  对比报告: 算子 | 参数 | dir0状态 | dir0kcore | dir1状态 | dir1kcore | 加速比
           | kernel 名称 | dir0次数 | dir0kcore | dir1次数 | dir1kcore
"""

import re
import sys
import os
import glob
from datetime import datetime
from collections import OrderedDict

# ============================================================
#  分类阈值 (可手动修改)
# ============================================================
# 加速比 = dir0 kcore / dir1 kcore
BIG_WIN = 2.0  # 大幅优化:  speedup >= BIG_WIN
MEDIUM_WIN = 1.2  # 中等优化:  MEDIUM_WIN <= speedup < BIG_WIN
REGRESSION = 0.9  # 负优化:    speedup < REGRESSION
# 不明显:    REGRESSION <= speedup < MEDIUM_WIN


def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y%m%d %H:%M:%S.%f")


def format_ns(ns):
    if ns >= 1_000_000_000:
        return f"{ns/1_000_000_000:.3f} s"
    elif ns >= 1_000_000:
        return f"{ns/1_000_000:.3f} ms"
    elif ns >= 1_000:
        return f"{ns/1_000:.1f} us"
    else:
        return f"{ns} ns"


def parse_single_file(filepath):
    """Parse a single-test log file. Returns one test dict or None."""
    with open(filepath, 'r') as f:
        content = f.read()
    clean = strip_ansi(content)
    lines = clean.split('\n')

    time_pat = re.compile(r'\[(\d{8}\s+\d{2}:\d{2}:\d{2}\.\d{3})\]')
    test_pat = re.compile(r'perf_test\.py::(test_accuracy_\S+)')
    begin_pat = re.compile(r'tsingmicro_launch:(\S+)\s+launch\s+card:\d+\s+begin')
    hpg_pat = re.compile(
        r'HPGR:\s*record\[.*?\]:\s*ap_start_time:(\d+)ns,ap_dur_time:(\d+)ns,kcore_start_time:\d+ns,kcore_dur_time:(\d+)ns'
    )
    passed_pat = re.compile(r'^\s*(PASSED|FAILED)\s*$')

    test_full = None
    status = 'UNKNOWN'
    launch_events = []  # (kernel_name,)
    hpg_events = []  # (ap_start, ap_dur, kcore_dur)

    for line in lines:
        tm = test_pat.search(line)
        if tm and test_full is None:
            test_full = tm.group(1)

        bm = begin_pat.search(line)
        if bm:
            launch_events.append(bm.group(1))

        hm = hpg_pat.search(line)
        if hm:
            hpg_events.append((int(hm.group(1)), int(hm.group(2)), int(hm.group(3))))

        pm = passed_pat.match(line.strip())
        if pm:
            status = pm.group(1)

    if test_full is None or not launch_events:
        return None

    # Extract op_name and params
    m = re.match(r'(test_accuracy_)?(.+?)\[(.+)\]', test_full)
    if m:
        op_name = m.group(2)
        params = m.group(3)
    else:
        op_name = test_full
        params = ""

    # Match HPGR to launches by sequential order
    kernels = OrderedDict()
    for i, kname in enumerate(launch_events):
        if kname not in kernels:
            kernels[kname] = {'count': 0, 'kcore_times': [], 'ap_times': [], 'ap_start_times': []}
        kernels[kname]['count'] += 1
        if i < len(hpg_events):
            ap_s, ap_d, kc_d = hpg_events[i]
            kernels[kname]['kcore_times'].append(kc_d)
            kernels[kname]['ap_times'].append(ap_d)
            kernels[kname]['ap_start_times'].append(ap_s)

    return {
        'test_full': test_full,
        'op_name': op_name,
        'params_display': params,
        'kernels': kernels,
        'status': status,
    }


def parse_batch_log(filepath):
    """Parse a batch log (multiple tests in one file). Uses timestamp matching."""
    with open(filepath, 'r') as f:
        content = f.read()
    clean = strip_ansi(content)
    lines = clean.split('\n')

    time_pat = re.compile(r'\[(\d{8}\s+\d{2}:\d{2}:\d{2}\.\d{3})\]')
    test_pat = re.compile(r'perf_test\.py::(test_accuracy_\S+)')
    begin_pat = re.compile(r'tsingmicro_launch:(\S+)\s+launch\s+card:\d+\s+begin')
    hpg_pat = re.compile(
        r'HPGR:\s*record\[.*?\]:\s*ap_start_time:(\d+)ns,ap_dur_time:(\d+)ns,kcore_start_time:\d+ns,kcore_dur_time:(\d+)ns'
    )
    passed_pat = re.compile(r'^\s*(PASSED|FAILED)\s*$')

    # Collect global events
    test_events = []
    launch_events = []
    hpg_events = []
    pass_events = []

    for line in lines:
        tsm = time_pat.search(line)
        ts = parse_ts(tsm.group(1)) if tsm else None

        tm = test_pat.search(line)
        if tm and ts:
            test_events.append((ts, tm.group(1)))

        bm = begin_pat.search(line)
        if bm and ts:
            launch_events.append((ts, bm.group(1)))

        hm = hpg_pat.search(line)
        if hm and ts:
            hpg_events.append((ts, int(hm.group(1)), int(hm.group(2)), int(hm.group(3))))

        pm = passed_pat.match(line.strip())
        if pm:
            pass_events.append(pm.group(1))

    # Match HPGR to launches by time proximity
    hpg_matched = [False] * len(hpg_events)
    launch_hpg = [None] * len(launch_events)

    for li, (lts, lname) in enumerate(launch_events):
        best = None
        best_diff = None
        for hi, (hts, ap_s, ap_d, kc_d) in enumerate(hpg_events):
            if hpg_matched[hi]:
                continue
            diff = abs((hts - lts).total_seconds()) if hts and lts else 999
            if diff < 10.0:
                if best is None or diff < best_diff:
                    best = hi
                    best_diff = diff
        if best is not None:
            hpg_matched[best] = True
            launch_hpg[li] = hpg_events[best][1:]

    # Assign to tests by timestamp
    tests = []
    for ti, (tts, test_full) in enumerate(test_events):
        next_tts = test_events[ti + 1][0] if ti + 1 < len(test_events) else datetime.max

        m = re.match(r'(test_accuracy_)?(.+?)\[(.+)\]', test_full)
        op_name = m.group(2) if m else test_full
        params = m.group(3) if m else ""

        kernels = OrderedDict()
        for li, (lts, lname) in enumerate(launch_events):
            if tts <= lts < next_tts:
                if lname not in kernels:
                    kernels[lname] = {'count': 0, 'kcore_times': [], 'ap_times': [], 'ap_start_times': []}
                kernels[lname]['count'] += 1
                if launch_hpg[li] is not None:
                    ap_s, ap_d, kc_d = launch_hpg[li]
                    kernels[lname]['kcore_times'].append(kc_d)
                    kernels[lname]['ap_times'].append(ap_d)
                    kernels[lname]['ap_start_times'].append(ap_s)

        status = pass_events[ti] if ti < len(pass_events) else 'UNKNOWN'

        tests.append({
            'test_full': test_full,
            'op_name': op_name,
            'params_display': params,
            'kernels': kernels,
            'status': status,
        })

    return tests


def load_tests(path):
    """Load tests from a directory of single-test logs or a batch log file."""
    if os.path.isdir(path):
        tests = []
        for f in sorted(glob.glob(os.path.join(path, '*.log'))):
            t = parse_single_file(f)
            if t:
                tests.append(t)
        print(f"  Loaded {len(tests)} tests from {path}")
        return tests
    else:
        tests = parse_batch_log(path)
        print(f"  Loaded {len(tests)} tests from {path}")
        return tests


def test_kcore(t):
    return sum(sum(k['kcore_times']) for k in t['kernels'].values())


def test_ap(t):
    return sum(sum(k['ap_times']) for k in t['kernels'].values())


# ============================================================
#  Markdown
# ============================================================


def md_single(tests, title):
    lines = [f"# {title}", "", f"共 {len(tests)} 个测试用例", ""]
    lines.append(
        "| 算子 | 参数 | 状态 | kcore_dur_time 合计 | ap_dur_time 合计 | kernel 名称 | 调用次数 | kcore_dur_time 单次 | ap_dur_time 单次 | ap_start_time |"
    )
    lines.append(
        "|------|------|------|-------------------|----------------|------------|---------|-------------------|----------------|--------------|"
    )

    for t in tests:
        op, params, status = t['op_name'], t['params_display'], t['status']
        kc, ap = test_kcore(t), test_ap(t)
        sm = f"`{status}`"
        if not t['kernels']:
            lines.append(f"| {op} | {params} | {sm} | {format_ns(kc)} | {format_ns(ap)} | — | — | — | — | — |")
        else:
            first = True
            for kname, kd in t['kernels'].items():
                kl = ", ".join(format_ns(x) for x in kd['kcore_times'])
                al = ", ".join(format_ns(x) for x in kd['ap_times'])
                sl = ", ".join(f"{x/1e9:.6f}s" for x in kd['ap_start_times'])
                if first:
                    lines.append(
                        f"| {op} | {params} | {sm} | {format_ns(kc)} | {format_ns(ap)} | `{kname}` | {kd['count']} | {kl} | {al} | {sl} |"
                    )
                    first = False
                else:
                    lines.append(f"| | | | | | `{kname}` | {kd['count']} | {kl} | {al} | {sl} |")
    lines.append("")
    return "\n".join(lines)


def md_comparison(tests0, tests1, label0, label1):
    lines = [f"# 对比: {label0} vs {label1}", ""]
    idx0 = {(t['op_name'], t['params_display']): t for t in tests0}
    idx1 = {(t['op_name'], t['params_display']): t for t in tests1}
    all_keys = sorted(set(list(idx0.keys()) + list(idx1.keys())))

    categories = {"大幅优化": [], "中等优化": [], "不明显": [], "负优化": []}
    for key in all_keys:
        t0, t1 = idx0.get(key), idx1.get(key)
        kc0 = test_kcore(t0) if t0 else 0
        kc1 = test_kcore(t1) if t1 else 0
        if kc0 > 0 and kc1 > 0:
            speedup = kc0 / kc1
            entry = (key[0], key[1], kc0, kc1, speedup)
            if speedup >= BIG_WIN: categories["大幅优化"].append(entry)
            elif speedup >= MEDIUM_WIN: categories["中等优化"].append(entry)
            elif speedup >= REGRESSION: categories["不明显"].append(entry)
            else: categories["负优化"].append(entry)

    cat_criteria = {
        "大幅优化": f"加速比 >= {BIG_WIN}",
        "中等优化": f"{MEDIUM_WIN} <= 加速比 < {BIG_WIN}",
        "不明显": f"{REGRESSION} <= 加速比 < {MEDIUM_WIN}",
        "负优化": f"加速比 < {REGRESSION}",
    }

    lines.append("## 汇总")
    lines.append("")
    lines.append(f"| 类别 | 数量 | 说明 |")
    lines.append(f"|------|------|------|")
    for cat, entries in categories.items():
        lines.append(f"| {cat} | {len(entries)} | {cat_criteria[cat]} |")
    lines.append("")

    for cat, entries in categories.items():
        if not entries: continue
        lines.append(f"### {cat} ({len(entries)})")
        lines.append("")
        lines.append(f"| 算子 | 参数 | {label0} kcore | {label1} kcore | 加速比 (old/new) |")
        lines.append(f"|------|------|-------------|-------------|-----------------|")
        for op, params, kc0, kc1, sp in sorted(entries, key=lambda x: -x[4]):
            lines.append(f"| {op} | {params} | {format_ns(kc0)} | {format_ns(kc1)} | {sp:.2f}x |")
        lines.append("")

    lines.append("## 明细")
    lines.append("")
    lines.append(
        f"| 算子 | 参数 | {label0} 状态 | {label0} kcore | {label1} 状态 | {label1} kcore | 加速比 | kernel 名称 | {label0} 次数 | {label0} kcore | {label1} 次数 | {label1} kcore |"
    )
    lines.append(
        f"|------|------|-------------|-----------|-------------|-----------|-------|------------|-------------|-------------|-------------|-------------|"
    )

    for key in all_keys:
        op, params = key
        t0, t1 = idx0.get(key), idx1.get(key)
        kc0 = test_kcore(t0) if t0 else 0
        kc1 = test_kcore(t1) if t1 else 0
        s0 = t0['status'] if t0 else 'N/A'
        s1 = t1['status'] if t1 else 'N/A'
        sp = f"{kc0/kc1:.2f}x" if kc0 > 0 and kc1 > 0 else "—"
        k0n = set(t0['kernels'].keys()) if t0 else set()
        k1n = set(t1['kernels'].keys()) if t1 else set()
        all_knames = list(OrderedDict.fromkeys(sorted(k0n | k1n)))
        if not all_knames:
            lines.append(
                f"| {op} | {params} | `{s0}` | {format_ns(kc0)} | `{s1}` | {format_ns(kc1)} | {sp} | — | — | — | — | — |"
            )
        else:
            first = True
            for kname in all_knames:
                kd0 = t0['kernels'].get(kname) if t0 else None
                kd1 = t1['kernels'].get(kname) if t1 else None
                c0, c1 = kd0['count'] if kd0 else 0, kd1['count'] if kd1 else 0
                k0 = format_ns(sum(kd0['kcore_times'])) if kd0 else "—"
                k1 = format_ns(sum(kd1['kcore_times'])) if kd1 else "—"
                if first:
                    lines.append(
                        f"| {op} | {params} | `{s0}` | {format_ns(kc0)} | `{s1}` | {format_ns(kc1)} | {sp} | `{kname}` | {c0} | {k0} | {c1} | {k1} |"
                    )
                    first = False
                else:
                    lines.append(f"| | | | | | | | `{kname}` | {c0} | {k0} | {c1} | {k1} |")
    lines.append("")
    return "\n".join(lines)


# ============================================================
#  Excel
# ============================================================


def excel_comparison(tests0, tests1, label0, label1, outpath):
    import openpyxl
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "perf对比"

    headers = [
        "算子", "参数", f"{label0}_状态", f"{label0}_kcore(ns)", f"{label0}_kcore", f"{label1}_状态", f"{label1}_kcore(ns)",
        f"{label1}_kcore", "加速比", "kernel名称", f"{label0}_次数", f"{label0}_kernel_kcore(ns)", f"{label0}_kernel_kcore",
        f"{label1}_次数", f"{label1}_kernel_kcore(ns)", f"{label1}_kernel_kcore"
    ]
    for ci, h in enumerate(headers, 1):
        ws.cell(row=1, column=ci, value=h).font = Font(bold=True)

    idx0 = {(t['op_name'], t['params_display']): t for t in tests0}
    idx1 = {(t['op_name'], t['params_display']): t for t in tests1}
    all_keys = sorted(set(list(idx0.keys()) + list(idx1.keys())))

    row = 2
    for key in all_keys:
        op, params = key
        t0, t1 = idx0.get(key), idx1.get(key)
        kc0 = test_kcore(t0) if t0 else 0
        kc1 = test_kcore(t1) if t1 else 0
        s0 = t0['status'] if t0 else 'N/A'
        s1 = t1['status'] if t1 else 'N/A'
        speedup = kc0 / kc1 if kc0 > 0 and kc1 > 0 else None
        k0n = set(t0['kernels'].keys()) if t0 else set()
        k1n = set(t1['kernels'].keys()) if t1 else set()
        all_knames = list(OrderedDict.fromkeys(sorted(k0n | k1n)))

        if not all_knames:
            ws.cell(row=row, column=1, value=op)
            ws.cell(row=row, column=2, value=params)
            ws.cell(row=row, column=3, value=s0)
            ws.cell(row=row, column=4, value=kc0)
            ws.cell(row=row, column=5, value=format_ns(kc0))
            ws.cell(row=row, column=6, value=s1)
            ws.cell(row=row, column=7, value=kc1)
            ws.cell(row=row, column=8, value=format_ns(kc1))
            if speedup is not None:
                ws.cell(row=row, column=9, value=round(speedup, 2))
            row += 1
        else:
            first = True
            for kname in all_knames:
                kd0 = t0['kernels'].get(kname) if t0 else None
                kd1 = t1['kernels'].get(kname) if t1 else None
                c0 = kd0['count'] if kd0 else 0
                c1 = kd1['count'] if kd1 else 0
                k0_sum = sum(kd0['kcore_times']) if kd0 else 0
                k1_sum = sum(kd1['kcore_times']) if kd1 else 0
                if first:
                    ws.cell(row=row, column=1, value=op)
                    ws.cell(row=row, column=2, value=params)
                    ws.cell(row=row, column=3, value=s0)
                    ws.cell(row=row, column=4, value=kc0)
                    ws.cell(row=row, column=5, value=format_ns(kc0))
                    ws.cell(row=row, column=6, value=s1)
                    ws.cell(row=row, column=7, value=kc1)
                    ws.cell(row=row, column=8, value=format_ns(kc1))
                    if speedup is not None:
                        ws.cell(row=row, column=9, value=round(speedup, 2))
                    first = False
                ws.cell(row=row, column=10, value=kname)
                ws.cell(row=row, column=11, value=c0)
                ws.cell(row=row, column=12, value=k0_sum)
                ws.cell(row=row, column=13, value=format_ns(k0_sum) if kd0 else "—")
                ws.cell(row=row, column=14, value=c1)
                ws.cell(row=row, column=15, value=k1_sum)
                ws.cell(row=row, column=16, value=format_ns(k1_sum) if kd1 else "—")
                row += 1

    for ci in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(ci)].width = 22
    wb.save(outpath)
    return outpath


# ============================================================
#  Main
# ============================================================


def main():
    args = [a for a in sys.argv[1:] if a != '--html']

    if len(args) >= 2:
        path0, path1 = args[0], args[1]
        label0 = os.path.basename(path0.rstrip('/'))
        label1 = os.path.basename(path1.rstrip('/'))
        tests0 = load_tests(path0)
        tests1 = load_tests(path1)
        md = md_comparison(tests0, tests1, label0, label1)
        out_md = os.path.join(os.path.dirname(path0.rstrip('/')), f"{label0}_vs_{label1}.md")
        with open(out_md, 'w') as f:
            f.write(md)
        print(f"Markdown: {out_md}  ({len(tests0)} vs {len(tests1)} tests)")
        out_xlsx = out_md.replace('.md', '.xlsx')
        excel_comparison(tests0, tests1, label0, label1, out_xlsx)
        print(f"Excel: {out_xlsx}")
    elif len(args) >= 1:
        path = args[0]
        tests = load_tests(path)
        title = os.path.basename(path.rstrip('/'))
        if os.path.isdir(path):
            out_md = os.path.join(os.path.dirname(path.rstrip('/')), f"{title}.md")
        else:
            out_md = path.replace('.log', '.md')
        md = md_single(tests, title)
        with open(out_md, 'w') as f:
            f.write(md)
        print(f"Markdown: {out_md}  ({len(tests)} tests)")
    else:
        print("Usage: python generate_report.py <dir|log> [<dir2|log2>]")
        sys.exit(1)


if __name__ == '__main__':
    main()
