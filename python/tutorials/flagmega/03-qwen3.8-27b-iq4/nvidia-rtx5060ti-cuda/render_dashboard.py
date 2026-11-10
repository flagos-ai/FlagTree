#!/usr/bin/env python3
"""Render a small, portable status dashboard; Python standard library only.

Update after the orchestrator has recorded a state transition:
  python render_dashboard.py --run .local/runs/<run-id>
Rebuild using only the public snapshot (no run/model/SDK access):
  python render_dashboard.py --snapshot .local/results/dashboard-status.json
Both modes default to dashboard.html beside this script. --run also writes
.local/results/dashboard-status.json there. Output paths can be overridden explicitly.
Published links are relative to the delivery root containing dashboard.html.
"""

import argparse
import hashlib
import html
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "flagmega.dashboard-status/v1"
OBJECTIVE = "在原始验收条件下，让 FlagMega C++ 解码在两个场景均超过 llama.cpp"
TRANSLATIONS = {
    "Chinese dashboard implementation: completed/current/next tasks and measured status": "制作中文进度页：已完成、正在做、接下来做与实测状态",
    "Dashboard CPU implementation and review complete; workers quiet before device execution": "等待 dashboard 的 CPU 实现与审查完成，并确认各工作进程安静后，才能执行设备任务",
    "head-coarse protected GPU window": "head-coarse 受保护 GPU 窗口",
    "head-coarse D/M/F": "head-coarse 的独立 D / M / F 诊断对照",
    "Remaining9 declared microprofile scopes": "剩余 9 个已声明的微观诊断范围",
    "Strict original matched llama acceptance in both scenarios with2 processes x3 rounds": "双方、双场景按原条件配对验收：各 2 个独立进程 × 3 轮，均严格低于 llama.cpp",
    "Strict original matched llama acceptance in both scenarios with 2 processes x 3 rounds": "双方、双场景按原条件配对验收：各 2 个独立进程 × 3 轮，均严格低于 llama.cpp",
    "Seven original head matrix scopes: head-i0-j1, head-i0-j2, head-i0-j3, head-i1-j0, head-i1-j1, head-i1-j2, head-i1-j3": "剩余 7 个 head 矩阵诊断范围：head-i0-j1、head-i0-j2、head-i0-j3、head-i1-j0、head-i1-j1、head-i1-j2、head-i1-j3",
    "Original GGUF public reconstruction": "完成原始 GGUF 的公开重建与复现门槛",
    "Current t0013 mandatory microprofile; t0012 scope coverage is not transferred": "完成当前 t0013 内核的 NCU、全 warp 流水、硬件活动、时钟与扰动分析及泳道图；历史 t0012 证据另行保留",
    "Strict original matched llama acceptance: both scenarios, both backends, 2 processes ×3 rounds": "双方、双场景按原条件配对验收：各 2 个独立进程 × 3 轮，均严格低于 llama.cpp",
    "Original-GGUF public reconstruction": "完成原始 GGUF 的公开重建与复现门槛",
}
ROLE_NAMES = {
    "root": "Root · 统筹 / 设备调度",
    "optimizer": "Optimizer · 实现专家",
    "kernel_optimizer": "Kernel Optimizer · 内核专家",
    "kernel_review": "Kernel Review · 独立审查",
    "validator": "Validator · 验证 / 发布",
    "architect": "Architect · 架构决策",
    "graph_architect": "Graph Architect · 图语义设计",
    "graph_optimizer": "Graph Optimizer · 图优化",
    "graph_review": "Graph Review · 图语义审查",
    "vector_architect": "Vector Architect · 数据表示设计",
    "vector_optimizer": "Vector Optimizer · 向量化优化",
    "vector_review": "Vector Review · 数据表示审查",
    "distribution_architect": "Distribution Architect · 分布与内存设计",
    "distribution_optimizer": "Distribution Optimizer · 分布与内存优化",
    "distribution_review": "Distribution Review · 分布与内存审查",
    "kernel_architect": "Kernel Architect · 内核设计",
}
DOMAIN_DUTIES = {
    "Graph": "图语义、融合与数值边界",
    "Vector": "向量化、数据表示与布局",
    "Distribution": "任务划分、数据归属与同步",
    "Kernel": "内核实现、NCU、WG 流水与硬件活动分析",
}
PRIVATE_TEXT = re.compile(
    r"\.local(?:/|\\)|(?:/home/|/mnt/|/tmp/|/root/)|\.env\b|"
    r"\b(?:argv|environ(?:ment)?|password|secret|api[_-]?key|access[_-]?token)\b|"
    r"[A-Z][A-Z0-9_]{2,}=|(?:https?|file)://|(?:\$\(|`)", re.I
)


def public_text(value, limit=700):
    """Select narrative strings, never serialize a source dictionary or command."""
    if not isinstance(value, str):
        return ""
    if PRIVATE_TEXT.search(value):
        return "内部细节已省略；请查受控运行记录"
    return TRANSLATIONS.get(value, value)[:limit]


def identifier(value):
    value = str(value or "").removeprefix("/root/").removeprefix("agents/")
    return value if re.fullmatch(r"[A-Za-z0-9_.:/+ -]{1,160}", value) and not value.startswith("/") else "未声明"


def href(value):
    if value == "optimization-trace.md":
        return value
    p = PurePosixPath(str(value))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Unsafe public link")
    final_figures = {"performance-vs-iteration.svg", "performance-comparison.svg"}
    if p.parts[0] == "results" or (p.parts[0] == "figures" and p.name not in final_figures):
        p = PurePosixPath(".local") / p
    if (not re.fullmatch(r"[A-Za-z0-9_./-]+", str(p))
            or not (p.parts[0] == "figures" or p.parts[:2] in
                    ((".local", "results"), (".local", "figures")))):
        raise ValueError("Links must target final figures or local result/figure archives")
    return str(p)


def relocated_evidence_path(value, delivery):
    """Resolve original publication pins after the byte-preserving archive move."""
    path = Path(value)
    try:
        relative = path.relative_to(delivery)
    except ValueError:
        return path
    if relative.parts and relative.parts[0] in {"results", "figures"}:
        return delivery / ".local" / relative
    return path


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publication_id(state, key):
    entry = state.get(key) or {}
    if entry.get("status") != "published_verified":
        raise ValueError(f"{key} is not a verified publication")
    export_id = entry.get("export_id")
    if not export_id:
        export_id = Path((entry.get("export") or {}).get("path", "")).parent.name
    if not re.fullmatch(r"e\d{4,}", str(export_id)):
        raise ValueError("Invalid publication ID")
    return export_id


def pending_gate_summary(history, scope, flow=None):
    """Prefer current complete dispatch gates; publications are a fallback."""
    current = (flow or {}).get("pending_gates")
    if current is not None:
        if not isinstance(current, list) or not all(isinstance(x, str) and x.strip() for x in current):
            raise ValueError("Current pending gates must be a complete list of descriptions")
        return list(dict.fromkeys(public_text(x) for x in current))
    latest = max((history, scope), key=lambda x: int(x["export_id"][1:]))
    remaining = scope.get("remaining_scope_names", [])
    gates = []
    for gate in latest.get("open_gates", []):
        if (remaining and all(name in gate for name in remaining)
                and re.search(r"head.*(?:scopes|范围)", gate, re.I)):
            gate = (f"{identifier(scope.get('candidate'))} 剩余 {scope['remaining_scopes']} 个诊断范围："
                    + "、".join(remaining))
        gates.append(public_text(gate))
    return list(dict.fromkeys(gates))


def task_clock(started_utc, observed_utc):
    """Describe elapsed time only for a recorded, timezone-aware start."""
    try:
        start = datetime.fromisoformat(started_utc)
        observed = datetime.fromisoformat(observed_utc)
        if start.utcoffset() is None or observed.utcoffset() is None:
            return {}
        seconds = (observed - start).total_seconds()
        if seconds < 0:
            return {}
    except (TypeError, ValueError):
        return {}
    return {"started_utc": start.isoformat(), "elapsed_seconds": seconds}


def dispatch_snapshot(state, observed_utc=None):
    """Current dispatch is authoritative; old reservation/capture fields are ignored."""
    observed_utc = observed_utc or datetime.now(timezone.utc).isoformat()
    flow = state.get("flow_current_dispatch") or {}
    workers = []
    for item in flow.get("running", []):
        if not isinstance(item, dict):
            continue
        assignment = identifier(item.get("assignment"))
        role = assignment.split("/")[0]
        workers.append({"agent": identifier(item.get("agent")), "assignment": assignment,
                        "expert": ROLE_NAMES.get(role, role), "work": public_text(item.get("work")) or "任务说明未声明",
                        **task_clock(item.get("started_utc"), observed_utc)})
    completed = [{"scope": public_text(x.get("scope")) or "已完成任务",
                  "result": public_text(x.get("result")) or "结果见运行记录",
                  "owner": identifier(x.get("owner") or "Root")}
                 for x in flow.get("completed", []) if isinstance(x, dict)]
    ready = [{"scope": public_text(x.get("scope") or x.get("task") or x.get("work")) or "待执行任务",
              "owner": identifier(x.get("owner") or "Root"),
              "reason": public_text(x.get("reason")) or "准备就绪；执行取决于当前调度和原始采集门槛。"}
             for x in flow.get("ready", []) if isinstance(x, dict)]
    waiting = [{"scope": public_text(x.get("scope") or x.get("task")) or "依赖等待",
                "reason": public_text(x.get("reason")) or "等待原因未声明",
                "owner": identifier(x.get("owner") or "Root")}
               for x in flow.get("waiting", []) if isinstance(x, dict)]
    job = flow.get("device_job")
    device = {"state": "idle", "label": "本轮无 GPU 任务（状态快照）", "scope": "", "measurement_id": "", "status": ""}
    if job:
        job = job if isinstance(job, dict) else {"status": "unconfirmed"}
        status = public_text(job.get("status")) or "unconfirmed"
        # A reservation, ready command or old completed job is not evidence of execution.
        if status in {"running", "executing", "active", "in_progress"}:
            kind, label = "running", "本轮 GPU 任务执行中（状态快照）"
        elif status in {"completed", "finished", "failed", "stopped", "cancelled", "terminal", "reviewed"}:
            kind, label = "idle", "GPU 当前登记任务已结束"
        elif status in {"ready", "prepared", "reserved", "waiting", "pending"}:
            kind, label = "waiting", "GPU 任务已准备，尚未执行"
        else:
            kind, label = "unknown", "GPU 已登记任务，执行状态未确认"
        device.update(state=kind, label=label, status=status,
                      scope=public_text(job.get("scope") or job.get("work") or job.get("kind")),
                      measurement_id=identifier(job.get("measurement_id")))
        if kind == "running":
            device.update(task_clock(job.get("started_utc"), observed_utc))
    gate = state.get("ncu_failure_gate_current") or {}
    ncu = {"active": gate.get("active") is True,
           "status": public_text(gate.get("status")),
           "measurement_id": identifier(gate.get("measurement_id") or gate.get("original_measurement_id")),
           "reason": public_text(gate.get("reason") or gate.get("trigger") or state.get("pause_reason"))}
    if ncu["active"]:
        ncu["reason"] = ncu["reason"] or "NCU 失败门槛仍生效，等待修复与独立检查后才能恢复后续采集"
        waiting.insert(0, {"scope": "NCU 采集暂停", "reason": ncu["reason"], "owner": "Root"})
    focus = flow.get("candidate_focus") or {}
    candidate_focus = {k: public_text(focus.get(k)) for k in ("title", "result", "next_decision")}
    return {"completed": completed, "workers": workers, "ready": ready, "waiting": waiting,
            "device": device, "ncu_gate": ncu, "candidate_focus": candidate_focus}


def collect_snapshot(run):
    run = Path(run).resolve()
    delivery = run.parents[2]
    state, state_pin = read_json(run / "state.json")
    scope_id = publication_id(state, "latest_swimlane_export")
    history_id = publication_id(state, "latest_public_export")
    pins = [{"label": "调度状态输入", **state_pin}]
    def verified_release(entry, export_id):
        expected = entry.get("export", {})
        release_path = Path(expected.get("path", run / "exports" / export_id / "export.json")).resolve()
        if release_path.parent != (run / "exports" / export_id).resolve():
            raise ValueError("Publication release is outside its reserved export directory")
        release, release_pin = read_json(release_path)
        if any(expected.get(k) != release_pin[k] for k in ("nbytes", "sha256")):
            raise ValueError("Publication release pin differs from current state")
        if release.get("status") != "published_verified" or release.get("export_id") != export_id:
            raise ValueError("Publication release is not verified for this ID")
        pins.append({"label": f"{export_id} 发布回执", **release_pin})
        return release

    release = verified_release(state["latest_swimlane_export"], scope_id)
    history_release = verified_release(state["latest_public_export"], history_id)

    def public_json(relative, label):
        relative = href(relative)
        path = (delivery / relative).resolve()
        allowed = (delivery / ".local/results").resolve()
        if not path.is_relative_to(allowed):
            raise ValueError("Publication points outside the local result archive")
        data, pin = read_json(path)
        pins.append({"label": label, "href": relative, **pin})
        return data

    scope = public_json(f"results/{scope_id}/scope-completion.json", "当前诊断范围完成回执")
    expected_scope = release.get("scope_completion", {})
    if any(expected_scope.get(k) != pins[-1][k] for k in ("nbytes", "sha256")):
        raise ValueError("Scope completion bytes differ from verified publication")
    history = public_json(f"results/{history_id}/summary.json", "普通性能历史")
    expected_history = history_release.get("public_summary", {})
    if any(expected_history.get(k) != pins[-1][k] for k in ("nbytes", "sha256")):
        raise ValueError("Ordinary history differs from its verified release")
    if scope.get("export_id") != scope_id or history.get("export_id") != history_id:
        raise ValueError("Publication ID does not match public content")
    # A scope keeps its historical history snapshot, independent of the latest ordinary release.
    historical = scope.get("cumulative_history", {})
    historical_id = historical.get("export_id")
    if not re.fullmatch(r"e\d{4,}", str(historical_id)):
        raise ValueError("Invalid historical scope history ID")
    public_json(f"results/{historical_id}/summary.json", "诊断发布时的普通历史快照")
    if any(historical.get(k) != pins[-1][k] for k in ("nbytes", "sha256")):
        raise ValueError("Historical scope history pin differs")
    diagnostics_path = href(f"results/{scope_id}/incremental-diagnostics.json")
    diagnostics = public_json(diagnostics_path, "最新诊断补充")
    diagnostic_pin = pins[-1]
    publication_path = run / "exports" / scope_id / "publication.json"
    publication, publication_pin = read_json(publication_path)
    if any(release.get("publication", {}).get(k) != publication_pin[k] for k in ("nbytes", "sha256")):
        raise ValueError("Diagnostic publication manifest pin differs")
    expected_diagnostics = [item["destination"] for item in publication.get("files", [])
                            if relocated_evidence_path(item["destination"]["path"], delivery).resolve()
                            == (delivery / diagnostics_path).resolve()]
    if len(expected_diagnostics) != 1 or any(expected_diagnostics[0].get(k) != diagnostic_pin[k] for k in ("nbytes", "sha256")):
        raise ValueError("Diagnostic data differs from its verified release")
    pins.append({"label": f"{scope_id} 复制校验清单", **publication_pin})
    # The current card is one tuple selected from verified ordinary records, never private allocations.
    qualified = [r for r in history.get("records", []) if r.get("kind") == "screening"
                 and r.get("measurement_valid") is True and r.get("numerical_status") == "passed"
                 and isinstance(r.get("iteration_index"), int) and r.get("trial_id")]
    if not qualified:
        raise ValueError("Published history has no qualified candidate screening records")
    latest = max(qualified, key=lambda r: (r["iteration_index"], r["attempt_index"]))
    current_iteration, current_attempt = latest["iteration_index"], latest["attempt_index"]
    published_candidate = f"{latest['trial_id']}/{latest['revision_id']}"
    published_records = [r for r in qualified if (r["iteration_index"], r["attempt_index"],
                         f"{r['trial_id']}/{r['revision_id']}") == (current_iteration, current_attempt, published_candidate)]
    current = history.get("latest_ordinary_candidate")
    p1_review = None
    if current:
        if (current.get("id"), current.get("iteration_index"), current.get("attempt_index")) != (published_candidate, current_iteration, current_attempt):
            raise ValueError("Current ordinary metadata differs from qualified records")
        declared_ids = set(current.get("measurement_ids", []))
        candidate_ids = {r["measurement_id"] for r in published_records}
        # A new core baseline has plot index 0 while the candidate keeps its
        # global iteration. Both sides still belong to the declared comparison.
        extra_ids = declared_ids - candidate_ids
        baseline_records = [r for r in history.get("records", [])
                            if r.get("measurement_id") in extra_ids
                            and r.get("variant") == "baseline" and r.get("iteration_index") == 0
                            and r.get("attempt_index") == current_attempt
                            and r.get("cohort") == current.get("cohort")
                            and r.get("kind") == "screening"
                            and r.get("measurement_valid") is True and r.get("numerical_status") == "passed"]
        if extra_ids and current.get("cohort") and all(
                r.get("cohort") == current["cohort"] for r in published_records):
            published_records += baseline_records
        if declared_ids != {r["measurement_id"] for r in published_records}:
            raise ValueError("Current ordinary measurement binding differs")
        for key in ("decision", "local_control"):
            expected = current[key]
            document = public_json(f"results/{history_id}/{expected['path']}", "当前普通结论" if key == "decision" else "当前性能独立审查")
            if any(expected.get(k) != pins[-1][k] for k in ("nbytes", "sha256")):
                raise ValueError("Current ordinary conclusion pin differs")
            if key == "decision":
                candidate = document.get("candidate")
                if candidate is None and all(k in document for k in (
                        "trial_id", "revision_id", "iteration_index", "attempt_index")):
                    candidate = document
                if candidate is None:
                    raise ValueError("Decision lacks an explicit candidate identity")
                if isinstance(candidate, str):
                    # Some decisions name the revision; the separately pinned
                    # allocation binds its iteration/attempt without changing it.
                    allocation_pin = current["allocation"]
                    allocation = public_json(f"results/{history_id}/{allocation_pin['path']}", "当前迭代关联")
                    if any(allocation_pin.get(k) != pins[-1][k] for k in ("nbytes", "sha256")):
                        raise ValueError("Current allocation differs from its verified publication")
                    if candidate != f"{allocation['trial_id']}/{allocation['revision_id']}":
                        raise ValueError("Decision revision differs from its allocation")
                    candidate = allocation
                if (f"{candidate['trial_id']}/{candidate['revision_id']}", candidate['iteration_index'], candidate['attempt_index']) != (published_candidate, current_iteration, current_attempt):
                    raise ValueError("Decision candidate differs from ordinary samples")
                if current["disposition"] == "rejected":
                    rejected = (document["status"].startswith("rejected_")
                                or (document["status"] == "rejected"
                                    and document.get("performance_accepted") is False))
                    if not rejected:
                        raise ValueError("Rejected card lacks the bound rejection")
                if current["disposition"] == "keep_provisional":
                    # Root dispositions use either field name for the same
                    # explicit decision; the candidate and document are pinned above.
                    keep_decision = document.get("disposition", document.get("decision"))
                    legacy_keep = (keep_decision in {"keep", "keep_provisional"}
                                   and document.get("status", "").startswith("keep_provisional"))
                    fixed_pair_keep = (
                        document.get("status") == "provisionally_retained_fixed_A49_falsifier_passed"
                        and document.get("matched_gain_claimed") is False
                        and document.get("full_microprofile_accepted") is False
                        and all(isinstance(document.get(key), dict) for key in ("NUM", "ordinary_O")))
                    conditions = document.get("conditions")
                    fixed_candidate_keep = (
                        document.get("schema") == "flagmega.Root-fixed-candidate-disposition/v1"
                        and document.get("status", "").startswith("keep_provisional_fixed_")
                        and document.get("measurement_valid") is True
                        and document.get("numerical_status") == "passed"
                        and document.get("retained_optimization_baseline") == published_candidate
                        and isinstance(conditions, dict) and bool(conditions)
                        and all(value is True for value in conditions.values())
                        and all(isinstance(document.get(key), dict) for key in ("NUM", "qualification")))
                    fixed_p1_keep = (
                        document.get("schema") == "flagmega.root-P1-disposition/v1"
                        and keep_decision == "keep_provisional"
                        and all(isinstance(document.get(key), dict) for key in ("NUM_qualification", "qualification"))
                        and set(document.get("scenarios", {})) == {"synthetic32", "chinese64"}
                        and all(v.get("strict_lower_mean") is True
                                and 0 < v["candidate_mean_ms"] < v["baseline_mean_ms"]
                                for v in document["scenarios"].values()))
                    if not (legacy_keep or fixed_pair_keep or fixed_candidate_keep or fixed_p1_keep) or document.get("performance_accepted") is not False:
                        raise ValueError("Provisional card lacks the bound provisional decision")
                decision_document = document
            else:
                if document.get("phase") == "P1":
                    if (document.get("passed") is not True or document.get("observations_qualified") is not True
                            or document.get("iteration_index") != current_iteration
                            or document.get("attempt_index") != current_attempt
                            or document.get("decision") != current["disposition"]
                            or document.get("scenarios") != decision_document.get("scenarios")):
                        raise ValueError("Full-forward P1 review differs from its bound decision")
                    by_id = {r["measurement_id"]: r for r in published_records}
                    expected_jobs = [f"{i:02d}-{by_id[mid]['variant']}-{by_id[mid]['scenario']}-p{by_id[mid]['process_index']}"
                                     for i, mid in enumerate(current["measurement_ids"], 1)]
                    if (document["jobs"] != expected_jobs
                            or document["independent_processes"] != len(published_records)
                            or document["total_samples"] != sum(r["sample_count"] for r in published_records)
                            or document["formal_rounds"] != sum(len(r["rounds"]) for r in published_records)):
                        raise ValueError("Full-forward P1 review does not cover the declared measurements")
                    p1_review = document
                    current = dict(current, local_change_percent=None)
                    continue
                metric_path = expected.get("metric_path", history_release.get(
                    "local_control_metric_path", ["summary", "combined128", "delta_percent"]))
                if not isinstance(metric_path, list) or not metric_path or not all(
                        isinstance(part, str) and part for part in metric_path):
                    raise ValueError("Local O metric requires a path in the bound review")
                value = document
                for part in metric_path:
                    if not isinstance(value, dict) or part not in value:
                        raise ValueError("Local O metric is absent from the bound review")
                    value = value[part]
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value != current["local_change_percent"]):
                    raise ValueError("Local O delta differs from the bound review")
    else:
        current = {"id": published_candidate,
                   "disposition": "rejected" if "reject" in json.dumps(history.get("latest_local_O_disposition", "")) else "unconfirmed",
                   "last_change": "已公布的固定候选；具体方案见对应历史记录",
                   "local_change_percent": None,
                   "reason": "普通样本和性能结论来自已验证历史；完整数值资格不等于最终性能接受"}
    rows = []
    for scenario, title in [("synthetic32", "Synthetic · 32 token"), ("chinese64", "中文 · 64 token")]:
        records = [r for r in published_records if r.get("iteration_index") == current_iteration
                   and r.get("scenario") == scenario and r.get("kind") == "screening" and r.get("measurement_valid") is True]
        if records:
            if p1_review is not None:
                stats = p1_review["scenarios"][scenario]
                samples = sum(r["sample_count"] for r in records)
                if samples != stats["samples_per_side"] or len(records) != len(stats["candidate_process_mean_ms"]):
                    raise ValueError("P1 scenario sample/process census differs from published records")
                mean = sum(r["mean_ms"] * r["sample_count"] for r in records) / samples
                if not math.isclose(mean, stats["candidate_mean_ms"], rel_tol=1e-12):
                    raise ValueError("P1 pooled mean differs from published process samples")
                rounds = {len(r["rounds"]) for r in records}
                if len(rounds) != 1:
                    raise ValueError("P1 processes have different formal-round counts")
                rows.append({"scenario": scenario, "label": title, "mean_ms": stats["candidate_mean_ms"],
                             "measurement_id": ", ".join(identifier(r["measurement_id"]) for r in records),
                             "processes": len(records), "rounds": rounds.pop(), "samples": samples,
                             "numerical_status": "passed"})
                continue
            record = records[-1]
            mean = record.get("mean_ms")
            if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean <= 0:
                raise ValueError("Invalid ordinary latency")
            rows.append({"scenario": scenario, "label": title, "mean_ms": mean,
                         "measurement_id": identifier(record.get("measurement_id")),
                         "processes": record.get("process_count", 0), "rounds": len(record.get("rounds", [])),
                         "samples": record.get("sample_count", 0), "numerical_status": public_text(record.get("numerical_status"))})
    if {r["scenario"] for r in rows} != {"synthetic32", "chinese64"}:
        raise ValueError("Current candidate requires both published scenarios")
    accepted = state.get("accepted_best")
    # An accepted candidate pointer alone is never a certificate for the full goal.
    goal_status = "active" if accepted is None else "acceptance_requires_review"
    generated_utc = datetime.now(timezone.utc).isoformat()
    dispatch = dispatch_snapshot(state, generated_utc)
    # Small immutable receipts substantiate recent progress without rescanning
    # model payloads, raw traces or unpublished large artifacts on each refresh.
    def progress_receipt(expected, label):
        path = Path(expected["path"]).resolve()
        if not path.is_relative_to(run) or path.stat().st_size > 1024 * 1024:
            raise ValueError("Dashboard progress evidence must be a small run receipt")
        document, actual = read_json(path)
        if any(expected.get(k) != actual[k] for k in ("nbytes", "sha256")):
            raise ValueError("Progress receipt differs from current dispatch")
        pins.append({"label": public_text(label), **actual})
        return document

    format_migration = (progress_receipt(state["figure_format_migration"], "SVG 格式修改记录")
                        if state.get("figure_format_migration") else {})

    def presentation_matches(expected, actual, target):
        fields = ("nbytes", "sha256")
        if all(expected.get(k) == actual[k] for k in fields):
            return True
        # Presentation revisions retain the original scientific review. Accept
        # only an explicitly pinned before/after mapping for this exact file.
        if (format_migration.get("figure_formats") != ["svg"]
                or target.suffix not in {".html", ".md"}):
            return False
        for change in format_migration.get("presentation_changes", []):
            before, after = change["before"], change["after"]
            if (Path(after["path"]).resolve() == target.resolve()
                    and all(before.get(k) == expected.get(k) for k in fields)
                    and all(after.get(k) == actual[k] for k in fields)):
                return True
        return False

    coverage_path = run / "expert-coverage.json"
    if coverage_path.is_file():
        coverage, coverage_pin = read_json(coverage_path)
        pins.append({"label": "四领域专家覆盖记录", **coverage_pin})
        domains = []
        for name, responsibility in DOMAIN_DUTIES.items():
            domain = coverage.get("domains", {}).get(name, {})
            review = domain.get("independent_review") or {}
            status = review.get("status", "unassessed")
            passed = status.startswith("passed")
            if passed:
                progress_receipt(review["handoff"], f"{name} 已有独立审查")
            disposition = domain.get("disposition", "unassessed")
            reused = disposition in {"unchanged", "unaffected_reused"}
            assignment = identifier(review.get("assignment"))
            domains.append({"domain": name, "responsibility": responsibility,
                            "review_assignment": assignment,
                            "workdir": f"agents/{assignment}" if assignment != "未声明" else "未分派",
                            "status": ("合约未变，复用已核验结果" if reused else "当前源码／合约审查通过")
                                      if passed else "审查待完成",
                            "active_assignments": [x["assignment"] for x in dispatch["workers"]
                                                   if x["assignment"].startswith(name.lower() + "_")]})
        dispatch["expert_coverage"] = {
            "candidate": identifier(f"{coverage.get('trial_id')}/{coverage.get('revision_id')}"),
            "domains": domains,
        }

    for item in (state.get("flow_current_dispatch") or {}).get("completed", []):
        if not isinstance(item, dict) or not item.get("evidence"):
            raise ValueError("Completed dashboard work requires an evidence receipt")
        for expected in item["evidence"]:
            progress_receipt(expected, item.get("scope") or "近期完成工作")
    focus = (state.get("flow_current_dispatch") or {}).get("candidate_focus")
    if focus:
        if not all(focus.get(k) for k in ("title", "result", "next_decision", "evidence")):
            raise ValueError("Current candidate focus requires its result, next check and evidence")
        for expected in focus["evidence"]:
            progress_receipt(expected, focus["title"])
    dispatch["timings"] = []
    seen_measurements = set()
    for item in (state.get("flow_current_dispatch") or {}).get("stage_timings", []):
        execution = progress_receipt(item["execution"], item["stage"])
        seconds = execution.get("elapsed_seconds")
        # Builds have terminal timing evidence but do not allocate measurements.
        measurement = (execution.get("measurement_id") or execution.get("execution_id")
                       or item.get("execution_id"))
        exit_code = execution.get("exit")
        if (not isinstance(exit_code, int) or isinstance(exit_code, bool)
                or not execution.get("ended_utc")
                or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Stage timing requires a terminal execution with measured elapsed time")
        execution_key = (measurement, execution.get("stage"))
        if not measurement or execution_key in seen_measurements:
            raise ValueError("Stage timing requires a unique execution ID")
        seen_measurements.add(execution_key)
        if dispatch["device"]["measurement_id"] == identifier(measurement):
            dispatch["device"].update(state="idle", label="GPU 当前登记任务已结束",
                                      status="completed" if exit_code == 0 else "failed")
            dispatch["device"].pop("elapsed_seconds", None)
            dispatch["device"].pop("started_utc", None)
        dispatch["timings"].append({"stage": public_text(item["stage"]), "seconds": seconds,
                                    "measurement_id": identifier(measurement),
                                    "exit_code": exit_code,
                                    "started_utc": public_text(execution.get("started_utc")),
                                    "ended_utc": public_text(execution["ended_utc"])})
        if item.get("breakdown"):
            binding = item["breakdown"]
            qualification = progress_receipt(binding["qualification"], "已核验的命令内耗时")
            scenario = qualification["scenario_receipts"][binding["scenario"]]
            if (qualification.get("passed") is not True
                    or scenario["execution"] != item["execution"]
                    or scenario["measurement_id"] != measurement
                    or scenario["execution_elapsed_seconds"] != seconds):
                raise ValueError("Stage breakdown must belong to this qualified execution")
            stages = scenario["existing_stage_timing_seconds"]
            if (not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) and v >= 0 for v in stages.values())
                    or sum(stages.values()) > seconds):
                raise ValueError("Recorded substeps must fit within command wall time")
            dispatch["timings"][-1]["breakdown"] = {
                "scenario": public_text(binding["scenario"]),
                "load_prepare_seconds": stages.get("load_prepare"),
                "full_NUM_seconds": stages.get("original_full_NUM"),
            }
    completed, declared = scope["completed_scopes"], scope["declared_scopes"]
    if not 0 <= completed <= declared or declared < 1 or scope["remaining_scopes"] != declared - completed:
        raise ValueError("Inconsistent diagnostic scope counts")
    operator_figure = "figures/operator-latency-candidate.svg"
    operator_description = f"{history_release.get('latest_operator_trial', published_candidate)} · 普通局部 O 对照，非 host ms/token"
    if state.get("local_operator_profile"):
        operator = progress_receipt(state["local_operator_profile"], "当前算子图安装资格")
        if (operator.get("status") != "qualified_operator_profile_installed"
                or not operator.get("independent_presentation_qualified")
                or not operator.get("installed_bytes_verified")):
            raise ValueError("Current operator figure lacks verified presentation and installation")
        operator_figure = href(operator["figure_href"])
        operator_description = public_text(operator["description"])
    links = [
        ("既有范围诊断图集", f"results/{scope_id}/portable/gallery.html", f"{scope_id} · {public_text(scope.get('scope'))} 的软件 / 硬件联合图"),
        ("普通性能曲线", "figures/performance-vs-iteration.svg", f"{history_id} · 历史普通样本；不含新诊断点"),
        ("最终性能对比", "figures/performance-comparison.svg", "最新原条件完整配对；两个场景和几何平均"),
        ("已验证算子耗时图", operator_figure, operator_description),
        ("layer3 诊断图集", "results/e0024/portable/gallery.html", "e0024 · 历史范围导出"),
        ("layer0 诊断图集", "results/e0023/index.html", "e0023 · 历史范围导出"),
        ("既有范围与复现说明", f"results/{scope_id}/readme.md", "诊断边界、失败保留与原始验收门槛"),
        ("完整优化历史", "optimization-trace.md", "性能改动、取舍、失败与后续诊断"),
        ("当前普通发布", f"results/{history_id}/readme.md", "当前候选、完整样本及局部对照结论"),
    ]
    for receipt in state.get("local_swimlane_galleries", []):
        gallery = progress_receipt(receipt, "SVG 泳道图库安装记录")
        if (gallery.get("status") != "qualified_svg_gallery_installed"
                or gallery.get("installed_bytes_verified") is not True
                or gallery.get("figure_formats") != ["svg"]):
            raise ValueError("SVG gallery lacks verified installation")
        relative = href(gallery["entry_href"])
        target = (delivery / relative).resolve()
        if not target.is_relative_to(delivery / ".local/results"):
            raise ValueError("SVG gallery is outside local results")
        raw = target.read_bytes()
        actual = {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if any(gallery["entry"].get(k) != actual[k] for k in actual):
            raise ValueError("SVG gallery differs from its installation record")
        links.insert(0, (public_text(gallery["title"]), relative, public_text(gallery["description"])))
    fullhead48 = state.get("current_gdn_fullhead48_diagnostic_export")
    if fullhead48:
        release = progress_receipt(fullhead48, "GDN fullhead48 图库安装凭据")
        if (release.get("status") != "published_verified"
                or release.get("all_installed_bytes_match_qualified_manifest") is not True):
            raise ValueError("GDN fullhead48 gallery is not a verified publication")
        for key, title, note in [
            ("public_stage_summary", "GDN 阶段与 overlap 分析", "逐 warp 软件等待与区间交集；包含记录器开销"),
            ("public_gallery", "GDN fullhead48 泳道图库", "两侧各 48 调用、全部物理 warp；3.008 µs 硬件活动采样"),
        ]:
            expected = release[key]
            target = relocated_evidence_path(expected["path"], delivery).resolve()
            if not target.is_relative_to(delivery / ".local/results") or target.stat().st_size > 1024 * 1024:
                raise ValueError("GDN fullhead48 link is outside the public results")
            raw = target.read_bytes()
            actual = {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            if not presentation_matches(expected, actual, target):
                raise ValueError("GDN fullhead48 link differs from its verified publication")
            relative = href(target.relative_to(delivery).as_posix())
            pins.append({"label": title, "href": relative, **actual})
            links.insert(0, (title, relative, note))
    for state_key, name, gallery_links in [
        ("current_common_q8_diagnostic_export", "Common Q8", [
            ("public_qualification", "Common Q8 图库独立核验", "完整数据映射、分页与离线复现；性能验收仍开放"),
            ("public_gallery", "t24 Common Q8 完整泳道图库", "父128 / 候112调用、59,904物理warp；3.008 µs GPU聚合硬件采样"),
        ]),
        ("current_output_norm_diagnostic_export", "Output NormStats", [
            ("public_qualification", "t26 泳道图独立核验", "完整数据映射、分页与离线复现"),
            ("public_stage_summary", "t26 阶段与 overlap 分析", "全部 warp 的观测软件区间与等待；包含插桩开销"),
            ("public_gallery", "t26 输出归一化完整泳道图", "两侧各64调用、49,152物理warp；3.008 µs GPU聚合硬件采样"),
        ]),
    ]:
        gallery_receipt = state.get(state_key)
        if not gallery_receipt:
            continue
        release = progress_receipt(gallery_receipt, f"{name} 图库安装凭据")
        if (release.get("status") != "published_verified"
                or release.get("all_installed_bytes_match_qualified_manifest") is not True):
            raise ValueError(f"{name} gallery is not a verified publication")
        for key, title, note in gallery_links:
            expected = release[key]
            target = relocated_evidence_path(expected["path"], delivery).resolve()
            if not target.is_relative_to(delivery / ".local/results") or target.stat().st_size > 1024 * 1024:
                raise ValueError(f"{name} link is outside the public results")
            raw = target.read_bytes()
            actual = {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            if not presentation_matches(expected, actual, target):
                raise ValueError(f"{name} link differs from its verified publication")
            relative = href(target.relative_to(delivery).as_posix())
            pins.append({"label": title, "href": relative, **actual})
            links.insert(0, (title, relative, note))
    current_gdn = state.get("current_gdn_diagnostic_export")
    if current_gdn:
        gdn_id = publication_id(state, "current_gdn_diagnostic_export")
        gdn_release = verified_release(current_gdn, gdn_id)
        gdn_scope = public_json(f"results/{gdn_id}/scope-completion.json", "当前 GDN 图库覆盖")
        if any(gdn_release.get("scope_completion", {}).get(k) != pins[-1][k]
               for k in ("nbytes", "sha256")):
            raise ValueError("GDN scope differs from its verified publication")
        coverage = gdn_scope.get("cohort_action_coverage", {})
        count = gdn_scope.get("actions_per_cohort_declared")
        if (gdn_scope.get("export_id") != gdn_id or not coverage
                or type(count) is not int or count <= 0
                or any(type(n) is not int or not 0 <= n <= count for n in coverage.values())):
            raise ValueError("Invalid GDN cohort/action coverage")
        index = delivery / ".local/results" / gdn_id / "index.html"
        raw = index.read_bytes()
        actual = {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if not presentation_matches(gdn_release.get("public_index", {}), actual, index):
            raise ValueError("GDN gallery differs from its verified publication")
        relative = href(index.relative_to(delivery).as_posix())
        pins.append({"label": f"{gdn_id} GDN 图库", "href": relative, **actual})
        note = "；".join(f"{identifier(name)} {n}/{count}" for name, n in coverage.items())
        links.insert(2 if fullhead48 else 0, ("既有 GDN 泳道图库" if fullhead48 else "当前 GDN 泳道图库", relative,
                         f"{gdn_id} · {note}；本包呈现覆盖，不表示完整候选验收"))
    for candidate, retained in sorted((state.get("retained_swimlane_exports") or {}).items()):
        retained_id = publication_id({"retained": retained}, "retained")
        if retained_id == scope_id:
            continue
        verified_release(retained, retained_id)
        expected = retained["public_index"]
        index = relocated_evidence_path(expected["path"], delivery).resolve()
        if not index.is_relative_to(delivery / ".local/results"):
            raise ValueError("Retained gallery points outside public delivery")
        raw = index.read_bytes()
        actual = {"nbytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if not presentation_matches(expected, actual, index):
            raise ValueError("Retained gallery differs from its verified publication")
        relative = href(index.relative_to(delivery).as_posix())
        pins.append({"label": f"{retained_id} 保留候选图库", "href": relative, **actual})
        links.append((f"保留候选泳道 · {identifier(candidate)}", relative,
                      f"{retained_id} · 已核验并发布的 WG / 硬件活动图；覆盖范围独立于当前候选"))
    matched = None
    final_gap = "尚未测得原条件下的最终配对差距"
    matched_release = state.get("latest_matched_comparison")
    if matched_release:
        if matched_release.get("status") != "published_verified":
            raise ValueError("Matched comparison has not been verified for publication")
        summary_path = href(matched_release["summary_href"])
        matched = public_json(summary_path, "完整配对比较")
        if any(matched_release["summary"].get(k) != pins[-1][k] for k in ("nbytes", "sha256")):
            raise ValueError("Matched comparison differs from its verified publication")
        if matched.get("schema") != "flagmega.public-matched-cohort-summary/v1":
            raise ValueError("Unsupported matched comparison schema")
        if not all(matched.get("qualified_gates", {}).values()) or not matched.get("qualified_gates"):
            raise ValueError("Matched comparison qualification is incomplete")
        differences = [f"{name} {v['candidate_latency_change_pct']:+.2f}%"
                       for name, v in matched["scenarios"].items()]
        final_gap = "完整配对延迟差：" + "；".join(differences)
        links.append(("完整配对比较", matched_release["readme_href"],
                      f"迭代 {matched['iteration_index']} · 两侧双场景各 2 进程 × 3 轮"))
    public_links = []
    for title, path, note in links:
        relative = href(path)
        if (delivery / relative).is_file():
            public_links.append({"title": title, "href": relative, "note": note})
    return {
        "schema": SCHEMA, "generated_utc": generated_utc,
        "state_updated_utc": public_text(state.get("updated_utc")), "evidence_updated_utc": public_text(scope.get("created_utc")),
        "run_id": identifier(run.name),
        "goal": {"status": goal_status, "objective": OBJECTIVE, "accepted_best": None if accepted is None else "已登记，完整验收需另核",
                 "final_gap": final_gap, "iteration": current_iteration, "attempt": current_attempt},
        "candidate": {"id": identifier(published_candidate), "disposition": current["disposition"],
                      "last_change": public_text(current["last_change"]),
                      "local_change_percent": current["local_change_percent"],
                      "reason": public_text(current["reason"])},
        "dispatch": dispatch,
        "performance": {"history_export": history_id, "unit": "host ms / decode token（含贪心采样与一次 CPU token 返回）",
                        "classification": "普通筛选数据；不是最终配对验收", "rows": rows,
                        "matched_acceptance_complete": False,
                        "matched_comparison": matched,
                        "latest_export_new_ordinary_samples": diagnostics.get("new_ordinary_samples", 0)},
        "diagnostics": {"export_id": scope_id, "candidate": identifier(scope.get("candidate")), "scope": public_text(scope.get("scope")),
                        "completed": completed, "declared": declared, "remaining": declared - completed,
                        "completed_names": [public_text(x) for x in scope.get("completed_scope_names", [])],
                        "new_attempts": scope.get("new_diagnostic_attempts", 0),
                        "new_iterations": scope.get("new_iterations", 0),
                        "measurements": [{"id": identifier(r.get("measurement_id")), "status": public_text(r.get("status")),
                                          "valid": r.get("diagnostic_valid") is True}
                                         for r in diagnostics.get("records", [])],
                        "retained_failures": [public_text(x) for x in scope.get("retained_failures", [])],
                        "limit": ((f"本表保留 {public_text(diagnostics['created_utc'])} 的审查快照；后续状态见上方工作进度。" if diagnostics.get("created_utc") else "")
                                  + "诊断只解释带插桩调用：存在记录器、等待和配置影响；不表示普通流水提速、精确引擎忙碌时间或逐 warp 硬件归因。")},
        "next_gates": pending_gate_summary(history, scope, state.get("flow_current_dispatch")),
        "links": public_links, "source_pins": pins,
    }


def validate_snapshot(s):
    if s.get("schema") != SCHEMA:
        raise ValueError("Unsupported dashboard snapshot schema")
    # Snapshot mode accepts only the compact public contract, never a run state.
    allowed = {"schema", "generated_utc", "state_updated_utc", "evidence_updated_utc", "run_id", "goal", "candidate", "dispatch", "performance", "diagnostics", "next_gates", "links", "source_pins"}
    if set(s) != allowed:
        raise ValueError("Snapshot fields differ from public contract")
    for item in s["links"] + s["source_pins"]:
        if "href" in item:
            href(item["href"])
    encoded = json.dumps(s, ensure_ascii=False, allow_nan=False)
    if re.search(r"\.local[/\\](?!(?:results|figures)[/\\])|/(?:home|mnt|tmp)/|\"(?:argv|environment|environ|password|secret|api_key)\"\s*:", encoded):
        raise ValueError("Private data is not allowed in dashboard snapshot")
    return s


def render_html(snapshot):
    s = validate_snapshot(snapshot)
    esc = lambda x: html.escape(str(x), quote=True)
    g, d, p, c = s["goal"], s["diagnostics"], s["performance"], s["candidate"]
    flow = s["dispatch"]
    paused = flow["ncu_gate"]["active"]
    badge = "NCU 采集暂停" if paused else "目标进行中 · 尚未完成" if g["status"] == "active" else "完整验收待核对"
    rejection = {"rejected": "已否决", "keep_provisional": "暂时保留"}.get(
        c["disposition"], "结论待确认")
    segments = ''.join(f'<span class="segment {"complete" if i < d["completed"] else ""}"></span>' for i in range(d["declared"]))
    def item(title, body, category=""):
        return f'<li><span class="tag">{esc(category)}</span><h3>{esc(title)}</h3><p>{esc(body)}</p></li>'
    def elapsed_label(task):
        seconds = task.get("elapsed_seconds")
        if seconds is None:
            return "开始时间未记录，耗时未知"
        return f"快照时已用 {seconds:.0f} 秒"
    recent = flow.get("completed", [])
    completed = ''.join(item(x["scope"], x["result"], f'近期完成 · {x["owner"]}') for x in recent[:4])
    older_completed = ''.join(item(x["scope"], x["result"], f'已完成 · {x["owner"]}') for x in recent[4:])
    older_completed += item(f"最近发布的性能改动：{c['id']}", f"{c['last_change']}。{c['reason']}。此值不是 host ms/token。", "性能优化")
    older_completed += item("数值验证与普通筛选已归档", f'迭代 {g["iteration"]} / 尝试 {g["attempt"]}。最近发布候选{rejection}，筛选数据保留；最终性能尚未接受。', "性能验证")
    older_completed += item(f'{d["completed"]} 个诊断范围已完成审查与发布', '、'.join(d["completed_names"]), "微观诊断")
    older_completed += item(f'{d["export_id"]} 新增 {d["new_attempts"]} 次诊断记录', f'新增普通样本 {p["latest_export_new_ordinary_samples"]}，新增迭代 {d["new_iterations"]}。原失败记录保留；修复与观察不构成提速证据。', "诊断 / 工具修复")
    completed_history = f'<details><summary>更多已完成工作与历史结果</summary><ul>{older_completed}</ul></details>'
    doing = ''.join(item(x["expert"], x["work"], f'{x["assignment"]} · {elapsed_label(x)}') for x in flow["workers"])
    if flow["device"]["state"] == "running":
        doing += item("Root · 设备执行", flow["device"]["scope"] or flow["device"]["measurement_id"], f'GPU 任务 · {elapsed_label(flow["device"])}')
    if not doing:
        doing = item("当前无已登记执行任务", "快照没有活动 agent 任务。已准备和等待中的任务列在“接下来做”。", "空闲")
    waits = ''.join(item(x["scope"], x["reason"], f'等待 · {x["owner"]}') for x in flow["waiting"])
    waiting_details = (f'<details><summary>等待依赖的工作（{len(flow["waiting"])} 项）</summary>'
                       f'<ul>{waits}</ul></details>') if waits else ''
    next_items = ''.join(item(x["scope"], x.get("reason", "准备就绪；执行取决于当前调度和原始采集门槛。"), f'下一项 · {x.get("owner", "Root")}') for x in flow["ready"])
    if not next_items:
        next_items = item("尚无已就绪任务", "待满足的依赖与负责人见下方。", "下一步")
    candidate_focus = flow.get("candidate_focus") or {}
    candidate_panel = (f'<section class="panel candidate-focus" aria-label="当前优化与下一次验证">'
                       f'<h2>{esc(candidate_focus["title"])}</h2><p>{esc(candidate_focus["result"])}</p>'
                       f'<p class="section-note"><strong>下一次验证：</strong>{esc(candidate_focus["next_decision"])}</p></section>') if candidate_focus.get("title") else ''
    gate_items = ''.join(item(f'未完成条件 {i + 1}', x, "原始目标") for i, x in enumerate(s["next_gates"]))
    gate_details = (f'<details><summary>全部待办与验收条件（{len(s["next_gates"])} 项）</summary>'
                    f'<ul>{gate_items}</ul></details>') if gate_items else ''
    active_work = '；'.join(x["work"] for x in flow["workers"])
    running_device = (flow["device"]["scope"] or flow["device"]["measurement_id"]) if flow["device"]["state"] == "running" else ""
    focus_text = (flow["ncu_gate"]["reason"] if paused else "") or '；'.join(
        x for x in (running_device, active_work) if x
    ) or "当前没有登记执行任务；已就绪的下一项和仍需等待的工作见下方看板。"
    timing_groups = {}
    for timing in flow.get("timings", []):
        group = timing_groups.setdefault(timing["stage"], {"seconds": 0, "measurements": [], "last_index": 0})
        group["seconds"] += timing["seconds"]
        label = timing["measurement_id"].split("-", 1)[0]
        if re.fullmatch(r"m\d+", label):
            group["last_index"] = max(group["last_index"], int(label[1:]))
        if timing.get("exit_code", 0) != 0:
            label += f'（失败，退出码 {timing["exit_code"]}）'
        group["measurements"].append(label)
    ordered_timings = sorted(timing_groups.items(), key=lambda x: x[1]["last_index"], reverse=True)
    def timing_row(stage, group):
        return f'<tr><th scope="row">{esc(stage)}</th><td>{group["seconds"]:.1f} 秒</td><td>{esc("、".join(group["measurements"]))}</td></tr>'
    timing_rows = ''.join(timing_row(stage, group) for stage, group in ordered_timings[:6])
    older_timing_rows = ''.join(timing_row(stage, group) for stage, group in ordered_timings[6:])
    timing_header = '<thead><tr><th>阶段</th><th>命令累计时间</th><th>来源</th></tr></thead>'
    timing_history = ('<details><summary>展开较早的命令耗时</summary><div class="overflow">'
                      f'<table>{timing_header}<tbody>{older_timing_rows}</tbody></table></div></details>') if older_timing_rows else ''
    timing_panel = ('<section class="panel" style="margin-top:16px"><h2>最近采集命令耗时</h2>'
                    '<p class="section-note">显示最近完成的六类命令，含失败尝试、模型加载与进程收尾。'
                    '尚未记录的排队、准备、审查和导出时间不在表内；各行不是内核耗时，也不是迭代总用时。</p>'
                    f'<div class="overflow"><table>{timing_header}<tbody>{timing_rows}</tbody></table></div>'
                    f'{timing_history}</section>') if timing_rows else ''
    breakdown_rows = []
    for timing in flow.get("timings", []):
        if not timing.get("breakdown"):
            continue
        b = timing["breakdown"]
        duration = lambda value: "未记录" if value is None else f"{value:.3f} 秒"
        breakdown_rows.append(
            f'<tr><th scope="row">{esc(b["scenario"])}</th>'
            f'<td>{duration(timing["seconds"])}</td>'
            f'<td>{duration(b.get("load_prepare_seconds"))}</td>'
            f'<td>{duration(b.get("full_NUM_seconds"))}</td>'
            f'<td>{esc(timing["measurement_id"].split("-", 1)[0])}</td></tr>')
    if breakdown_rows:
        timing_panel += ('<details class="panel"><summary>已记录的加载与数值验证耗时分项</summary>'
                         '<p class="section-note">下列分项来自同一条已核验命令，包含在上表命令时间内。'
                         '加载与准备是合计区间，现有记录无法进一步归因到文件读取、哈希或驱动；'
                         '其余启动、等待和收尾时间未在分项列中展开。</p>'
                         '<div class="overflow"><table><thead><tr><th>场景</th><th>整条命令</th>'
                         '<th>加载与准备</th><th>完整数值验证</th><th>来源</th></tr></thead>'
                         f'<tbody>{"".join(breakdown_rows)}</tbody></table></div></details>')
    rows = ''.join(f'<tr><th scope="row">{esc(x["label"])}</th><td class="number">{x["mean_ms"]:.4f}<small> ms</small></td><td>{esc(x["processes"])} 进程 × {esc(x["rounds"])} 轮<br><small>{esc(x["samples"])} 样本 · {esc(x["measurement_id"])}</small></td></tr>' for x in p["rows"])
    if not rows:
        rows = '<tr><td colspan="3">当前迭代暂无可公开的合格普通筛选样本</td></tr>'
    matched = p.get("matched_comparison")
    matched_panel = ""
    comparison_note = f"历史 llama 数据仅作背景，不能用于宣称当前胜出。{c['reason']}。"
    if matched:
        cohort = matched["cohort"]
        matched_rows = []
        for name, result in matched["scenarios"].items():
            values = result["backends"]
            candidate = next(v for k, v in values.items() if k != "llama.cpp")
            reference = values["llama.cpp"]
            matched_rows.append(
                f'<tr><th scope="row">{esc(name)}</th>'
                f'<td>{candidate["host_ms_per_token"]["mean"]:.4f}</td>'
                f'<td>{reference["host_ms_per_token"]["mean"]:.4f}</td>'
                f'<td>{result["candidate_minus_reference_ms"]:+.4f} ms'
                f' / {result["candidate_latency_change_pct"]:+.2f}%</td></tr>')
        comparison_note = "最新完整配对比较已独立核验；其结果与上方历史筛选数据分别保留。"
        verdict = "两场景平均延迟均更低；其他验收条件仍见待办" if matched["strict_two_scenario_win"] else "两个场景尚未均超过 llama.cpp，继续优化"
        matched_panel = (
            '<section class="panel" style="margin-top:16px"><h2>与 llama.cpp 还有多少差距</h2>'
            f'<p class="section-note">{esc(matched["trial_id"])} · 迭代 {matched["iteration_index"]}。'
            f'{cohort["independent_processes"]} 个独立进程、{cohort["measured_rounds"]} 轮、'
            f'{cohort["measured_samples"]} 个测量样本；排除 {cohort["excluded_warmup_rounds"]} 轮预热。'
            '单位为 host ms/token，含贪心采样与一次 CPU token 返回。正差值表示候选更慢。</p>'
            '<div class="overflow"><table><thead><tr><th>场景</th><th>FlagMega</th>'
            '<th>llama.cpp</th><th>延迟差</th></tr></thead>'
            f'<tbody>{"".join(matched_rows)}</tbody></table></div>'
            f'<p class="callout">{esc(verdict)}。完整进程与轮分布见下方比较数据链接。</p></section>')
    links = ''.join(f'<a class="evidence" href="{esc(x["href"])}"><strong>{esc(x["title"])} ↗</strong><span>{esc(x["note"])}</span></a>' for x in s["links"])
    agents = ''.join(f'<tr><td>{esc(x["expert"])}</td><td>{esc(x["assignment"])}<br><small>{esc(x["agent"])}</small></td><td>快照中执行</td></tr>' for x in flow["workers"])
    agents += '<tr><td>Root · 统筹 / 设备调度</td><td>Root</td><td>' + esc(flow["device"]["label"]) + '</td></tr>'
    expert_coverage = flow.get("expert_coverage")
    domain_panel = ""
    if expert_coverage:
        domain_rows = ''.join(
            f'<tr><th scope="row">{esc(x["domain"])}</th><td>{esc(x["responsibility"])}</td>'
            f'<td>{esc(x["status"])}<br><small>{esc(x["workdir"])}</small></td>'
            f'<td>{esc("、".join(x["active_assignments"]) or "当前无执行任务")}</td></tr>'
            for x in expert_coverage["domains"])
        domain_panel = ('<h2>Graph / Vector / Distribution / Kernel</h2>'
                        f'<p class="section-note">覆盖候选：{esc(expert_coverage["candidate"])}。'
                        '目录相对运行根目录；已有合约审查与当前执行分别显示，完整验收见待办。</p>'
                        '<table><thead><tr><th>领域</th><th>职责</th><th>已有审查与工作目录</th>'
                        f'<th>正在执行</th></tr></thead><tbody>{domain_rows}</tbody></table>')
    diag_rows = ''.join(f'<tr><td>{esc(x["id"])}</td><td>{"范围内通过" if x["valid"] else "待独立复核（发布快照）" if "pending" in x["status"] else "未取得有效诊断"}<br><small>{esc(x["status"])}</small></td></tr>' for x in d["measurements"])
    pin_rows = ''.join('<tr><td>' + (f'<a href="{esc(x["href"])}">{esc(x["label"])}</a>' if x.get("href") else esc(x["label"])) + f'</td><td>{x["nbytes"]}</td><td class="hash">{esc(x["sha256"])}</td></tr>' for x in s["source_pins"])
    embedded = json.dumps(s, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>FlagMega · 优化进度</title>
<style>
:root{{--ink:#172f40;--muted:#526675;--line:#dce4e7;--paper:#fff;--bg:#f3f6f7;--green:#11736a;--soft:#e9f4f1;--amber:#875116;--amberbg:#fff4df}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--bg);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif}}main{{max-width:1280px;margin:auto;padding:32px 28px 48px}}h1,h2,h3,p{{margin:0}}h1{{font-size:32px;letter-spacing:-.8px}}h2{{font-size:18px}}h3{{font-size:15px;line-height:1.5;margin:5px 0}}a{{color:#076b77;text-underline-offset:3px}}a:hover{{text-decoration:underline}}button,summary,a{{-webkit-tap-highlight-color:transparent}}a:focus-visible,button:focus-visible,summary:focus-visible{{outline:3px solid #167d9a;outline-offset:4px}}header{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:22px}}.eyebrow{{color:var(--green);font-size:12px;letter-spacing:1.3px;font-weight:750}}.subtitle{{color:var(--muted);margin-top:5px;max-width:780px}}.badge{{display:inline-block;background:var(--amberbg);color:var(--amber);border:1px solid #ead2a7;padding:5px 12px;border-radius:24px;font-size:13px;white-space:nowrap}}.meta{{color:var(--muted);font-size:12px;margin-top:9px}}.snapshot{{text-align:right;min-width:230px}}button{{font:inherit;font-size:12px;padding:5px 10px;border:1px solid #bdcdd3;border-radius:6px;background:white;color:var(--ink);cursor:pointer;margin-top:6px}}.panel{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px}}.stats{{display:grid;grid-template-columns:1.05fr 1.2fr 1fr 1.7fr;gap:12px;margin-bottom:14px}}.stat{{padding:17px 20px}}.label,.muted{{color:var(--muted)}}.label{{font-size:12px}}.value{{font-size:25px;font-weight:730;margin-top:3px;line-height:1.35}}.value small{{font-size:13px;font-weight:500;color:var(--muted)}}.stat p{{font-size:12px;color:var(--muted);margin-top:5px}}.scope-line{{display:flex;gap:4px;margin-top:12px}}.segment{{height:8px;flex:1;background:#e4e9ec;border-radius:2px}}.segment.complete{{background:var(--green)}}.focus{{border-left:4px solid var(--green);display:flex;gap:22px;align-items:flex-start;margin-bottom:18px;background:#eef6f5}}.focus.paused{{border-color:#a95324;background:var(--amberbg)}}.focus strong{{white-space:nowrap}}.focus p{{font-size:14px}}.board{{display:grid;grid-template-columns:1fr 1.05fr 1fr;gap:14px}}.column{{padding:20px 18px}}.column h2{{display:flex;align-items:center;gap:8px}}.dot{{height:9px;width:9px;border-radius:50%;background:#8b9da5}}.done .dot{{background:var(--green)}}.doing .dot{{background:#2c78a4}}.next .dot{{background:#ac7836}}ul{{list-style:none;padding:0;margin:15px 0 0}}li{{border-top:1px solid var(--line);padding-top:13px;margin-top:13px}}li:first-child{{margin-top:0}}li p{{color:var(--muted);font-size:13px}}.tag{{font-size:11px;letter-spacing:.2px;color:#49707f}}.lower{{display:grid;grid-template-columns:1.15fr 1fr;gap:14px;margin-top:16px}}.section-note{{font-size:12px;color:var(--muted);margin:5px 0 12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:11px 8px;vertical-align:top}}th{{font-weight:650}}td:first-child,th:first-child{{padding-left:0}}th:last-child,td:last-child{{padding-right:0}}thead{{color:var(--muted);font-size:12px}}.number{{font-variant-numeric:tabular-nums;font-size:23px;font-weight:700;white-space:nowrap}}small{{font-size:11px;color:var(--muted);font-weight:400}}.callout{{background:var(--amberbg);color:#754d20;border-radius:7px;padding:12px;margin-top:15px;font-size:13px}}.evidence-grid{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:14px}}.evidence{{padding:11px;border:1px solid var(--line);border-radius:7px;text-decoration:none;display:flex;flex-direction:column;gap:4px}}.evidence:hover{{background:#f4f9f9}}.evidence strong{{font-size:13px}}.evidence span{{font-size:11px;color:var(--muted)}}details{{margin-top:16px}}summary{{font-weight:650;cursor:pointer}}.details-body{{margin-top:12px;overflow-x:auto}}.hash{{font:11px/1.8 ui-monospace,monospace;overflow-wrap:anywhere;max-width:620px}}footer{{font-size:12px;color:var(--muted);margin-top:20px}}.overflow{{overflow-x:auto}}.no-win{{color:var(--amber)}}
@media(max-width:1000px){{.stats{{grid-template-columns:1fr 1fr}}.board{{grid-template-columns:1fr 1fr}}.next{{grid-column:1/-1}}.lower{{grid-template-columns:1fr}}}}
@media(max-width:650px){{main{{padding:20px 14px 32px}}header{{display:block}}h1{{font-size:27px}}.snapshot{{text-align:left;margin-top:14px;min-width:0}}.stats{{gap:8px}}.stat{{padding:13px}}.value{{font-size:22px}}.board{{grid-template-columns:1fr}}.next{{grid-column:auto}}.panel{{padding:16px}}.focus{{display:block}}.focus p{{margin-top:6px}}.evidence-grid{{grid-template-columns:1fr}}.number{{font-size:20px}}th,td{{padding:10px 5px}}.lower{{gap:12px}}}}
@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto}}}}
.candidate-focus{{margin-bottom:16px;border-left:4px solid #2c78a4}}.candidate-focus p{{margin-top:8px}}.candidate-focus .section-note{{margin-bottom:0}}.column,.stat{{min-width:0;overflow-wrap:anywhere}}
</style></head><body><main>
<header><div><div class="eyebrow">FLAGMEGA / RTX 5060 Ti / QWEN3.8 27B IQ4</div><h1>优化进度，一眼看清</h1><p class="subtitle">{esc(g["objective"])}</p></div><div class="snapshot"><span class="badge">{esc(badge)}</span><div class="meta">生成：<time id="generated-time" datetime="{esc(s['generated_utc'])}">{esc(s['generated_utc'])}</time><br><span id="age">状态快照，并非实时监控</span></div><button type="button" onclick="location.reload()">重新加载已生成的页面</button></div></header>
<section class="stats" aria-label="关键状态"><div class="panel stat"><div class="label">最终性能验收</div><div class="value no-win">尚未完成</div><p>accepted_best = {esc('null' if g['accepted_best'] is None else g['accepted_best'])}</p></div><div class="panel stat"><div class="label">最近已发布候选：{esc(c['id'])}</div><div class="value">{esc(rejection)}</div><p>迭代 {g['iteration']} · 尝试 {g['attempt']}</p></div><div class="panel stat"><div class="label">本轮 GPU 调度</div><div class="value">{esc({'idle':'空闲','running':'执行中','waiting':'等待执行','unknown':'待确认'}[flow['device']['state']])}</div><p>{esc('NCU 暂停门槛生效' if paused else f'当前 agent 任务：{len(flow["workers"])}')}</p></div><div class="panel stat"><div class="label">微观诊断范围 · {esc(d.get('candidate', '历史候选'))}</div><div class="value">{d['completed']} / {d['declared']} <small>剩余 {d['remaining']} 个</small></div><div class="scope-line" role="progressbar" aria-label="诊断范围已完成" aria-valuemin="0" aria-valuemax="{d['declared']}" aria-valuenow="{d['completed']}">{segments}</div></div></section>
<section class="panel focus {'paused' if paused else ''}" aria-label="当前执行与等待"><strong>{esc('NCU 采集暂停' if paused else flow['device']['label'])}</strong><p>{esc(focus_text)}{('<br><small>' + esc(flow['device']['measurement_id']) + '</small>') if flow['device']['measurement_id'] else ''}</p></section>
{candidate_panel}
<section class="board" aria-label="任务看板"><article class="panel column done"><h2><span class="dot"></span>已完成</h2><ul>{completed}</ul>{completed_history}</article><article class="panel column doing"><h2><span class="dot"></span>正在做</h2><ul>{doing}</ul></article><article class="panel column next"><h2><span class="dot"></span>接下来做</h2><ul>{next_items}</ul>{waiting_details}{gate_details}</article></section>
{matched_panel}
{timing_panel}
<div class="lower"><section class="panel"><h2>普通性能：真实筛选数据</h2><p class="section-note">{esc(p['unit'])}。{esc(p['classification'])}。</p><div class="overflow"><table><thead><tr><th>场景</th><th>平均延迟</th><th>测量规模 / 来源</th></tr></thead><tbody>{rows}</tbody></table></div><div class="callout"><strong>{esc(g['final_gap'])}</strong><br>{esc(comparison_note)}</div></section><section class="panel"><h2>已验证图表与证据</h2><p class="section-note">既有范围统计：{esc(d.get('candidate', '历史候选'))} / {esc(d['export_id'])} / {esc(d['scope'])}；普通性能历史：{esc(p['history_export'])}。</p><div class="evidence-grid">{links}</div></section></div>
<details class="panel"><summary>专家与 agent 分工</summary><div class="details-body">{domain_panel}<table><thead><tr><th>专家角色</th><th>当前 assignment / agent</th><th>状态</th></tr></thead><tbody>{agents}</tbody></table><p class="section-note">执行表只显示当前调度任务；历史审查与预留编号不代表仍在执行。</p></div></details>
<details class="panel"><summary>诊断与工具修复的边界 · {d['new_attempts']} 次新增记录 / {p['latest_export_new_ordinary_samples']} 个普通样本</summary><div class="details-body"><p class="section-note">{esc(d['limit'])}</p><p class="section-note">保留原失败：{esc('；'.join(d['retained_failures']))}。清理观察不证明因果修复或驱动 / context 已清空。</p><table><thead><tr><th>测量</th><th>诊断资格</th></tr></thead><tbody>{diag_rows}</tbody></table></div></details>
<details class="panel"><summary>快照时间与来源校验</summary><div class="details-body"><p class="section-note">状态字段更新时间：{esc(s['state_updated_utc'])}<br>范围证据时间：{esc(s['evidence_updated_utc'])}<br>页面生成时间：{esc(s['generated_utc'])}<br>状态字段时间与证据发布时间分别保留；此页不轮询进程，也不承诺实时状态。公开 JSON 内含全部重画数据。</p><table><thead><tr><th>来源</th><th>字节</th><th>SHA-256</th></tr></thead><tbody>{pin_rows}</tbody></table></div></details>
<footer>这是生成时刻的进度快照。重新加载只显示最近一次生成结果；运行状态变化后须由调度者更新快照。<br>所有图表均链接已有公开文件；进度条仅表示诊断范围完成数量。</footer>
</main><script id="dashboard-data" type="application/json">{embedded}</script><script>
const generated = Date.parse(document.getElementById('generated-time').dateTime);
const minutes = Math.floor((Date.now() - generated) / 60000);
if (Number.isFinite(minutes)) document.getElementById('age').textContent = minutes < 0 ? '生成时间晚于本机时钟；非实时监控' : '生成于 ' + minutes + ' 分钟前 · 非实时监控';
</script></body></html>'''


def atomic_write_text(path, content):
    """Keep a reload from observing a partially written status page or snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            os.fchmod(stream.fileno(), path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--run", type=Path, help="Read the private run and verified public exports")
    sources.add_argument("--snapshot", type=Path, help="Read only a compact public snapshot")
    parser.add_argument("--output-html", type=Path, default=Path(__file__).resolve().parent / "dashboard.html")
    parser.add_argument("--output-json", type=Path, help="Public JSON destination; --run only")
    args = parser.parse_args()
    if args.snapshot and args.output_json:
        parser.error("--output-json is only valid with --run; snapshot mode preserves its timestamp")
    snapshot = collect_snapshot(args.run) if args.run else read_json(args.snapshot)[0]
    rendered = render_html(snapshot)
    if args.run:
        json_path = args.output_json or Path(__file__).resolve().parent / ".local/results/dashboard-status.json"
        atomic_write_text(json_path, json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    atomic_write_text(args.output_html, rendered)
    print(f"Rendered {args.output_html.name}: {snapshot['goal']['status']}, {snapshot['dispatch']['device']['state']}, "
          f"{snapshot['diagnostics']['completed']}/{snapshot['diagnostics']['declared']} scopes")


if __name__ == "__main__":
    main()
