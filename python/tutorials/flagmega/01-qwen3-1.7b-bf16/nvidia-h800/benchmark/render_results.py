#!/usr/bin/env python3
"""CPU-only maintained-template live views, immutable sync and persistent watching."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

TEMPLATES = {"graph-trace": "graph", "operator-latency": "operators",
             "performance-vs-milestone": "milestones", "performance-vs-iteration": "history",
             "backend-comparison": "backends"}
_OWNED_CHILDREN = {}
EVIDENCE_ROOTS = (".local/baseline", ".local/milestones", "milestones")
ROOT_ARTIFACT_VIEW = ".local/views/root-artifacts.json"
REVISION_OPTIMIZATIONS = {
    'r002': 'GU 的 BF16 输入共享暂存',
    'r004': '移植输出连续的 split-K GU',
    'r005': '缩小输出分块至 BN128，增加 CTA 覆盖',
    'r006': '共享输入按 16 字节向量加载',
    'r007': '复用 async-wait 后的现有 CTA 屏障',
    'r008': '独立生产者流水加载 W TMA',
    'r009': '约束 warp-specialized 初始寄存器预算为 64',
    'r010': '复用 warp-specialized 分区的现有合流屏障',
    'r011': '生产者以 TMA 加载 X 输入',
    'r012': '采用 BN256 和 8 个消费 warp 的输出映射',
    'r013': 'W/X 共用完成屏障，消费者一次等待',
    'r014': '消费者 8 包展开改为两组 4 包',
    'r015': '合并连续 mbarrier 初始化的发布屏障',
    'r016': '消费者逐 lane release-arrival 替代消费后同步',
}
# Short descriptions belong to the measured source, not the latest revision.
# Keep the original immutable technical plan as each description's evidence.
HISTORY_OPTIMIZATIONS = {
    "fb6068cfb67603d976e522eca268b2d4009dcda3b921e6516f545c6f32996ad5": (
        "仅由规范写入 CTA 执行归一化统计，消除重复归约",
        ".local/agents/dispatch/consolidate_r000_canonical_norm_execution_plan/plan-body-v1.md",
        "1e16260d2ca000212a3ccc374c6b775e617be8162992dce2a66c98ca7cd8f013"),
    "f27dbf51551d84e0cbfee992623f056bdc3c87261e1cf9787ff8e112327aa63e": (
        "Down GEMV 的 BF16 输入异步暂存到共享内存",
        ".local/milestones/m000/revisions/r000/results/next-down-kernel-optimization/plan-v1.md",
        "91236e68484e9145ed798cc1f8236ef028491e506d529ad7789fa471cbe29ac2"),
    "516d2699c45169334bdb95da365468bd80b40fa07a36665ad7cf1076f81d95a6": (
        "提前吸收数值类型转换，保留原 BF16 舍入",
        ".local/milestones/m000/revisions/r003/results/g3-plan/plan-body-v1.md",
        "1888140f5a39cab243381b408fba64cc5c16654fff9c37403d079b743833cb28"),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def dump(path, value):
    atomic(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def find_skill():
    return next(p / "skills/flagmega-developer" for p in Path(__file__).resolve().parents
                if (p / "skills/flagmega-developer/SKILL.md").is_file())


class Inputs:
    def __init__(self, work):
        self.work, self.pins, self.errors, self.external = work, {}, [], {}
        self.captured = {}

    def raw(self, relative, *, capture=False):
        if str(relative) in self.captured:
            return self.captured[str(relative)]
        path = self.work / relative
        if not path.exists():
            return None
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("input changed while reading")
            self.pins[str(relative)] = {"path": str(relative), "sha256": hashlib.sha256(raw).hexdigest()}
            if capture:
                self.captured[str(relative)] = raw
            return raw
        except (ValueError, OSError) as error:
            self.errors.append({"path": str(relative), "error": str(error)})
            return None

    def read(self, relative):
        raw = self.raw(relative)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError as error:
            self.errors.append({"path": str(relative), "error": str(error)})
            return None


def read_dashboard(inputs, skill):
    raw = inputs.raw("dashboard.md", capture=True)
    if raw is None:
        return {"rows": [], "table": "", "columns": [], "schema": "flagmega.dashboard/v1"}
    parser = skill / "scripts/dashboard.py"
    spec = importlib.util.spec_from_file_location("live_dashboard_parser", parser)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs.external[parser.name] = {"path": str(parser), "sha256": sha(parser)}
    return module.parse(raw.decode())


def evidence_paths(row):
    return [match.split("#", 1)[0] for match in re.findall(r"\]\(([^)]+)\)", row.get("evidence", ""))
            if not match.startswith(("https:", "http:", "app:")) and ".." not in Path(match).parts]


def evidence_files(work, pattern):
    # workspace.py owns .local/milestones. Keep the former tracked root as
    # explicit legacy input, while excluding immutable render-input copies.
    return {path for root in EVIDENCE_ROOTS for path in (work / root).glob(pattern)
            if path.is_file() and path.resolve().is_relative_to(work.resolve()) and
            not {"input-snapshots", "template-sources"}.intersection(path.relative_to(work).parts)}


def live_measurements(work):
    return sorted(evidence_files(work, "**/measurement.json") |
                  set(work.glob(".local/cache/comparison/*baseline-round*.json")) |
                  set(work.glob(".local/cache/comparison/native-rounds/*.json")))


def pinned_workspace_object(inputs, pin):
    target = Path(pin['path'])
    target = (target if target.is_absolute() else inputs.work / target).resolve(strict=True)
    if not target.is_relative_to(inputs.work.resolve()) or not target.is_file():
        raise ValueError('selected evidence must be a regular workspace file')
    raw = inputs.raw(str(target.relative_to(inputs.work.resolve())))
    if raw is None or hashlib.sha256(raw).hexdigest() != pin['sha256'] or len(raw) != pin.get('bytes', pin.get('nbytes')):
        raise ValueError('selected evidence full pin changed')
    return json.loads(raw)


def current_model_artifacts(inputs, published):
    """A latest payload may include operator experiments beside the model."""
    selected = published.get('source_manifest')
    if not selected:
        return published
    path = Path(selected['path'])
    path = (path if path.is_absolute() else inputs.work/path).resolve()
    if not path.is_relative_to(inputs.work.resolve()):
        return published  # The existing runtime selector rejects foreign paths.
    stage = inputs.read(str(path.relative_to(inputs.work))) or {}
    if stage.get('schema') != 'flagmega.latest-artifacts/v1':
        return published
    stage = pinned_workspace_object(inputs, selected)
    model_pin = stage['whole_model_source_manifest']
    model = pinned_workspace_object(inputs, model_pin)
    if not isinstance(model.get('source_id'), str) or not model['source_id']:
        raise ValueError('latest artifacts need the actual whole-model source identity')
    return {**published, 'source_id': model['source_id'], 'source_manifest': model_pin,
            'published_latest_source_manifest': selected}


def selected_runtime(work, inputs, artifacts, source, program_hashes):
    selected = artifacts.get("source_manifest")
    expected = None
    if selected:
        manifest = work / selected.get("path", "")
        if not manifest.is_file() or not manifest.resolve().is_relative_to(work.resolve()):
            return None
        data = inputs.read(manifest.relative_to(work))
        if not data or inputs.pins[str(manifest.relative_to(work))]["sha256"] != selected.get("sha256"):
            inputs.errors.append({"path": str(manifest.relative_to(work)), "error": "selected source manifest changed"})
            return None
        if "codegen" in data:
            path, expected = manifest, selected["sha256"]
        elif data.get("runtime_manifest"):
            pin = data["runtime_manifest"]
            path, expected = work / pin["path"], pin["sha256"]
        elif data.get('schema') == 'flagmega.source-stage-manifest/v1' and data.get('runtime_artifact_manifest'):
            if data.get('source_id') != source:
                return None
            pin = data['runtime_artifact_manifest']
            pinned_workspace_object(inputs, pin)
            path, expected = work / pin['path'], pin['sha256']
        elif data.get("input_identities", {}).get("runtime_artifact"):
            path = work / data["input_identities"]["runtime_artifact"] / "artifact.json"
        elif data.get('schema') == 'flagmega.source-stage-adoption/v1' and data.get('original_runnable_binding'):
            if data.get('source_id') != source:
                return None
            runnable = pinned_workspace_object(inputs, data['original_runnable_binding'])
            if runnable['generated_source']['sha256'] not in program_hashes:
                return None
            pinned_workspace_object(inputs, runnable['runtime_artifact'])
            path, expected = work / runnable['runtime_artifact']['path'], runnable['runtime_artifact']['sha256']
        else:
            return None
    else:
        # Older fixed-baseline publications precede explicit source selection.
        path = work / ".local/baseline/build" / str(source) / "runtime-artifact/artifact.json"
    if not path.is_file() or not path.resolve().is_relative_to(work.resolve()):
        return None
    data = inputs.read(path.relative_to(work))
    if not data or (expected and inputs.pins[str(path.relative_to(work))]["sha256"] != expected):
        return None
    code = path.parent / data.get("codegen", {}).get("source", "")
    if code.is_file():
        if not code.resolve().is_relative_to(work.resolve()) or sha(code) not in program_hashes:
            return None
        inputs.pins[str(code.relative_to(work))] = {"path": str(code.relative_to(work)), "sha256": sha(code)}
    elif selected or not program_hashes:
        return None
    return path


def selected_num_receipt(inputs, evidence, manifest_hash, program_hashes):
    for _, path, receipt in reversed(evidence):
        if (receipt.get('agent_name') == 'reviewer' and receipt.get('NUM_qualified') and
                receipt.get('inputs', {}).get('runtime_manifest', {}).get('sha256') == manifest_hash):
            return receipt
        if receipt.get('schema') != 'flagmega.actual-NUM-CU-verdict/v1' or not receipt.get('NUM_qualified') or not receipt.get('CU_qualified'):
            continue
        if receipt.get('agent_id') != '/root/reviewer_sol' or receipt.get('signature', {}).get('agent_id') != receipt['agent_id']:
            continue
        body = {k: v for k, v in receipt.items() if k != 'signature'}
        if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != receipt['signature']['body_sha256']:
            raise ValueError('actual NUM/CU reviewer signature changed')
        report = pinned_workspace_object(inputs, receipt['actual_report'])
        terminal = pinned_workspace_object(inputs, receipt['actual_receipt'])
        bundle = pinned_workspace_object(inputs, receipt['frozen_native'])
        artifact_pin = bundle['artifact_identity']
        if (artifact_pin['sha256'] != manifest_hash or report['artifact']['manifest_sha256'] != manifest_hash or
                report['native_admission']['runtime']['artifact_source_sha256'] not in program_hashes):
            continue
        pinned_workspace_object(inputs, artifact_pin)
        entry = bundle['entries'][0]
        if (report['status'] != 'passed' or not report['NUM_qualified'] or
                sorted(s['prompt_length'] for s in report['scenarios']) != [32, 128, 250, 1024, 2048] or
                report['native_admission']['runtime']['num_layers'] != 28 or
                report['native_admission']['runtime']['vocab_size'] != 151936 or
                entry['source_identity']['sha256'] != report['native_admission']['runtime']['artifact_source_sha256'] or
                entry['native_key'] != report['native_admission']['kernels'][0]['metadata']['hash'] or
                terminal['status'] != 'completed' or terminal['exit_code'] != 0 or not terminal['terminal'] or
                not terminal['post_terminal_idle'] or not terminal['owned_session_cleanup']['terminal_qualified']):
            raise ValueError('actual NUM/CU source/native/full5/terminal qualification mismatch')
        for pin in entry['metadata_group'].values():
            target = Path(pin['path'])
            raw = target.read_bytes()
            if hashlib.sha256(raw).hexdigest() != pin['sha256'] or len(raw) != pin.get('bytes', pin.get('nbytes')):
                raise ValueError('actual NUM/CU frozen native member changed')
            inputs.raw(str(target.resolve().relative_to(inputs.work.resolve())))
        # Normalize field names from the signed actual schema, never its acceptance state.
        origin = inputs.pins[str(Path(path))]
        return {**receipt, 'inputs': {'runtime_manifest': artifact_pin},
                'selection': {'original_verdict': origin, 'actual_report': receipt['actual_report'],
                              'frozen_native': receipt['frozen_native'], 'source_native_full5_terminal_checked': True}}
    return None


def primary(report):
    return next((s for s in (report or {}).get("scenarios", []) if s["prompt_length"] == 128), None)


def device_identity(report):
    index = str((report or {}).get("visible_devices", ""))
    identity = {"index": index, "uuid": None}
    for line in str((report or {}).get("gpu_before", "")).splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) > 1 and cells[0] == index:
            identity["uuid"] = cells[1]
    return identity


def original_gpu_workload(report):
    workload = report.get("workload", {})
    return (workload.get("batch") == 1 and workload.get("layers") == 28 and
            workload.get("dtype") in ("bfloat16", "BF16") and workload.get("decode_tokens") == 64 and
            sorted(workload.get("prompt_lengths", [])) == [32, 128, 250, 1024, 2048] and
            report.get("cohort") == "event_instrumented")


def registered_native_backend_states(inputs):
    """Explicit current selection; receipts bind source, cohort, card and interval."""
    descriptor = inputs.read("benchmark/current-native-backend-input.json")
    if descriptor is None:
        return {}
    states = {}
    def pinned(pin):
        path = Path(pin["path"])
        path = path if path.is_absolute() else inputs.work / path
        if not path.resolve().is_relative_to(inputs.work.resolve()):
            raise ValueError("descriptor receipt is outside workspace")
        relative = str(path.relative_to(inputs.work))
        value = inputs.read(relative)
        raw = path.read_bytes()
        if (not isinstance(value, dict) or hashlib.sha256(raw).hexdigest() != pin["sha256"] or
                ("bytes" in pin and len(raw) != pin["bytes"])):
            raise ValueError("receipt/source hash or object type mismatch: " + relative)
        return value
    def source_pin(pin):
        path = Path(pin["path"])
        path = path if path.is_absolute() else inputs.work / path
        raw = path.read_bytes()
        if (hashlib.sha256(raw).hexdigest() != pin["sha256"] or
                ("bytes" in pin and len(raw) != pin["bytes"])):
            raise ValueError("selected source bytes changed: " + str(path))
        if path.resolve().is_relative_to(inputs.work.resolve()):
            inputs.raw(str(path.relative_to(inputs.work)))
        else:
            # External native sources are frozen with the chart just like the
            # shared renderer/parser, without absolute archive destinations.
            name = hashlib.sha256(str(path).encode()).hexdigest()[:16] + "-" + path.name
            inputs.external[name] = {"path": str(path), "sha256": pin["sha256"]}
    try:
        if descriptor["schema"] != "flagmega.current-native-backend-input/v1":
            raise ValueError("unsupported explicit selection schema")
        entries = descriptor["cohorts"]
        keys = [(e["backend"], e["cohort"]) for e in entries]
        duplicates = {key for key in keys if keys.count(key) > 1}
    except (KeyError, TypeError, ValueError) as error:
        inputs.errors.append({"path": "benchmark/current-native-backend-input.json", "error": "registered native cohort: " + str(error)})
        return {}
    for entry in entries:
        try:
            backend, cohort = entry["backend"], entry["cohort"]
            if (backend, cohort) in duplicates:
                raise ValueError("duplicate explicit backend/cohort selection")
            if cohort not in ("ordinary_no_events", "event_instrumented"):
                raise ValueError("unsupported timing cohort")
            report, receipt, config, manifest, verdict = [pinned(entry[key]) for key in
                ("report", "job_receipt", "control_config", "source_manifest", "source_verdict")]
            identity = entry["measurement_id"]
            if not (report.get("schema") == "flagmega.model-measurement/v1" and report.get("status") == "measured" and
                    report.get("GPU_initialized") and report.get("NUM_passed") and report.get("all64_exact") and
                    report.get("backend") == config.get("backend") == receipt.get("backend") == backend and
                    report.get("cohort") == cohort and report.get("measurement_id") == config.get("job_id") == receipt.get("job_id") == identity and
                    receipt.get("status") == "completed" and receipt.get("exit_code") == 0 and receipt.get("terminal") and
                    receipt.get("prelaunch_idle") and receipt.get("post_terminal_idle") and not receipt.get("error") and
                    receipt.get("config_sha256") == entry["control_config"]["sha256"]):
                raise ValueError("completed report/job/config/NUM identities differ")
            uuid = entry["physical_gpu_uuid"]
            if not (uuid == config.get("GPU_UUID") == receipt.get("GPU_UUID") ==
                    report.get("device_identity", {}).get("physical_gpu_uuid") ==
                    report.get("identity_at_completion", {}).get("cuda", {}).get("uuid")):
                raise ValueError("physical CUDA/NVML/report card identities differ")
            pins = config["source_pins"]
            for pin in pins:
                source_pin(pin)
            selected = entry["source_manifest"]
            if (config.get("source_manifest", {}).get("sha256") != selected["sha256"] or
                    not any(pin["sha256"] == selected["sha256"] and Path(pin["path"]).resolve() ==
                            (inputs.work / selected["path"]).resolve() for pin in pins)):
                raise ValueError("config does not bind the selected source manifest")
            script = report["script"]
            if not any(pin["sha256"] == script["sha256"] and Path(pin["path"]).resolve() ==
                       Path(script["path"]).resolve() for pin in manifest["files"]):
                raise ValueError("report adapter bytes differ from selected source")
            workload = report["workload"]
            if not (workload.get("batch") == 1 and workload.get("layers") == 28 and workload.get("vocab") == 151936 and
                    workload.get("dtype") == workload.get("kv_dtype") == "bfloat16" and workload.get("decode_tokens") == 64 and
                    sorted(workload.get("prompt_lengths", [])) == [32,128,250,1024,2048] and
                    report["controls"].get("warmups_per_scenario") == 2 and report["controls"].get("repeats") == 3):
                raise ValueError("original full-scope workload/control differs")
            if sorted(s["prompt_length"] for s in report["scenarios"]) != [32,128,250,1024,2048]:
                raise ValueError("full five scenarios absent")
            for scenario in report["scenarios"]:
                if len(scenario["runs"]) != 3:
                    raise ValueError("three own trajectories absent")
                for run in scenario["runs"]:
                    if not (len(run["token_ids"]) == 64 and len(run["decode_step_ms"]) == 63 and
                            run["NUM"]["first10_exact"] and run["NUM"]["all64_exact"]):
                        raise ValueError("own64/63 NUM trajectory incomplete")
                    if cohort == "event_instrumented" and len(run.get("device_decode_step_ms", [])) != 63:
                        raise ValueError("DEVICE event intervals incomplete")
                    if cohort == "ordinary_no_events" and "device_decode_step_ms" in run:
                        raise ValueError("ordinary HOST contains DEVICE event intervals")
            scenario = primary(report)
            values = {"HOST_TPS": scenario["host_tokens_per_second"]}
            if cohort == "event_instrumented":
                values["DEVICE_TPS"] = scenario["device_tokens_per_second"]
            if any(not isinstance(v, (int,float)) or not math.isfinite(v) or v <= 0 for v in values.values()):
                raise ValueError("positive finite recorded throughput required")
            boundary = report["boundary"]
            if not isinstance(boundary.get("host"), str) or not boundary["host"] or (cohort == "event_instrumented" and not boundary.get("device")):
                raise ValueError("explicit HOST/DEVICE interval boundary absent")
            if not isinstance(entry.get("review_qualification"), str) or not entry["review_qualification"] or not verdict.get("schema"):
                raise ValueError("source verdict and diagnostic qualification required")
            current = {**values, "report": entry["report"], "job_receipt": entry["job_receipt"],
                "control_config": entry["control_config"], "source_manifest": selected, "source_verdict": entry["source_verdict"],
                "measurement_id": identity, "cohort": cohort, "physical_gpu_uuid": uuid, "boundary": boundary,
                "review_qualification": entry["review_qualification"], "formal_comparison_qualified": False,
                "NUM": "all five ×3 own64 outputs exact; full KV/finite-row/independent timing review remains pending"}
            state = states.setdefault(backend, {"current_cohorts": {}, "phase": "native GPU NUM/timing diagnostics completed"})
            state["current_cohorts"][cohort] = current
        except (OSError, KeyError, TypeError, ValueError) as error:
            inputs.errors.append({"path": entry.get("report", {}).get("path", "benchmark/current-native-backend-input.json"),
                                  "error": "registered native cohort: " + str(error)})
    for state in states.values():
        ordinary = state["current_cohorts"].get("ordinary_no_events")
        event = state["current_cohorts"].get("event_instrumented")
        selected = ordinary or event
        state.update(measurement_id=selected["measurement_id"], physical_gpu_uuid=selected["physical_gpu_uuid"],
                     report=selected["report"], HOST_TPS=ordinary["HOST_TPS"] if ordinary else None,
                     HOST_boundary=ordinary["boundary"]["host"] if ordinary else "ordinary HOST pending",
                     DEVICE_TPS=event["DEVICE_TPS"] if event else None,
                     DEVICE_boundary=event["boundary"]["device"] if event else "DEVICE pending")
        clauses = ([f"ordinary HOST/no events {ordinary['HOST_TPS']:.6f} TPS"] if ordinary else ["ordinary HOST pending"])
        clauses += ([f"event DEVICE {event['DEVICE_TPS']:.6f} TPS; instrumented HOST {event['HOST_TPS']:.6f} TPS (separate boundary)"] if event else ["DEVICE pending"])
        state["reason"] = "Native GPU full5 own64/63 NUM completed; " + "; ".join(clauses) + "; unreviewed diagnostics; full KV/finite rows and three rotated matched rounds pending"
    return states


def native_backend_states(inputs, dashboard_evidence):
    """Project frozen runtime outcomes without admitting HOST values to DEVICE bars."""
    registered = registered_native_backend_states(inputs)
    states = {}
    phase = next((receipt for _, _, receipt in reversed(dashboard_evidence)
                  if receipt.get("schema") == "flagmega.agent-handoff/v1" and
                  receipt.get("assignment_id") == "reference_gpu_num_gpu3"), None)
    if not phase:
        return registered

    def pinned(pin, members=None, as_raw=False):
        relative = pin.get("path", "")
        path = inputs.work / relative
        if not relative or not path.resolve().is_relative_to(inputs.work.resolve()):
            inputs.errors.append({"path": relative, "error": "native outcome pin is missing or outside workspace"})
            return None
        receipt = inputs.raw(relative) if as_raw else inputs.read(relative)
        wrong_type = receipt is None if as_raw else not isinstance(receipt, dict)
        if (wrong_type or inputs.pins.get(relative, {}).get("sha256") != pin.get("sha256") or
                (members is not None and members.get(relative) != pin.get("sha256"))):
            inputs.errors.append({"path": relative, "error": "frozen native outcome provenance/type mismatch"})
            return None
        return receipt

    manifest = pinned(phase.get("raw_evidence_manifest", {}))
    summary = pinned(phase.get("results", {}))
    if (not manifest or not summary or manifest.get("schema") != "flagmega.immutable-evidence-manifest/v1" or
            summary.get("schema") != "flagmega.native-reference-runtime-phase/v1"):
        return registered
    members = {pin["path"]: pin["sha256"] for pin in manifest.get("files", [])}
    for outcome in summary.get("successful_backends", []):
        report = pinned(outcome["report"], members)
        if not report:
            continue
        scenario = primary(report)
        if (report.get("schema") != "flagmega.model-measurement/v1" or report.get("status") != "measured" or
                report.get("backend") != outcome["backend"] or not report.get("GPU_initialized") or not report.get("NUM_passed") or
                not report.get("all64_exact") or not outcome.get("all64_exact") or not scenario):
            inputs.errors.append({"path": outcome["report"]["path"], "error": "native NUM/HOST outcome fields incomplete"})
            continue
        reason = (f"Native CUDA completed; GPU{summary['physical_index']} NUM: {outcome['exact_output_count']} own outputs exact (producer); "
                  f"HOST {scenario['host_tokens_per_second']:.6f} TPS (unreviewed diagnostic, 63 synchronous intervals/request); "
                  f"KV: {outcome['KV_export']}; DEVICE/stream/graph-use and three rotated matched rounds pending")
        states[outcome["backend"]] = {"reason": reason, "report": outcome["report"],
            "measurement_id": report["measurement_id"], "phase": "NUM/HOST diagnostic completed",
            "source_identity": {"checkpoint": report.get("checkpoint"), "version": report.get("version"),
                                "source_pins": report.get("source_pins"), "native_build": report.get("native_build")},
            "physical_gpu_uuid": summary["GPU_UUID"], "cohort": report["cohort"],
            "workload": report["workload"], "native_execution_mode": report.get("execution_mode"),
            "HOST_boundary": outcome["HOST_boundary"], "HOST_TPS": scenario["host_tokens_per_second"], "DEVICE_measurement": outcome["DEVICE_measurement"],
            "formal_three_rotated_performance_comparison": outcome["formal_three_rotated_performance_comparison"]}
    for outcome in summary.get("failed_backends", []):
        relative = next((path for path in members if path.endswith(outcome["job"] + "/measurement.failed.json")), None)
        report = pinned({"path": relative or "", "sha256": members.get(relative)}, members)
        causal = pinned(outcome["causal_evidence"], members, as_raw=True)
        if (report and causal and report.get("schema") == "flagmega.model-measurement/v1" and
                report.get("status") == "failed" and report.get("backend") == outcome["backend"]):
            states[outcome["backend"]] = {"reason": "Failed before NUM: " + outcome["root_cause"] + "; native HOST/DEVICE and fair rounds pending",
                "phase": "native runtime failed before NUM", "report": {"path": relative, "sha256": members[relative]},
                "measurement_id": report["measurement_id"], "causal_evidence": outcome["causal_evidence"]}
    for backend, current in registered.items():
        current["historical_state"] = states.get(backend)
        historical = current["historical_state"]
        if (current.get("HOST_TPS") is None and historical and
                historical.get("phase") == "NUM/HOST diagnostic completed" and
                historical.get("cohort") == "ordinary_no_events" and
                isinstance(historical.get("HOST_TPS"), (int, float)) and
                math.isfinite(historical["HOST_TPS"]) and historical["HOST_TPS"] > 0):
            current["historical_ordinary_HOST"] = historical
            current["reason"] = current["reason"].replace("ordinary HOST pending",
                f"historical measured ordinary HOST {historical['HOST_TPS']:.6f} TPS; original qualification retained")
        states[backend] = current
    return states


def current_work_progress(inputs, dashboard_evidence, dashboard_rows=()):
    """Typed, hash-bound progress scopes; NUM/CPU facts create no timing points."""
    progress = {}
    def phase(assignment):
        return next((receipt for _, _, receipt in reversed(dashboard_evidence)
                     if receipt.get("schema") == "flagmega.agent-handoff/v1" and
                     receipt.get("assignment_id") == assignment), None)
    def pinned(pin):
        path = Path(pin["path"])
        if not path.is_absolute(): path = inputs.work / path
        if not path.resolve().is_relative_to(inputs.work.resolve()):
            parts = path.parts
            if ".local" not in parts: raise ValueError("progress pin outside workspace")
            path = inputs.work.joinpath(*parts[parts.index(".local"):])
        relative = str(path.relative_to(inputs.work))
        data = inputs.read(relative)
        if not isinstance(data, dict) or inputs.pins.get(relative, {}).get("sha256") != pin["sha256"]:
            raise ValueError("progress receipt type/hash mismatch: " + relative)
        return data
    for assignment, key in [("execute_hardware_trace_full_census_num_gpu3", "census"),
                            ("implement_native_reference_runtime_v3", "runtime_v3"),
                            ("implement_flashrt_dense_bf16_python_v1", "flashrt_python")]:
        receipt = phase(assignment)
        if not receipt: continue
        try:
            if key == "census":
                summary = pinned(receipt["summary"])
                rows = summary["rows"]
                if (summary.get("schema") != "flagmega.hardware-trace-NUM-summary/v1" or
                        sorted(r["prompt_length"] for r in rows) != [32,128,250,1024,2048] or
                        not summary["NUM_owner_validation_passed"] or
                        not all(r["output_count"] == 64 and r["decode_intervals"] == 63 and r["native_reference_exact"] and
                                r["finite_KV"] and r["every_valid_all28_KV_position_written"] and r["tails_zero"] for r in rows) or
                        summary["actual_profile_pointer"] != summary["preallocated_owner_pointer"] or
                        not summary["actual21_argument_owner_query_verified"]):
                    raise ValueError("census NUM/state/pointer scope incomplete")
                resources = summary["native"]["resources"]
                progress[key] = {"owner_NUM_passed": True, "independent_GPU_NUM_accepted": receipt["independent_GPU_NUM_accepted"],
                    "contexts": sorted(r["prompt_length"] for r in rows), "recording_enabled": summary["recording_enabled"],
                    "observed_timestamp_writes": summary["disabled_primary128_step0_audit"]["observed_timestamp_writes"],
                    "GPU_operator_timeline_accepted": summary["GPU_operator_timeline_accepted"], "resources": resources,
                    "native_cubin_sha256": summary["native"]["cubin_sha256"], "profile_pointer":summary["actual_profile_pointer"],
                    "summary_pin":receipt["summary"], "physical_uuid":summary["physical_uuid"],
                    "reason":f"Historical disabled-recording census NUM: full5 own64/63 + all28 KV/state passed (producer); regs{resources['registers_per_thread']}/warps{resources['num_warps']}/shared{resources['shared_memory_bytes']}/local{resources['local_memory_bytes']}/resident{resources['required_resident_blocks_per_sm']}; active21-slot profile pointer equals owner. This NUM cohort recorded0 timestamp writes; separate enabled capture status follows, no performance point."}
            elif key == "runtime_v3":
                manifest = pinned(receipt["source_manifest"])
                checks = pinned(receipt["selfcheck_receipt"])
                jobs = checks["selfchecks"]
                if (manifest.get("schema") != "flagmega.native-reference-runtime-source/v3" or len(manifest["files"]) != 6 or
                        checks["tests"]["exit_code"] != 0 or checks["tests"]["count"] != 22 or len(jobs) != 8 or
                        not all(j["terminal"] and j["exit_code"] == 0 and j["GPU_initialized"] is False for j in jobs)):
                    raise ValueError("runtime v3 CPU/source/CLI scope incomplete")
                progress[key] = {"CPU_tests":22,"CLI_preflights":8,"GPU_accepted":False,
                    "source_manifest":receipt["source_manifest"],"selfcheck_receipt":receipt["selfcheck_receipt"],
                    "reason":"Native runtime v3 CPU implemented: six benchmark files,22 tests +8 guarded CLI preflights passed; numeric CVD/UUID identity, actual SGLang graph_runner and native FULL INPUTtrue -> SELECTEDfalse replay observer repaired. Actual GPU identity/NUM/graph/stream/KV/overhead/fair rounds pending."}
            else:
                manifest = pinned(receipt["source_manifest"])
                headers = pinned(receipt["real_checkpoint_headers"])
                tests = pinned(receipt["CPU_regression_receipt"])
                if (manifest["production_source_count"] != 6 or tests["final"]["exit_code"] != 0 or
                        tests["final"]["tests"] != "56passed" or headers["actual_tensor_headers"] != 311 or
                        headers["all_header_dtypes"] != "BF16" or headers["complete_layers"] != 28 or
                        headers["parsed_config"]["vocab_size"] != 151936 or headers["GPU_access"] is not False):
                    raise ValueError("FlashRT-derived CPU/header/source scope incomplete")
                progress[key] = {"CPU_tests":56,"tensor_headers":311,"native_GPU_accepted":False,
                    "source_manifest":receipt["source_manifest"],"headers":receipt["real_checkpoint_headers"],
                    "reason":"FlashRT-derived six-Python BF16 port implemented:56 CPU tests +311 actual BF16 tensor headers passed (28 layers/full151936, meta-only). Native library/build/admission, own64 GPU NUM/reset/graph, classification and HOST/DEVICE timing pending."}
        except (OSError, ValueError, KeyError, TypeError) as error:
            inputs.errors.append({"path":assignment,"error":str(error)})
    # These scopes update working status without creating performance points.
    ring = phase("execute_hardware_trace_full_ring_num_gpu3_v2")
    if ring:
        try:
            summary = pinned(ring["summary"])
            pinned(ring["first_fault"])
            if summary.get("schema") != "flagmega.actual-native-resource-admission-failure/v1" or summary["fault_class"] != "RuntimeContractError":
                raise ValueError("ring failure schema/class mismatch")
            progress["ring_failure"] = {"spill_sum_bytes":summary["native_cached_resource_values_reported_by_guard"]["spill_sum_bytes"],
                "contexts_completed":summary["contexts_completed"], "summary_pin":ring["summary"],
                "reason":f"Full-ring NUM admission failed before model launch: RuntimeContractError, {summary['native_cached_resource_values_reported_by_guard']['spill_sum_bytes']} spill bytes against zero-spill contract; {summary['contexts_completed']} contexts completed. Producer/variant resource identity diagnosis T080; ordinary NUM unchanged, no Docker/SourceCounters cause claim."}
        except (OSError, ValueError, KeyError, TypeError) as error:
            inputs.errors.append({"path":"ring failure", "error":str(error)})
    approval = next((r for _, _, r in reversed(dashboard_evidence) if r.get("schema") == "flagmega.implemented-source-scoped-approval/v1" and
                    r.get("assignment_id") == "review_runtime_v3_actual_implementation_graph_contract"), None)
    if approval and progress.get("runtime_v3"):
        try:
            pinned(approval["subject"]["source_manifest"])
            pinned(approval["CPU_actual_after"])
            if (approval["decision"] != "approve_exact_implemented_source_in_own_graph_contract_scope" or
                    approval["subject"]["source_manifest"]["sha256"] != progress["runtime_v3"]["source_manifest"]["sha256"]):
                raise ValueError("runtime v3 approval source mismatch")
            progress["runtime_v3"].update(source_scoped_approved=True, GPU_accepted=False)
            progress["runtime_v3"]["reason"] += " Actual six-file source independently approved in graph-contract CPU scope; GPU execution/NUM/stream/fullKV and performance remain pending."
        except (OSError, ValueError, KeyError, TypeError) as error:
            inputs.errors.append({"path":"runtime v3 approval", "error":str(error)})
    native = phase("implement_flashrt_sm90_cmake_and_native_build_v1")
    if native:
        try:
            outputs = {Path(pin["path"]).name:pin for pin in native["outputs"]}
            pinned(outputs["configured-native-only-000.json"])
            pinned(outputs["configured-default-000.json"])
            configured = outputs["selfcheck-configured-pass-after-000.log"]
            relative = str(Path(configured["path"]).relative_to(inputs.work))
            raw = inputs.raw(relative)
            if inputs.pins[relative]["sha256"] != configured["sha256"] or b"Ran 6 tests" not in raw or not raw.rstrip().endswith(b"OK"):
                raise ValueError("FlashRT configured six-check proof mismatch")
            job = inputs.read(str(Path(native["live_job"]["path"]).relative_to(inputs.work)))
            progress["flashrt_native"] = {"configured_checks":6, "native_GPU_accepted":False,
                "job_status":job.get("status") if isinstance(job,dict) else None,
                "reason":"FlashRT-derived SM90 native-only/default configurations and six configured checks passed. Native build lifecycle per current dashboard: " + "; ".join(row["id"] + " " + row["status"] + " " + row["description"] for row in dashboard_rows if row["id"] in ("T074","T084")) + "; native admission, GPU NUM/graph/classification and timings unaccepted."}
            if progress.get("flashrt_python"):
                progress["flashrt_python"]["reason"] += " " + progress["flashrt_native"]["reason"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            inputs.errors.append({"path":"FlashRT native progress", "error":str(error)})
    census_root = ".local/agents/dispatch/prepare_remaining_hardware_trace_phase_bindings/full-census-recording"
    terminal = inputs.read(census_root + "/captures/full-census/census/terminal.json")
    if isinstance(terminal,dict) and terminal.get("mode") == "census":
        try:
            supervisor = inputs.read(census_root + "/runs/hardware-first-20261114T172911Z-census-gpu3/terminal.json")
            if not (terminal["passed"] and terminal["all5_own64_63"] and terminal["all5_variant_full28_state_evidence"] and
                    terminal["actual_profile_pointer_query_verified"] and supervisor["passed"] and supervisor["exit_code"] == 0 and supervisor["postidle"]):
                raise ValueError("enabled census terminal/control scope incomplete")
            progress["census_recording"] = {"actual_terminal_passed":True, "raw_bytes":terminal["raw_bytes"], "physical_uuid":terminal["physical_uuid"],
                "cubin_sha256":terminal["native"]["cubin_sha256"], "source_binding_sha256":terminal["source_binding_sha256"],
                "GPU_performance_accepted":False,
                "reason":f"Enabled census capture terminal0: full5 own64/63 + full28 KV/state/21-slot owner passed; {terminal['raw_bytes']} raw bytes, GPU lease released. Leader timestamps are diagnostic, full warp/busy/drop/clock calibration and perturbation unverified; no performance point."}
        except (OSError, ValueError, KeyError, TypeError) as error:
            inputs.errors.append({"path":"enabled census", "error":str(error)})
    return progress


def census_leader_capture(inputs, progress):
    """Consume decoder-produced timestamps; no guessed resolution or busy time."""
    primary_path = ".local/agents/dispatch/execute_hardware_trace_census_recording_gpu3/primary-128-step0-intervals.json"
    primary = inputs.read(primary_path)
    if not isinstance(primary,dict) or not progress.get("census_recording"): return None
    def pinned_bytes(pin):
        path = Path(pin["path"])
        if not path.is_absolute(): path = inputs.work / path
        if not path.resolve().is_relative_to(inputs.work.resolve()):
            raise ValueError("census pin outside workspace")
        relative = str(path.relative_to(inputs.work))
        raw = inputs.raw(relative)
        if raw is None or inputs.pins[relative]["sha256"] != pin["sha256"]:
            raise ValueError("census dependency hash mismatch: " + relative)
        return raw
    try:
        scope = progress["census_recording"]
        if (primary.get("schema") != "flagmega.census-primary-intervals/v1" or primary["prompt_length"] != 128 or primary["decode_index"] != 0 or
                primary["source_binding"]["sha256"] != scope["source_binding_sha256"] or primary["cubin_sha256"] != scope["cubin_sha256"] or
                primary["physical_uuid"] != scope["physical_uuid"] or not primary["nominal_nanosecond_unit_verified"] or
                primary["all_physical_warp_coverage"] or primary["observed_engine_busy"]):
            raise ValueError("census primary source/device/clock/scope mismatch")
        for key in ("source_binding","raw_frame","decoder","dictionary","clock_documentation"):
            pinned_bytes(primary[key])
        intervals = primary["CTA0_role_leader_intervals"]
        if len(intervals) != len(primary["aggregate_occurrence_envelopes"]) or not intervals or any(
                not 0 <= r["relative_start_ticks"] <= r["relative_end_ticks"] <= primary["observed_leader_window_delta"] or
                r["cta"] != 0 or r["role"] not in ("consumer","producer") for r in intervals):
            raise ValueError("census emitted interval coverage/order invalid")
        result = {"primary":primary,"primary_path":primary_path,"intervals":intervals}
        compact = inputs.read(".local/agents/dispatch/execute_hardware_trace_census_recording_gpu3/decoded-315-compact.json")
        if isinstance(compact,dict):
            if (compact.get("schema") != "flagmega.census315-compact-validation/v1" or compact["source_binding"] != primary["source_binding"] or
                    compact["decoder"] != primary["decoder"] or compact["dictionary"] != primary["dictionary"] or
                    compact["primary"]["sha256"] != inputs.pins[primary_path]["sha256"] or not compact["all315_checksums_count_sentinels_mapping_valid"] or
                    compact["frame_count"] != 315 or compact["timestamp_order_errors"] != 0 or
                    sorted(r["prompt_length"] for r in compact["contexts"]) != [32,128,250,1024,2048]):
                raise ValueError("census compact hash/coverage mismatch")
            result["compact"] = compact
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        inputs.errors.append({"path":primary_path,"error":str(error)})
        return None


def three_probe_diagnostic(inputs, dashboard_evidence, runtime_hash, program_hashes):
    """Read pinned diagnostic outputs; they never enter ordinary history/control rows."""
    descriptor = inputs.read("benchmark/current-diagnostic-input.json")
    if not descriptor or not any(receipt.get("assignment_id") == "execute_ncu_001_diagnostic_gpu3" and
                                 receipt.get("schema") == "flagmega.assignment-handoff/v1"
                                 for _, _, receipt in dashboard_evidence):
        return None

    def pinned(pin, as_raw=False):
        path = Path(pin["path"])
        if not path.is_absolute():
            path = inputs.work / path
        if not path.resolve().is_relative_to(inputs.work.resolve()):
            # Identical copied evidence in an explicit CPU fixture keeps its
            # original absolute origins; read only its contained hash-bound copy.
            parts = path.parts
            if ".local" not in parts:
                raise ValueError("diagnostic pin outside workspace")
            path = inputs.work.joinpath(*parts[parts.index(".local"):])
        relative = str(path.relative_to(inputs.work))
        value = inputs.raw(relative) if as_raw else inputs.read(relative)
        wrong_type = value is None if as_raw else not isinstance(value, dict)
        if wrong_type or inputs.pins.get(relative, {}).get("sha256") != pin["sha256"]:
            inputs.errors.append({"path": relative, "error": "diagnostic receipt provenance/type mismatch"})
            raise ValueError("diagnostic pin changed or missing: " + relative)
        return value

    try:
        if descriptor.get("schema") != "flagmega.current-diagnostic-input/v1":
            raise ValueError("unknown diagnostic input schema")
        handoff = pinned(descriptor["handoff"])
        summary = pinned(handoff["summary"])
        helper = pinned(handoff["P2_helper_inspection"])
        pinned(handoff["P2_report"], as_raw=True)
        error = pinned(handoff["actual_error"])
        units = pinned(descriptor["dimensional_proof"])
        p2 = next(row for row in summary["rows"] if row["probe_id"] == "P2-hardware-counter-control")
        process = pinned(p2["process_packet"])
        if (summary.get("schema") != "flagmega.actual-GPU-diagnostic-summary/v1" or
                helper.get("schema") != "flagmega.existing-report-CPU-inspection/v1" or
                units.get("schema") != "flagmega.diagnostic-report-values/v1" or
                not p2["passed_scope"] or sorted(p2["whole_contexts_completed"]) != [32, 128, 250, 1024, 2048] or
                helper["report"]["sha256"] != handoff["P2_report"]["sha256"] or
                units["report"]["sha256"] != helper["report"]["sha256"]):
            raise ValueError("diagnostic full scope/report identity mismatch")
        code_hash = process["native"]["runtime"]["artifact_source_sha256"]
        if code_hash != summary["source_id"].split(":")[-1]:
            raise ValueError("diagnostic runtime/generated source mismatch")
        binding = {"source": summary["source_id"], "gpu": summary["physical_index"], "uuid": summary["physical_uuid"],
                   "runtime_manifest_sha256": process["artifact"]["manifest_sha256"], "generated_source_sha256": code_hash,
                   "native_key": p2["native_key"], "cubin_sha256": p2["cubin_sha256"],
                   "matching_selected_runtime": process["artifact"]["manifest_sha256"] == runtime_hash and code_hash in program_hashes}
        dimension_rows = {row["prompt_length"]: row for row in units["rows"]}
        rows = []
        for action in helper["actions"]:
            m = action["required_metrics"]
            dimension = dimension_rows[action["prompt_length"]]
            rate = m["dram__bytes.sum.per_second"]["value"]
            pct = m["dram__throughput.avg.pct_of_peak_sustained_elapsed"]["value"]
            if (action["source_id"] != binding["source"] or action["physical_uuid"] != binding["uuid"] or
                    action["native_key"] != binding["native_key"] or action["cubin_sha256"] != binding["cubin_sha256"] or
                    dimension["dram_bytes_per_second"] != rate or dimension["dram_fixed_peak_pct"] != pct or
                    abs(dimension["duration_seconds_from_bytes_div_bandwidth"] - dimension["duration_seconds_from_raw_ns"]) > 1e-12):
                raise ValueError("diagnostic action identity/unit proof mismatch")
            rows.append({"prompt_length": action["prompt_length"], "action_index": action["action_index"],
                         "DRAM_bytes_per_second": rate, "reported_dram_pct_peak_sustained_elapsed": pct,
                         "duration_ms": dimension["duration_ms"], "raw_duration_value": dimension["raw_duration_value"],
                         "raw_duration_unit_label": dimension["raw_duration_unit_label"], "metric_count": len(action["metric_names"]),
                         "binding": binding, "scope": summary["scope"], "raw_metrics": m})
        if sorted(row["prompt_length"] for row in rows) != [32, 128, 250, 1024, 2048]:
            raise ValueError("diagnostic action context coverage mismatch")
        host_diagnostic, host_timeline = None, None
        if "P1_stages" in descriptor:
            p1 = next(row for row in summary["rows"] if row["probe_id"] == "P1-ordinary-control")
            p1_process = pinned(p1["process_packet"])
            stage_bytes = pinned(descriptor["P1_stages"], as_raw=True)
            observer_source = pinned(descriptor["P1_observations_source"], as_raw=True)
            if (not p1["passed_scope"] or p1["native_key"] != binding["native_key"] or
                    p1["cubin_sha256"] != binding["cubin_sha256"] or
                    p1_process["artifact"]["manifest_sha256"] != binding["runtime_manifest_sha256"] or
                    b"monotonic_ns=time.monotonic_ns()" not in observer_source):
                raise ValueError("P1 HOST/source/clock evidence mismatch")
            request = next(case for case in p1_process["scenarios"] if case["prompt_length"] == 128)["runs"][0]
            if not request["full64_native_reference_exact"] or len(request["decode_step_host_ns"]) != 63:
                raise ValueError("P1 HOST full request coverage incomplete")
            host_diagnostic = {"tokens_per_second": 63e9/request["whole_decode_host_ns"],
                "whole_decode_host_ns": request["whole_decode_host_ns"], "tokens":63,
                "measurement_id":"P1-ordinary-control", "report":p1["process_packet"], "binding":binding,
                "boundary":p1_process["boundary"]["host_interval"],
                "observer_overhead":"Stage JSON logging and guard observations retained; one diagnostic request, no matched rotated comparison."}
            stages = [json.loads(line) for line in stage_bytes.splitlines()]
            selected = [e for e in stages if e.get("prompt_length") == 128 and e.get("request_phase") == "measurement" and e.get("repeat_index") == 0]
            markers = {name:[e for e in selected if e["name"] == name] for name in ["marker_start_entry","marker_stop_entry"]}
            if any(len(events) != 1 for events in markers.values()):
                raise ValueError("P1 HOST whole decode markers incomplete")
            start, end = [markers[name][0]["monotonic_ns"] for name in markers]
            selected = [e for e in selected if start <= e["monotonic_ns"] <= end]
            if (any(e["local_pid"] != p1_process["pid"] or e.get("marked_ncu") for e in selected) or
                    [e["monotonic_ns"] for e in selected] != sorted(e["monotonic_ns"] for e in selected)):
                raise ValueError("P1 HOST common clock/process/cohort mismatch")
            lanes = []
            for prefix,label,kind in [("decode_append","HOST input staging","unknown"),
                                      ("sample","HOST sample() + delivery","unknown"),
                                      ("token_D2H","HOST D2H enqueue API","transfer"),
                                      ("stream_sync","HOST completion wait","sync")]:
                intervals = []
                for token in range(63):
                    pair = [[e for e in selected if e["name"] == prefix + suffix and e.get("decode_index") == token] for suffix in ["_entry","_return"]]
                    if any(len(events) != 1 for events in pair):
                        raise ValueError("P1 HOST paired stage coverage incomplete: " + prefix)
                    intervals.append({"id":f"P1-{prefix}-{token}","start_ns":pair[0][0]["monotonic_ns"],"end_ns":pair[1][0]["monotonic_ns"],
                                      "kind":kind,"evidence":"inflight_envelope","token_index":token,
                                      "scope":"HOST API envelope only; asynchronous GPU inner execution unknown"})
                lanes.append({"label":label,"status":"captured","intervals":intervals})
            lanes.append({"label":"GPU / operators / WG","status":"unavailable","intervals":[],
                          "reason":"GPU trace not captured; hardware-only acquisition preparation. SourceCounters not required."})
            host_timeline = {"title":"HOST API timeline — GPU/operator capture unavailable", "window_ns":[start,end],
                "timebase":{"source":"P1 single-process time.monotonic_ns() / CLOCK_MONOTONIC; pinned observer source", "resolution_ns":1,"alignment":"validated_common"},
                "invocation":"P1 prompt128 -> own64 / full63 synchronous decode intervals",
                "scope":"whole_model", "rows":lanes, "binding":binding, "measurement_channel":"HOST",
                "aggregation":"Recorded HOST call envelopes; nested sample/D2H/wait spans overlap. GPU busy time and operator latency unknown; logging overhead retained.",
                "host_recorded_duration_ns":request["whole_decode_host_ns"], "stage_marker_duration_ns":end-start,
                "source_stage_pin":descriptor["P1_stages"]}
        return {"rows": rows, "binding": binding, "scope": summary["scope"], "error": error,
                "probes": [{k: row[k] for k in ("probe_id", "exit_code", "passed_scope", "whole_contexts_completed",
                            "actual_original_function", "actual_launch", "native_parameter_count", "actual_max_poll_gap_ns")} for row in summary["rows"]],
                "report": helper["report"], "unit_proof": descriptor["dimensional_proof"],
                "actual_NCU_error": summary["actual_NCU_error"], "frozen_supervisor_error_flag": summary["frozen_supervisor_error_flag"],
                "missing": summary["missing"], "HOST_diagnostic":host_diagnostic, "HOST_timeline":host_timeline, "ordinary_goal_or_performance_acceptance": False}
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        inputs.errors.append({"path": "benchmark/current-diagnostic-input.json", "error": str(error)})
        return None


def current_hardware_diagnostics(inputs, dashboard_evidence, runtime_hash, program_hashes):
    """Read authentic full-decode hardware sidecars without relabeling old P2."""
    candidates = []
    seen = set()

    def pinned(item, raw=False):
        path = Path(item['path'])
        if not path.is_absolute():
            path = inputs.work / path
        if not path.resolve().is_relative_to(inputs.work.resolve()):
            # Immutable CPU fixtures retain original absolute .local origins.
            if '.local' not in path.parts:
                raise ValueError('hardware pin outside workspace')
            path = inputs.work.joinpath(*path.parts[path.parts.index('.local'):])
        if not path.resolve().is_relative_to(inputs.work.resolve()):
            raise ValueError('hardware pin escapes workspace')
        relative = str(path.relative_to(inputs.work))
        value = inputs.raw(relative)
        if (value is None or len(value) != item['bytes'] or
                hashlib.sha256(value).hexdigest() != item['sha256']):
            raise ValueError('hardware full pin changed: ' + relative)
        return value if raw else json.loads(value)

    for _, path, receipt in dashboard_evidence:
        if (path in seen or receipt.get('schema') not in ('flagmega.own-scoped-handoff/v1', 'flagmega.hardware-publication/v1') or
                not isinstance(receipt.get('metrics'), dict) or 'report' not in receipt):
            continue
        seen.add(path)
        try:
            raw = inputs.raw(path)
            handoff = json.loads(raw)
            body = dict(handoff)
            signature = body.pop('signature')
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
            canonical = handoff['schema'] == 'flagmega.hardware-publication/v1'
            signed = (handoff.get('agent_id') in ('/root/kernel_profiler', '/root/kernel_profiler_sol') and
                      signature.get('agent_id') == handoff['agent_id'] and signature.get('body_sha256') == digest)
            if not canonical:
                signed = (handoff.get('agent_id') == '/root/kernel_profiler' and
                          signature == {'agent_id': handoff['agent_id'], 'body_sha256': digest})
            if (not signed or
                    handoff.get('successful_GPU_collections') != 1 or
                    handoff.get('actual_NCU_error') is not False or
                    handoff.get('all_owned_tools_terminal') is not True or
                    handoff.get('formal_performance_gain_HBM60_or_goal_accepted') is not False):
                raise ValueError('hardware handoff signature/terminal/diagnostic scope mismatch')
            boundary = None
            if canonical:
                terminal = pinned(handoff['terminal'])
                if (terminal.get('exit_code') != 0 or terminal.get('actual_NCU_error') is not False or
                        terminal.get('passed') is not True or terminal.get('postidle') is not True or
                        terminal['cleanup'].get('cleanup_complete') is not True or
                        terminal['cleanup']['whole_owned_session'].get('terminal_qualified') is not True):
                    raise ValueError('hardware publication actual terminal/whole-session mismatch')
                if handoff.get('boundary_diagnostics'):
                    boundary = pinned(handoff['boundary_diagnostics'])
            data = pinned(handoff['metrics'])
            source = pinned(handoff['source_provenance'])
            if (data.get('schema') != 'flagmega.actual-hardware-only-full-decode-metrics/v1' or
                    data.get('action_count') != 5 or data.get('range_containers') != 1 or
                    data.get('ordinary_performance_gain_HBM60_or_goal_accepted') is not False or
                    data['controls'].get('SourceCounters_PC_SW_PM') is not False or
                    data['report'] != handoff['report']):
                raise ValueError('hardware sidecar scope/report mismatch')
            pinned(data['report'], raw=True)
            mapping = pinned(data['actual_live_library_mapping'])
            rows = data['rows']
            if (len(rows) != 5 or sorted(r['prompt_length'] for r in rows) != [32, 128, 250, 1024, 2048] or
                    sorted(r['action_index'] for r in rows) != list(range(5))):
                raise ValueError('hardware full five-context coverage mismatch')
            first = rows[0]
            identity_keys = ('source_id', 'runtime_artifact_manifest_sha256', 'native_key', 'cubin_sha256',
                             'actual_mapped_library_sha256', 'physical_index', 'physical_UUID', 'PCI_bus_id')
            if (any(any(r[key] != first[key] for key in identity_keys) for r in rows) or
                    source['source_id'] != first['source_id'] or
                    source['artifact_manifest_sha256'] != first['runtime_artifact_manifest_sha256'] or
                    source['native_key'] != first['native_key'] or source['cubin_sha256'] != first['cubin_sha256'] or
                    source['actual_mapped_library_sha256'] != first['actual_mapped_library_sha256'] or
                    mapping.get('native_mapping_verified') is not True or
                    mapping['mapped_native']['sha256'] != first['actual_mapped_library_sha256']):
                raise ValueError('hardware source/native/route/live-mapping mismatch')
            for row in rows:
                raw_metrics = row['required_raw_metrics']
                elapsed = row['gpu_elapsed_duration_ns'] / 1e9
                rate = row['actual_total_bytes_by_fixed_GPU_elapsed_Bps']
                peak = row['fixed_peak_Bps']
                if (row['own_outputs'] != 64 or row['feedback_intervals'] != 63 or
                        row.get('full64_native_reference_exact') is not True or
                        row['elapsed_unit_evidence']['actual_CSV_unit'] != 'ns' or elapsed <= 0 or rate <= 0 or peak <= 0 or
                        row['gpu_elapsed_duration_ns'] != raw_metrics['gpu__time_duration.sum']['value'] or
                        row['actual_dram_total_bytes'] != raw_metrics['dram__bytes_read.sum']['value'] + raw_metrics['dram__bytes_write.sum']['value'] or
                        abs(rate - raw_metrics['dram__bytes.sum.per_second']['value']) > max(
                            math.ulp(rate), math.ulp(raw_metrics['dram__bytes.sum.per_second']['value'])) or
                        row['NCU_reported_DRAM_percent'] != raw_metrics['dram__throughput.avg.pct_of_peak_sustained_elapsed']['value'] or
                        not math.isclose(row['actual_dram_total_bytes'] / rate, elapsed, rel_tol=1e-12) or
                        not math.isclose(row['fixed_GPU_elapsed_bytes_by_fixed_peak_percent'], 100 * rate / peak, rel_tol=1e-12)):
                    raise ValueError('hardware own64/63/raw units/bytes-time proof mismatch')
            code_hash = first['source_id'].split(':')[-1]
            binding = {'source': first['source_id'], 'gpu': first['physical_index'], 'uuid': first['physical_UUID'],
                       'PCI_bus_id': first['PCI_bus_id'], 'native_key': first['native_key'], 'cubin_sha256': first['cubin_sha256'],
                       'runtime_manifest_sha256': first['runtime_artifact_manifest_sha256'], 'generated_source_sha256': code_hash,
                       'matching_selected_runtime': first['runtime_artifact_manifest_sha256'] == runtime_hash and code_hash in program_hashes}
            if boundary and (sorted(r['prompt_length'] for r in boundary['rows']) != [32, 128, 250, 1024, 2048] or
                             any(r['reported_GPU_elapsed_ns'] != next(v['gpu_elapsed_duration_ns'] for v in rows if v['prompt_length'] == r['prompt_length']) for r in boundary['rows']) or
                             boundary['maximum_observed_poll_gap_ns'] != data['actual_monitor_limitations']['maximum_observed_poll_gap_ns'] or
                             handoff['maximum_observed_poll_gap_ns'] != boundary['maximum_observed_poll_gap_ns']):
                raise ValueError('hardware boundary diagnostics source/poll-gap mismatch')
            candidates.append({'capture_id': handoff['capture_id'] if canonical else handoff['assignment'], 'UTC': handoff['UTC'], 'binding': binding,
                               'rows': rows, 'scope': data['scope'], 'report': handoff['report'],
                               'metrics': handoff['metrics'], 'handoff_path': path,
                               'time_weighted': data['full_five_ranges_time_weighted'],
                               'monitor_limitations': data['actual_monitor_limitations'], 'six_dimensions': data['six_dimensions_availability'],
                               'boundary_diagnostics': boundary})
        except (ValueError, KeyError, TypeError, OSError) as error:
            inputs.errors.append({'path': path, 'error': 'current hardware sidecar: ' + str(error)})
    candidates.sort(key=lambda item: (item['UTC'], item['handoff_path']))
    current = next((item for item in reversed(candidates) if item['binding']['matching_selected_runtime']), None)
    return {'current': current, 'historical': [item for item in candidates if item is not current]}


def measured_optimization(inputs, source_sha256):
    description = HISTORY_OPTIMIZATIONS.get(source_sha256)
    if description is None:
        return {'main_improvement': '主要优化点待补充'}
    summary, relative, expected = description
    raw = inputs.raw(relative)
    if raw is None or hashlib.sha256(raw).hexdigest() != expected:
        return {'main_improvement': '主要优化点待补充',
                'optimization_reason': 'Matching immutable technical plan is unavailable or changed'}
    return {'main_improvement': summary,
            'optimization_source_sha256': source_sha256,
            'optimization_plan': {**inputs.pins[relative], 'bytes': len(raw)}}


def history_revisions(inputs, dashboard_rows):
    revisions = []
    for path in sorted((inputs.work/'.local/milestones').glob('m*/revisions/r*/revision.json')):
        relative = str(path.relative_to(inputs.work))
        revision = inputs.read(relative)
        if not revision or revision.get('schema') != 'flagmega.revision/v1':
            continue
        milestone, rid = revision['milestone_id'], revision['revision_id']
        if path.parent.name != rid or path.parents[2].name != milestone:
            raise ValueError('revision table metadata lineage mismatch')
        plan = str((path.parent/'plan.md').relative_to(inputs.work))
        raw = inputs.raw(plan)
        row = next((r for r in dashboard_rows if r['id'] == 'V'+rid[1:]), {})
        integrating = '实施中' in row.get('status', '') and '整模' in row.get('description', '')
        prefix = '整模集成中：' if integrating else '未集成内核实验：'
        summary = REVISION_OPTIMIZATIONS.get(rid)
        revisions.append({'milestone_id':milestone, 'revision_id':rid, 'unit_key':milestone+'/'+rid,
                          'point_id':rid, 'value':None,
                          'model_status':'整模未测（集成实施中）' if integrating else '整模未测 / 未集成',
                          'main_improvement':prefix+summary if summary and raw else '主要优化点待补充',
                          'revision_metadata':inputs.pins[relative],
                          'optimization_plan':inputs.pins.get(plan),
                          'actual_hypothesis':revision.get('hypothesis'),
                          'dashboard_status':row.get('status')})
    return revisions


def paired_performance_history(inputs, evidence):
    """Working measurements retain their own source/cohort, before final sealing.

    Discover only explicitly linked typed handoffs, not arbitrary result globs or
    the current artifact's state timestamp. Repeated refreshes reuse the same
    three-round points and their original measurement IDs.
    """
    def pinned(item):
        path = Path(item['path'])
        path = path if path.is_absolute() else inputs.work / path
        path = path.resolve()
        if not path.is_relative_to(inputs.work.resolve()):
            raise ValueError('paired performance input outside workspace')
        raw = inputs.raw(path.relative_to(inputs.work))
        if raw is None or hashlib.sha256(raw).hexdigest() != item['sha256'] or len(raw) != item['bytes']:
            raise ValueError('paired performance full pin changed')
        return json.loads(raw)
    def signature(receipt):
        body = {key: value for key, value in receipt.items() if key != 'signature'}
        declared = receipt.get('signature', {}).get('body_sha256', receipt.get('signature', {}).get('signed_body_sha256'))
        if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != declared:
            raise ValueError('paired performance signature changed')
    experiments, seen = [], set()
    for _, path, receipt in evidence:
        if receipt.get('schema') != 'flagmega.agent-handoff/v1' or not isinstance(receipt.get('actual_results'), dict):
            continue
        try:
            packet = pinned(receipt['actual_results'])
            down_lhs = packet.get('schema') == 'flagmega.down-lhs-actual-paired-results/v1'
            direct_revision = packet.get('schema') in ('flagmega.actual-paired-results/v1',
                                                       'flagmega.early-cast-actual-paired-results/v1')
            two_subjects = down_lhs or direct_revision
            if packet.get('schema') != 'flagmega.norm-owner-actual-paired-results/v1' and not two_subjects:
                continue
            if receipt['actual_results']['sha256'] in seen:
                continue
            signature(receipt)
            if packet['request'] != receipt['request'] or len(packet['jobs']) != (12 if two_subjects else 18):
                raise ValueError('paired original request/job scope mismatch')
            request = pinned(packet['request'])
            jobs = []
            for job in packet['jobs']:
                report, terminal, launch = pinned(job['report']), pinned(job['receipt']), pinned(job['launch'])
                workload = report.get('workload', {})
                full_workload = (workload.get('batch') == 1 and workload.get('layers') == 28 and
                    workload.get('dtype') in ('bfloat16', 'BF16') and workload.get('decode_tokens') == 64 and
                    sorted(workload.get('prompt_lengths', [])) == [32, 128, 250, 1024, 2048] and
                    report.get('cohort') in ('event_instrumented', 'ordinary_no_events'))
                if (report.get('schema') != 'flagmega.model-measurement/v1' or report['status'] != 'measured' or
                        not report['NUM_passed'] or not full_workload or
                        terminal['status'] != 'completed' or terminal['exit_code'] != 0 or not terminal['terminal'] or
                        not terminal['post_terminal_idle'] or not terminal['owned_session_cleanup']['terminal_qualified'] or
                        report['measurement_id'] != job['job_id'] or report['cohort'] != job['cohort']):
                    raise ValueError('paired actual measurement/terminal/boundary mismatch')
                jobs.append((job, report, terminal, launch))
            review = next((r for _, _, r in evidence if r.get('agent_id') in ('/root/reviewer', '/root/reviewer_sol') and
                           (r.get('producer_final_handoff', {}).get('sha256') == sha(inputs.work / path) or
                            r.get('producer_handoff', {}).get('sha256') == sha(inputs.work / path) or
                            (r.get('producer_signature_verified') is True and
                             r.get('producer_signed_handoff', {}).get('sha256') == sha(inputs.work / path)))), None)
            if review:
                signature(review)
                pinned(review['verdict'])
            points = []
            for summary in packet['summaries']:
                if summary['prompt_length'] != 128 or not (summary['boundary'] == 'DEVICE' or summary['cohort'] == 'ordinary_no_events'):
                    continue
                matched = [(j, r, t, l) for j, r, t, l in jobs if j['cohort'] == summary['cohort'] and j['subject'] == summary['subject']]
                if (len(matched) != 3 or sorted(j['round_index'] for j, _, _, _ in matched) != [0, 1, 2] or
                        summary['tokens'] != 567 or summary['independent_rounds'] != 3 or
                        not math.isclose(summary['tokens_per_second'], 1000 * summary['tokens'] / summary['total_ms'], rel_tol=1e-12)):
                    raise ValueError('paired total tokens/time or three rotated rounds mismatch')
                boundary = summary['boundary']
                points.append({'label': boundary + (' ordinary3' if boundary == 'HOST' else ' paired3'),
                    'point_id': summary['subject'] + ' ' + boundary + '×3 / ' + receipt['actual_results']['sha256'][:6],
                    'axis_label': summary['subject'] + ' ' + boundary,
                    'value': summary['tokens_per_second'], 'status': 'qualified' if review else 'measured',
                    'measurement_ids': [j['job_id'] for j, _, _, _ in matched], 'cohort': summary['cohort'],
                    'boundary': boundary, 'statistic': '567 feedback tokens / total measured seconds; three rotated rounds',
                    'first_measured_launch_utc': min(l['started_utc'] for _, _, _, l in matched),
                    'last_measured_terminal_utc': max(t['ended_utc'] for _, _, t, _ in matched),
                    'actual_sources': [j['native'] for j, _, _, _ in matched],
                    'review_status': 'independently audited measurement; original final gates pending' if review else 'producer measured; independent review/final gates pending',
                    'accepted_milestone': False})
            if len(points) != (4 if two_subjects else 6):
                raise ValueError('paired separate HOST/DEVICE/subject scope mismatch')
            # A unit result is the declared candidate's primary DEVICE result.
            # HOST/control subjects and trials remain evidence, not revisions.
            candidates = {c['candidate'] for c in packet['paired_contrasts']}
            if len(candidates) != 1:
                raise ValueError('one declared revision candidate required')
            candidate = next(iter(candidates))
            primary_point = next(p for p in points if p['boundary'] == 'DEVICE' and
                                 p['measurement_ids'][0] in [j['job_id'] for j, _, _, _ in jobs if j['subject'] == candidate])
            report = next(r for j, r, _, _ in jobs if j['subject'] == candidate)
            match = re.search(r'/milestones/(m[0-9]+)/revisions/(r[0-9]+)/', report['frozen_subject']['artifact']['path'])
            if not match:
                raise ValueError('candidate artifact lacks actual milestone/revision lineage')
            milestone_id, revision_id = match.groups()
            if down_lhs:
                adoption = pinned(receipt['revision_assignment'])
                signature(adoption)
                runnable = pinned(adoption['runnable_binding'])
                if (adoption.get('schema') != 'flagmega.root-revision-assignment/v1' or
                        adoption.get('agent_id') != '/root' or adoption['previous_execution_revision_id'] != revision_id or
                        runnable['runtime_artifact']['sha256'] != report['frozen_subject']['artifact']['sha256'] or
                        runnable['generated_source']['sha256'] != report['runtime']['artifact_source_sha256'] or
                        runnable['native_key'] != report['actual_native']['kernels'][0]['native_key'] or
                        runnable['native_cubin']['sha256'] != report['actual_native']['kernels'][0]['CUBIN']['sha256']):
                    raise ValueError('paired logical adoption must bind the actual frozen source/native')
                milestone_id, revision_id = adoption['milestone_id'], adoption['revision_id']
            elif direct_revision:
                signature(request)
                if (request.get('schema') != 'flagmega.request/v1' or
                        any((item.get('milestone_id'), item.get('revision_id')) != (milestone_id, revision_id)
                            for item in (request, receipt))):
                    raise ValueError('paired direct revision must match signed request and actual artifact lineage')
            relative = Path('.local/milestones') / milestone_id
            milestone = inputs.read(relative / 'milestone.json')
            revision = inputs.read(relative / 'revisions' / revision_id / 'revision.json')
            if (not milestone or not revision or milestone.get('schema') != 'flagmega.milestone/v1' or
                    revision.get('schema') != 'flagmega.revision/v1' or milestone['milestone_id'] != milestone_id or
                    revision['milestone_id'] != milestone_id or revision['revision_id'] != revision_id or
                    milestone['primary_metric'] != 'full_decode_tokens_per_second'):
                raise ValueError('declared unit/primary metric metadata mismatch')
            unit = {**primary_point, 'milestone_id': milestone_id, 'revision_id': revision_id,
                    **measured_optimization(inputs, report['runtime']['artifact_source_sha256']),
                    'unit_key': milestone_id + '/' + revision_id, 'point_id': revision_id,
                    'axis_label': revision_id, 'label': 'FlagMega primary DEVICE',
                    'runtime_manifest_sha256': report['frozen_subject']['artifact']['sha256'],
                    'historical_baseline': milestone['baseline'],
                    'final_milestone_completed': bool(review and review.get('winning_milestone_or_goal_accepted') is True
                                                    and milestone.get('release_receipt'))}
            seen.add(receipt['actual_results']['sha256'])
            experiments.append({'handoff': path, 'packet': receipt['actual_results'], 'points': points,
                'revision_primary': unit,
                'contrasts': packet['paired_contrasts'], 'review': review,
                'first_measured_launch_utc': min(p['first_measured_launch_utc'] for p in points),
                'qualifications': [packet['isolation_qualification'], packet['boundary_qualification']],
                'remaining_final_gates': packet['remaining_final_gates']})
        except (ValueError, KeyError, TypeError, OSError) as error:
            inputs.errors.append({'path': path, 'error': 'paired working performance: ' + str(error)})
    return sorted(experiments, key=lambda item: item['first_measured_launch_utc'])


def current_tir_views(inputs, artifacts):
    """Admit a complete diagnostic-source view without relabeling ordinary time."""
    selected = artifacts.get("source_manifest")
    if not selected:
        return None
    def pinned(item):
        path = Path(item["path"])
        path = path if path.is_absolute() else inputs.work / path
        target = path.resolve(strict=True)
        if not target.is_relative_to(inputs.work.resolve()):
            raise ValueError("TIR view input is outside workspace")
        value = inputs.read(str(target.relative_to(inputs.work.resolve())))
        if not isinstance(value, dict) or sha(target) != item["sha256"]:
            raise ValueError("TIR view input hash/object changed")
        return value
    try:
        source = pinned(selected)
        diagnostic = source.get("diagnostic_runtime_manifest")
        if diagnostic:
            pinned(diagnostic)
        for path in sorted(evidence_files(inputs.work, "**/profiles/graph/**/manifest.json"), key=lambda p:p.stat().st_mtime_ns, reverse=True):
            view = inputs.read(str(path.relative_to(inputs.work)))
            if not isinstance(view, dict) or view.get("schema") != "flagmega.observed-TIR-views/v1" or view.get("status") != "captured":
                continue
            if diagnostic and view.get("diagnostic_artifact", {}).get("sha256") != diagnostic["sha256"]:
                continue
            pinned(view["diagnostic_artifact"])
            dataset = pinned(view["dataset"])
            if (dataset.get("synthetic") is not False or dataset.get("full315_coverage") is not False or
                    dataset.get("scope") != {"prompt_length":128,"decode_index":0,"layers":28,"final315_coverage":False} or
                    len(dataset.get("calls", [])) != 623 or dataset.get("engine_busy") is not None):
                raise ValueError("TIR diagnostic scope or complete caller join missing")
            charts = {name:pinned(item) for name,item in view["figures"].items()}
            if set(charts) != {"graph","operators"} or any(c.get("shared_span_dataset") != view["dataset"] or c.get("synthetic") is not False for c in charts.values()):
                raise ValueError("TIR timeline/ranking must share the same genuine dataset")
            return {"manifest": {"path":str(path.relative_to(inputs.work)),"sha256":sha(path)}, "descriptor":view, "charts":charts,
                    "matching_selected_source": diagnostic is not None}
    except (ValueError, OSError, KeyError, TypeError) as error:
        inputs.errors.append({"path":"current TIR diagnostic views", "error":str(error)})
    return None


def collect(work, skill=None):
    inputs = Inputs(work)
    dashboard = read_dashboard(inputs, skill or find_skill())
    dashboard_rows = dashboard.get("rows", [])
    dashboard_evidence = []
    for row in dashboard_rows:
        for path in evidence_paths(row):
            if path.endswith(".json"):
                receipt = inputs.read(path)
                # Auxiliary evidence may be a JSON array. Its raw bytes remain
                # pinned, but only objects can supply receipt/device fields.
                if isinstance(receipt, dict) and receipt:
                    dashboard_evidence.append((row, path, receipt))
    paired_history = paired_performance_history(inputs, dashboard_evidence)
    backend_states = native_backend_states(inputs, dashboard_evidence)
    progress = current_work_progress(inputs, dashboard_evidence, dashboard_rows)
    if progress.get("runtime_v3"):
        for backend in ("vllm", "sglang"):
            state = backend_states.setdefault(backend, {"measurement_id":"runtime-v3-CPU-only"})
            if state.get("current_cohorts"):
                state["runtime_v3_CPU_history"] = progress["runtime_v3"]
                continue
            state.update(runtime_v3=progress["runtime_v3"], reason="Historical native attempt: " + state.get("reason", "unavailable") +
                         "; " + progress["runtime_v3"]["reason"], phase="runtime v3 CPU ready; actual GPU pending")
    flash_plan_only = any(receipt.get("assignment_id") == "approve_flashrt_v1_graph_profiler" and
                          receipt.get("status") == "scoped_FlashRT_v1_measurement_PLAN_approved_only"
                          for _, _, receipt in dashboard_evidence)
    flash_six_scopes = flash_plan_only and any(row["id"] == "T021" and "六" in row.get("description", "")
                                               for row in dashboard_rows)
    task = inputs.read(".local/task.yaml") or {}
    artifacts = current_model_artifacts(inputs, inputs.read(ROOT_ARTIFACT_VIEW) or {})
    selected_source = artifacts.get("source_id")
    program_hashes = set()
    for item in artifacts.get("files", []):
        path = work / item["path"]
        if path.name == "generated_kernels.py" and path.is_file() and sha(path) == item.get("sha256"):
            program_hashes.add(item["sha256"])
            inputs.pins[item["path"]] = {"path": item["path"], "sha256": item["sha256"]}
    current_runtime_hash = None
    runtime_path = selected_runtime(work, inputs, artifacts, selected_source, program_hashes)
    if runtime_path:
        current_runtime_hash = inputs.pins[str(runtime_path.relative_to(work))]["sha256"]
    selected_manifest = inputs.pins.get(str(runtime_path.relative_to(work))) if runtime_path else inputs.pins.get(ROOT_ARTIFACT_VIEW)
    latest_diagnostic = three_probe_diagnostic(inputs, dashboard_evidence, current_runtime_hash, program_hashes)
    hardware_diagnostics = current_hardware_diagnostics(inputs, dashboard_evidence, current_runtime_hash, program_hashes)
    device_index = "unselected"
    selected_uuid = None
    device_row_id = None
    for row, path, receipt in sorted(dashboard_evidence, key=lambda item: item[0].get("time_utc", "")):
        source = receipt.get("source_id", "")
        if (receipt.get("runtime_manifest_sha256") == current_runtime_hash and
                ":" in source and source.split(":")[-1] in program_hashes and
                source.split(":")[0] == selected_source and receipt.get("physical_gpu_uuid")):
            device_index, selected_uuid = str(receipt["physical_gpu_index"]), receipt["physical_gpu_uuid"]
            device_row_id = row["id"]
    trace = inputs.read(".local/agents/dispatch/build_trace_controls/handoff.json") or {}
    llama_build = inputs.read(".local/cache/comparison/llama-reference-build001.json")
    llama_driver = inputs.read(".local/cache/comparison/llama-driver-build001.json")
    tile = inputs.read(".local/cache/comparison/tilefoundry-classification-correction-000.json")
    native_paths = sorted(evidence_files(work, "**/native-diagnostic-data-*.json"), key=lambda p: (p.name, str(p)))
    packets = [inputs.read(path.relative_to(work)) for path in native_paths]
    packets = [p for p in packets if p]
    historical_mapping = inputs.read(".local/agents/dispatch/profile_idle_h800/historical-native-provenance-001.json")
    if historical_mapping and historical_mapping.get("data", {}).get("path") in inputs.pins and inputs.pins[historical_mapping["data"]["path"]]["sha256"] == historical_mapping["data"]["sha256"]:
        bindings = {r["capture_id"]: r for r in historical_mapping["mapping"]}
        for packet in packets:
            if packet.get("schema") == "flagmega.native-diagnostic-figure-data/v1" and "physical_gpu_index" not in packet:
                for row in packet.get("rows", []):
                    cid = "c0010" if "mainkernel" in row["scope"] else "c0009" if "fullprimary" in row["scope"] else "c0011"
                    if cid in bindings:
                        row.update({key: bindings[cid][key] for key in ("source_id", "runtime_manifest_sha256", "physical_gpu_index", "physical_gpu_uuid")})
                        row["capture_id"] = cid
    native = packets[-1] if packets else {}
    def native_binding(item, packet):
        source = item.get("source_id", packet.get("source_id", ""))
        code_hash = source.split(":")[-1] if ":" in source else item.get("actual_program_sha256")
        runtime = item.get("runtime_manifest_sha256", packet.get("runtime_manifest_sha256"))
        index = item.get("physical_gpu_index", packet.get("physical_gpu_index"))
        uuid = item.get("physical_gpu_uuid", packet.get("physical_gpu_uuid"))
        if index is None:
            for control in packet.get("controls", []):
                for key, text in control.items():
                    if key.startswith("before_device") and isinstance(text, str):
                        cells = [c.strip() for c in text.split(",")]
                        if len(cells) > 1:
                            index, uuid = cells[:2]
                            break
        matching = bool(current_runtime_hash and runtime == current_runtime_hash and code_hash in program_hashes and
                        source.split(":")[0] == selected_source and str(index) == device_index and
                        (not selected_uuid or uuid == selected_uuid))
        return {"matching_current": matching, "source": source or "unbound historical source",
                "gpu": index if index is not None else "unknown", "uuid": uuid or "unknown"}
    native_rows, native_controls, historical_native_controls, seen = [], [], [], set()
    for packet in packets:
        capture_time = ""
        process_pin = packet.get("event_process", {})
        if process_pin.get("path"):
            process_path = Path(process_pin["path"])
            if not process_path.is_absolute():
                process_path = work / process_path
            if not process_path.is_file() or not process_path.resolve().is_relative_to(work.resolve()):
                # Older packets record repo-relative paths, and immutable
                # fixtures can relocate the same hash-bound .local payload.
                parts = Path(process_pin["path"]).parts
                if ".local" in parts:
                    process_path = work.joinpath(*parts[parts.index(".local"):])
            if (process_path.is_file() and process_path.resolve().is_relative_to(work.resolve()) and
                    sha(process_path) == process_pin.get("sha256")):
                process = inputs.read(process_path.relative_to(work))
                if isinstance(process, dict):
                    capture_time = process.get("timestamp_utc", "")
        for row in packet.get("rows", []):
            key = json.dumps(row, sort_keys=True)
            if key not in seen:
                seen.add(key)
                native_rows.append({**row, "binding": native_binding(row, packet)})
        for control in packet.get("controls", []):
            binding = native_binding(control, packet)
            if binding["matching_current"]:
                native_controls.append({**control, "binding": binding})
            elif (control.get("cohort") == "event_instrumented" and
                  control.get("prompt_length") == 128 and control.get("tokens_per_second") is not None and
                  control.get("runtime_manifest_sha256") and ":" in control.get("source_id", "") and
                  control.get("physical_gpu_uuid")):
                # History belongs to the measured source/device, independently
                # of which artifact or NUM receipt is current now.
                historical_native_controls.append({**control, "binding": binding,
                                                   "capture_timestamp_utc": capture_time})
    native = {**native, "rows": native_rows}
    reports = []
    for path in live_measurements(work):
        report = inputs.read(path.relative_to(work))
        if report and report.get("schema") == "flagmega.model-measurement/v1" and report.get("status") == "measured":
            reports.append(report)
    reports.sort(key=lambda r: r.get("timestamp_utc", ""))
    def source_matches(report):
        artifact = report.get("artifact", {})
        return bool(current_runtime_hash and artifact.get("manifest_sha256") == current_runtime_hash and
                    report.get("runtime", {}).get("artifact_source_sha256") in program_hashes and selected_source and
                    (artifact.get("source_id") == selected_source or selected_source in Path(artifact.get("path", "")).parts))
    def current_matches(report):
        device = device_identity(report)
        return source_matches(report) and device["index"] == device_index and (not selected_uuid or device["uuid"] == selected_uuid)
    def latest(backend, cohort=None):
        return next((r for r in reversed(reports) if r.get("backend") == backend and
                     (cohort is None or r.get("cohort") == cohort) and
                     (backend != "flagmega" or current_matches(r))), None)
    events = latest("flagmega", "event_instrumented")
    if events and not original_gpu_workload(events):
        events = None
    source_events = next((r for r in reversed(reports) if r.get("backend") == "flagmega" and source_matches(r) and original_gpu_workload(r)), None)
    manifest_hash = current_runtime_hash if program_hashes else None
    num_receipt = selected_num_receipt(inputs, dashboard_evidence, manifest_hash, program_hashes)
    num_matches = bool(num_receipt and manifest_hash and num_receipt.get("inputs", {}).get("runtime_manifest", {}).get("sha256") == manifest_hash)
    numerical = "independently passed for selected source" if num_matches and num_receipt.get("NUM_qualified") else "pending for selected source"
    captures = {}
    for role in ("graph", "operators"):
        paths = evidence_files(work, "**/figure-" + role + ".json")
        for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True):
            manifest = inputs.read(path.parent.relative_to(work) / "manifest.json")
            if manifest and manifest.get("status") == "captured" and manifest.get("artifact_manifest_sha256") == manifest_hash:
                chart = inputs.read(path.relative_to(work))
                if chart:
                    for pin in chart.get("provenance", []):
                        original = (path.parent / pin["path"]).resolve()
                        if not original.is_relative_to(work) or sha(original) != pin["sha256"]:
                            raise ValueError("capture raw provenance does not match")
                        inputs.pins[str(original.relative_to(work))] = {"path": str(original.relative_to(work)), "sha256": pin["sha256"]}
                    captures[role] = chart
                    break
    tir_views = current_tir_views(inputs, artifacts)
    census_capture = census_leader_capture(inputs, progress)
    jobs = [{"id": row["id"], "source_id": row.get("description", ""), "status": row["status"],
             "flow_steps": row.get("flow_steps"), "owner": row.get("owner")} for row in dashboard_rows]
    dashboard_latest = max((r.get("time_utc", "") for r in dashboard_rows), default="unavailable")
    phase = "; ".join(r["id"] + ": " + r["status"] for r in dashboard_rows if r["status"] not in ("完成", "complete", "completed")) or "dashboard pending"
    acceptance = next((r["description"] for r in dashboard_rows if r["id"] == "G002"), "No accepted objective receipt in dashboard")
    resources = "; ".join(r["description"] for r in dashboard_rows if r["id"].startswith("R")) or "resource record pending"
    return {"dashboard": dashboard, "dashboard_latest_utc": dashboard_latest, "phase": phase,
            "acceptance": acceptance, "resources": resources, "artifacts": artifacts, "selected_source": selected_source,
            "source_binding": {"runtime_manifest_sha256": current_runtime_hash, "generated_source_sha256": sorted(program_hashes),
                               "source_manifest": selected_manifest},
            "selected_device": {"index": device_index, "uuid": selected_uuid, "dashboard_row_id": device_row_id}, "numerical": numerical,
            "num_receipt": num_receipt if num_matches else None, "captures": captures, "jobs": jobs,
            "trace": trace, "native": native or {}, "native_controls": native_controls,
            "historical_native_controls": historical_native_controls, "reports": reports,
            "native_backend_states": backend_states, "flash_plan_only": flash_plan_only, "flash_six_scopes": flash_six_scopes,
            "latest_diagnostic": latest_diagnostic, "current_hardware": hardware_diagnostics['current'],
            "hardware_history": hardware_diagnostics['historical'], "current_progress": progress, "census_leaders":census_capture, "tir_views":tir_views,
            "llama_ready": bool(llama_build and llama_driver and llama_build.get("status") == llama_driver.get("status") == "built_cpu_only"),
            "tile_correction": tile, "flashrt": task.get("comparators", {}).get("flashrt") or any("flashrt-project/FlashRT" in r["description"] for r in dashboard_rows),
            "historical_events": [r for r in reports if r.get("backend") == "flagmega" and original_gpu_workload(r) and not current_matches(r)],
            "events": events, "host": latest("flagmega", "ordinary_no_events"),
            "event_history": [r for r in reports if r.get("backend") == "flagmega" and current_matches(r) and original_gpu_workload(r)],
            "paired_performance": paired_history,
            "history_revisions": history_revisions(inputs, dashboard_rows),
            "vllm": latest("vllm"), "sglang": latest("sglang")}, inputs


def template_data(skill, inputs, directory, template, title):
    path = skill / "assets/workspace-figures" / (template + ".json")
    snapshot = directory / "template-sources/workspace-figures" / path.name
    atomic(snapshot, path.read_text())
    inputs.pins[str(snapshot.relative_to(inputs.work))] = {"path": str(snapshot.relative_to(inputs.work)), "sha256": sha(snapshot)}
    data = json.loads(path.read_text())
    data.update(synthetic=False, title=title, rows=[], provenance=[],
                cohort="Actual B1 BF16 original contexts; working view; acceptance remains pending",
                uncertainty_method="No accepted uncertainty estimate; measured baseline cohort only",
                template_binding={"path": str(snapshot), "original_path": str(path), "template_id": template, "sha256": sha(snapshot),
                                  "renderer_sha256": sha(skill / "scripts/render_workspace_figures.py")})
    for key in ("timebase", "window_ns", "milestones"):
        data.pop(key, None)
    return data


def fixed_peak_ratio(row):
    return row.get("full_range_or_kernel_bytes_time_ratio_to_fixed_peak_pct",
                   row.get("full_range_or_kernel_bytes_time_ratio_to_provisional_fixed_peak_pct"))


def build_charts(data, inputs, skill, directory):
    diagnostic = data.get("latest_diagnostic")
    hardware = data.get('current_hardware')
    progress = data.get("current_progress", {})
    charts = {name: template_data(skill, inputs, directory, source, title) for name, source, title in [
        ("graph-trace", "graph", "Full model execution timeline — actual capture pending"),
        ("operator-latency", "operators", "Graph operator time — attribution pending"),
        ("performance-vs-milestone", "milestones", "Performance versus milestone — no achieved milestone"),
        ("performance-vs-iteration", "history", "Performance versus iteration — numerical baseline only"),
        ("backend-comparison", "backends", "Backend comparison — matched GPU boundaries pending") ]}
    reason = "GPU/operator/WG common-clock capture not acquired; hardware-only capture preparation. SourceCounters excluded by current user scope."
    if diagnostic:
        reason += "; historical P3 SourceCounters failed; no actual GPU/operator/WG timeline yet"
    if progress.get("census"): reason += "; " + progress["census"]["reason"]
    for name in ("graph-trace", "operator-latency"):
        charts[name].update(pending_only=True, pending_reason="GPU/operator capture not acquired. Hardware-only capture preparation; SourceCounters not required. Await new root capture request and device lease.",
                            rows=[{"label": label, "value": None, "reason": reason, "intervals": []}
                                  for label in ("all28 model", "LM head + greedy", "state / transfers", "WG / hardware activity")])
        role = "graph" if name == "graph-trace" else "operators"
        if role in data["captures"]:
            capture = data["captures"][role]
            charts[name].update({key: value for key, value in capture.items() if key not in ("provenance", "template_binding", "working_status")})
            charts[name].pop("pending_only", None)
            charts[name].pop("pending_reason", None)
    if diagnostic and diagnostic.get("HOST_timeline") and "graph" not in data["captures"]:
        charts["graph-trace"].update(diagnostic["HOST_timeline"])
        if not diagnostic["binding"]["matching_selected_runtime"]:
            charts["graph-trace"]["title"] = "Historical P1 " + charts["graph-trace"]["title"]
        charts["graph-trace"].pop("pending_only", None)
        charts["graph-trace"].pop("pending_reason", None)
    leaders = data.get("census_leaders")
    if leaders:
        primary_capture = leaders["primary"]
        grouped = {}
        for interval in leaders["intervals"]:
            label = interval["function"] + " / " + interval["role"] + " / " + (interval.get("call") or "barrier markers")
            grouped.setdefault(label,[]).append(interval)
        operator_rows = [{"label":label,"value":sum(r["relative_end_ticks"]-r["relative_start_ticks"] for r in rows)/1000,
                          "status":"measured","measurement_ids":["census128-step0"],"occurrences":len(rows),
                          "identity_sha256":[r["identity_sha256"] for r in rows]}
                         for label,rows in grouped.items()]
        charts["operator-latency"].update(title="Recorded CTA0 role-leader envelopes — diagnostic", metric="Sum of emitted leader marker spans", unit="nominal us",
            rows=operator_rows + [{"label":"unobserved operators / prefix / tail / D2H", "value":None,"reason":"No timestamps; not zero latency"}],
            aggregation="Sum of CTA0 role-local entry/exit or barrier leader spans grouped by function/role/call; overlapping waits/services are not exclusive operator or whole-model time. PTX nominal ns; resolution/calibration/perturbation pending.",
            recorded_source={"primary":leaders["primary_path"],"source_binding":primary_capture["source_binding"],"cubin_sha256":primary_capture["cubin_sha256"],"physical_uuid":primary_capture["physical_uuid"]})
        charts["operator-latency"].pop("pending_only",None)
        charts["operator-latency"].pop("pending_reason",None)
        def archive_pin(origin):
            path = Path(origin["path"])
            relative = str(path.relative_to(inputs.work)) if path.is_absolute() else str(path)
            snapshot = inputs.archive_pins[relative]
            return {"path":os.path.relpath(inputs.work / snapshot["path"], directory),"sha256":snapshot["sha256"]}
        origin = min(r["raw_start"] for r in leaders["intervals"])
        end = max(r["raw_end"] for r in leaders["intervals"])
        leader_rows = []
        for role in ("consumer","producer"):
            for kind in ("barrier","service"):
                observed = [r for r in leaders["intervals"] if r["role"] == role and r["kind"] == kind]
                leader_rows.append({"label":f"CTA0 {role} " + ("publication/sync" if kind == "barrier" else "MLP gate/up service"),
                    "status":"captured", "intervals":[{"id":"census128-step0-" + str(r["occurrence"]),
                        "start_ns":r["raw_start"]-origin,"end_ns":r["raw_end"]-origin,
                        "kind":"sync" if kind == "barrier" else "tensor","evidence":"inflight_envelope",
                        "cta":r["cta"],"role":r["role"],"identity_sha256":r["identity_sha256"],"raw_points":r["raw_points"]} for r in observed]})
        leader_rows.append({"label":"unobserved GPU prefix/tail/D2H", "status":"unavailable", "intervals":[]})
        charts["graph-trace"].update(title="Recorded CTA0 consumer/producer leaders — diagnostic", scope="observed_role_leaders",
            invocation="Full-census prompt128 / decode index0; one CTA role leaders", rows=leader_rows,
            observed_window_ns=[0,end-origin], observed_raw_origin=origin,
            timebase={"source":primary_capture["clock_source"],"resolution_ns":primary_capture["clock_resolution_ns"],
                      "alignment":"pending","nominal_ns_verified":primary_capture["nominal_nanosecond_unit_verified"]},
            leader_capture={"source_pin":archive_pin({"path":leaders["primary_path"]}),
                            "clock_pin":archive_pin(primary_capture["clock_documentation"]),
                            "health_validated":bool(leaders.get("compact",{}).get("all315_checksums_count_sentinels_mapping_valid")),
                            "observer":"role_local_leader","cta":0,"observed_engine_busy":False,"full_model_window":False},
            aggregation="311 publication/synchronization +56 MLP gate/up role-service sites; plotted actual CTA0 leader envelopes only, not exclusive engine/operator/all-warp time. Clock calibration, missing footer/drop and perturbation remain pending.")
        for key in ("window_ns","pending_only","pending_reason"):
            charts["graph-trace"].pop(key,None)
    event, host = primary(data["events"]), primary(data["host"])
    native_event = next((c for c in reversed(data["native_controls"]) if c.get("prompt_length") == 128 and c["cohort"] == "event_instrumented"), None)
    native_host = next((c for c in reversed(data["native_controls"]) if c.get("prompt_length") == 128 and c["cohort"] == "ordinary_no_events"), None)
    event_value = event["device_tokens_per_second"] if event else native_event["tokens_per_second"] if native_event else None
    event_ids = [data["events"]["measurement_id"]] if event else [native_event["capture_id"]] if native_event else []
    for name in ("performance-vs-milestone", "performance-vs-iteration"):
        point_rows = []
        seen_points = set()
        for control in data["native_controls"]:
            if control.get("prompt_length") == 128 and control.get("cohort") == "event_instrumented":
                key = (control["capture_id"], control["tokens_per_second"])
                if key not in seen_points:
                    seen_points.add(key)
                    point_rows.append({"label":"FlagMega event DEVICE", "x":len(point_rows), "value":control["tokens_per_second"], "status":"measured",
                        "point_id":str(data["selected_source"]) + " baseline" if not point_rows else control["capture_id"], "measurement_ids":[control["capture_id"]]})
        for report in data.get("event_history", []):
            scenario = primary(report)
            if scenario:
                point_rows.append({"label":"FlagMega event DEVICE", "x":len(point_rows), "value":scenario["device_tokens_per_second"], "status":"measured",
                    "point_id":str(data["selected_source"]) + " baseline" if not point_rows else report["measurement_id"], "measurement_ids":[report["measurement_id"]]})
        charts[name].update(x_axis="milestone" if "milestone" in name else "iteration", direction="higher", milestones=[],
            x_labels={str(row["x"]):row["point_id"] for row in point_rows}, rows=point_rows or [{"label": "FlagMega current event DEVICE", "x": 0,
            "value": event_value, "status": "measured",
            "measurement_ids": event_ids,
            "reason": "No qualifying baseline device measurement"}],
            aggregation="Full decode DEVICE; numerical changes only; no achieved optimization milestone",
            cohort=f"{data['selected_source']} / GPU{data['selected_device']['index']} / B1 BF16 128→64 /63steps / event DEVICE; goal pending")
        if event_value is None:
            charts[name].update(pending_only=True, pending_reason="No accepted iteration or milestone exists.")
            historical = next((r for r in reversed(data["historical_events"]) if primary(r)), None)
            native_history = sorted(data.get("historical_native_controls", []),
                                    key=lambda c: c.get("capture_timestamp_utc", ""))
            native_latest = native_history[-1] if native_history else None
            if native_latest and (not historical or native_latest["capture_timestamp_utc"] >= historical.get("timestamp_utc", "")):
                # Use the latest actual measurement cohort, never current
                # state timestamps, and do not join different GPU histories.
                selected_uuid = native_latest["physical_gpu_uuid"]
                rows, unique = [], set()
                for control in native_history:
                    if control["physical_gpu_uuid"] != selected_uuid:
                        continue
                    key = (control["capture_id"], control["source_id"], control["tokens_per_second"])
                    if key in unique:
                        continue
                    unique.add(key)
                    source = control["source_id"].split(":")[0]
                    rows.append({"label": f"Historical {source} GPU{control['physical_gpu_index']}",
                                 "x": len(rows), "value": control["tokens_per_second"], "status": "measured",
                                 "point_id": f"historical {source} baseline" if not rows else control["capture_id"],
                                 "measurement_ids": [control["capture_id"]], "binding": control["binding"],
                                 "runtime_manifest_sha256": control["runtime_manifest_sha256"],
                                 "source_id": control["source_id"], "capture_timestamp_utc": control["capture_timestamp_utc"]})
                charts[name].pop("pending_only", None)
                charts[name]["rows"] = rows
                charts[name]["x_labels"] = {str(row["x"]): row["point_id"] for row in rows}
                charts[name]["aggregation"] = "Historical native event cohort retains its measured source/device; current selected source/device timing pending."
            elif historical:
                source = Path(historical.get("artifact", {}).get("path", "unknown")).parent.name
                device = device_identity(historical)
                charts[name].pop("pending_only", None)
                charts[name]["rows"] = [{"label": f"Historical {source} GPU{device['index']}", "x": 0,
                    "value": primary(historical)["device_tokens_per_second"], "status": "measured",
                    "measurement_ids": [historical["measurement_id"]]}]
                charts[name]["x_labels"] = {"0": f"historical {source}; current pending"}
                charts[name]["aggregation"] = "Historical event cohort retains its actual source/device; current selected source/device metric pending."
    for name in ("performance-vs-milestone", "performance-vs-iteration"):
        # Native controls and request repeats are baseline/measurement evidence,
        # not additional history units. Units require declared revision lineage.
        anchors = [row for row in charts[name]['rows'] if row.get('value') is not None]
        if anchors:
            charts[name]['rows'] = [{**anchors[0], 'x': 0, 'point_id': 'baseline', 'axis_label': 'baseline'}]
            charts[name]['x_labels'] = {'0': 'baseline'}
        if data.get('paired_performance'):
            existing = [r for r in charts[name]['rows'] if r.get('value') is not None]
            units = {}
            for experiment in data['paired_performance']:
                point = experiment['revision_primary']
                units[point['unit_key']] = point
            points = sorted(units.values(), key=lambda p: (p['milestone_id'], p['revision_id']))
            baseline = existing[0] if existing else None
            if baseline is None:
                raise ValueError('starting historical baseline evidence missing; no synthetic history anchor')
            baseline = {**baseline, 'point_id': 'baseline', 'axis_label': 'baseline', 'label': 'FlagMega primary DEVICE',
                        'main_improvement': '起始基线'}
            if name == 'performance-vs-milestone':
                # Open milestones have no result point. A final release is a
                # separate accepted receipt, never inferred from positive gain.
                completed = {}
                for point in points:
                    if point['final_milestone_completed']:
                        completed[point['milestone_id']] = {**point, 'point_id': point['milestone_id'],
                                                          'axis_label': point['milestone_id']}
                points = list(completed.values())
            charts[name]['rows'] = [{**row, 'x': index} for index, row in enumerate([baseline] + points)]
            charts[name]['x_labels'] = {str(r['x']): r.get('axis_label', r.get('point_id', r['label'])) for r in charts[name]['rows']}
            charts[name].pop('pending_only', None)
            charts[name].pop('pending_reason', None)
            charts[name].update(title='One primary result per ' + ('completed milestone' if name.endswith('milestone') else 'revision'),
                cohort='Primary DEVICE only; historical baseline anchor differs from fresh paired gain baseline; HOST/controls/raw trials retained in source evidence',
                aggregation='One result per history unit. TPS = total tokens / total measured seconds; final milestone gates remain pending.',
                uncertainty_method='Original paired round-gain 95% Student-t audit in source packet; no invented TPS error bars',
                paired_measurement_packets=data['paired_performance'])
    history = charts['performance-vs-iteration']
    numerical = {row.get('unit_key'):row for row in history['rows'] if row.get('revision_id')}
    if data.get('history_revisions') and history['rows'][0].get('value') is not None:
        table = [{**history['rows'][0], 'main_improvement':'起始基线'}]
        for revision in data['history_revisions']:
            point = numerical.get(revision['unit_key'])
            table.append({**revision, **point, 'model_status':'整模已测'} if point else revision)
        history.update(revision_table=table, full_model_only=True,
                       title='整模性能随 revision 变化', metric='Full-model decode throughput')
    backend_states = data["native_backend_states"]
    def backend_row(label, key):
        state = backend_states.get(key)
        event = state.get("current_cohorts", {}).get("event_instrumented") if state else None
        return {"label": label, "value": event["DEVICE_TPS"] if event else None, "status": "measured",
                "reason": ("Event DEVICE; " + event["boundary"]["device"] + "; " + event["review_qualification"]) if event else
                          state["reason"] if state else "Frozen native GPU NUM/HOST outcome evidence pending; DEVICE/fair rounds pending",
                "measurement_channel": "DEVICE", "measurement_ids": [event["measurement_id"]] if event else [],
                "boundary": event["boundary"]["device"] if event else "pending",
                "state_evidence": event or state}
    charts["backend-comparison"]["rows"] = [
        {"label": "FlagMega DEVICE", "value": event_value,
         "reason": "device cohort pending", "status": "measured",
         "measurement_ids": event_ids},
        backend_row("vLLM", "vllm"), backend_row("SGLang", "sglang"),
        backend_row("TileFoundry", "tilefoundry"), backend_row("llama.cpp", "llama.cpp"),
        backend_row("FlashRT-derived", "flashrt")]
    charts["backend-device-comparison"] = dict(charts["backend-comparison"])
    charts["backend-device-comparison"].update(title="DEVICE comparison — matched full GPU boundaries pending",
        measurement_channel="DEVICE", aggregation="DEVICE events only; HOST/NCU measurements excluded; formal comparison pending.")
    host_rows = []
    p1_host = diagnostic.get("HOST_diagnostic") if diagnostic else None
    flagmega_host = p1_host["tokens_per_second"] if p1_host else host["host_tokens_per_second"] if host else native_host["tokens_per_second"] if native_host else None
    host_rows.append({"label":("FlagMega P1" if p1_host["binding"]["matching_selected_runtime"] else "Historical FlagMega P1") if p1_host else "FlagMega", "value":flagmega_host,"status":"measured",
        "measurement_channel":"HOST", "measurement_ids":[p1_host["measurement_id"]] if p1_host else [data["host"]["measurement_id"]] if host else [native_host["capture_id"]] if native_host else [],
        "reason":"P1 HOST diagnostic; logging/guard overhead retained; one request, no formal win" if p1_host else "Source-matched ordinary HOST cohort" if flagmega_host is not None else "Current source-matched HOST data pending",
        "boundary":p1_host["boundary"] if p1_host else "synchronous append+sample+per-step token delivery; prefill/first output excluded",
        "source_evidence":p1_host if p1_host else native_host if native_host else data["host"]})
    for label,key in [("vLLM","vllm"),("SGLang","sglang"),("TileFoundry","tilefoundry"),("llama.cpp","llama.cpp"),("FlashRT-derived","flashrt")]:
        state = backend_states.get(key)
        historical_host = state.get("historical_ordinary_HOST") if state else None
        host_state = historical_host or state
        host_rows.append({"label":"Historical " + label + " ordinary HOST" if historical_host else label,"value":host_state.get("HOST_TPS") if host_state else None,"status":"measured", "measurement_channel":"HOST",
            "measurement_ids":[host_state["measurement_id"]] if host_state else [], "reason":host_state["reason"] if host_state else "Native HOST evidence pending",
            "boundary":host_state.get("HOST_boundary","Failed before measurement") if host_state else "pending", "source_evidence":host_state})
        instrumented = state.get("current_cohorts", {}).get("event_instrumented") if state else None
        if instrumented:
            host_rows.append({"label":label + " instrumented HOST", "value":instrumented["HOST_TPS"], "status":"measured",
                "measurement_channel":"HOST", "measurement_ids":[instrumented["measurement_id"]],
                "reason":"Event-instrumented HOST; separate execution from ordinary HOST; " + instrumented["review_qualification"],
                "boundary":instrumented["boundary"]["host"], "source_evidence":instrumented})
    charts["backend-comparison"].update(title="Available HOST diagnostics — formal comparison pending", measurement_channel="HOST",
        metric="Full63 generated intervals / recorded HOST seconds", unit="tokens/s", rows=host_rows, formal_comparison_qualified=False,
        aggregation="HOST only. Different capture dates/configurations and observer overhead; numbers are available diagnostics, not a ranked or matched multi-round win.")

    for name, metric, unit, field, scale in [
        ("hbm-throughput", "Actual native DRAM bytes / counter range seconds", "TB/s", "DRAM_bytes_per_second", 1e12),
        ("hbm-ncu-reported", "NCU-reported DRAM unit percentage (not ordinary full decode utilization)", "%", "reported_dram_pct_peak_sustained_elapsed", 1)]:
        chart = template_data(skill, inputs, directory, "backends", "HBM diagnostic — isolation/normalization pending")
        chart.update(metric=metric, unit=unit, aggregation="Current source/device-bound NCU ranges; ordinary timing is a separate cohort. Historical c10/c11 contended and c9 isolation unproven; >60% not accepted.",
            rows=[{"label": ("current " if r["binding"]["matching_current"] else "historical ") + f"GPU{r['binding']['gpu']} " + str(r["prompt_length"]) + (" main" if "mainkernel" in r["scope"] else " /63"),
                   "value": r[field]/scale, "status": "measured", "measurement_ids": [(r.get("capture_id") or ("c0010" if "mainkernel" in r["scope"] else "c0009" if "fullprimary" in r["scope"] else "c0011")) + "-range-" + r["NCU_range_id"]],
                   "isolation": r["isolation"], "scope": r["scope"], "binding": r["binding"]} for r in data["native"].get("rows", [])])
        if not chart["rows"]:
            chart.update(pending_only=True, pending_reason="Actual native counter evidence unavailable", rows=[{"label": "HBM", "value": None, "reason": "capture pending"}])
        if diagnostic:
            prefix = "latest working" if diagnostic["binding"]["matching_selected_runtime"] else "historical"
            chart.pop("pending_only", None)
            chart.update(title="P2 hardware-counter diagnostic — unreviewed", metric="NCU-reported DRAM bandwidth" if unit == "TB/s" else "NCU-reported DRAM sustained-peak percentage",
                aggregation="Five P2 hardware10 /63 ranges; unreviewed diagnostic, not ordinary utilization; Historical P3 SourceCounters failed; SC excluded from new capture scope; >60% goal not accepted.",
                rows=[{"label": f"{prefix} P2 GPU{r['binding']['gpu']} {r['prompt_length']} /63", "value": r[field]/scale,
                       "status": "measured", "measurement_ids": ["threeprobe-P2-range-" + str(r["action_index"])],
                       "binding": r["binding"], "scope": r["scope"], "metric_count": r["metric_count"],
                       "duration_ms": r["duration_ms"], "raw_duration_value": r["raw_duration_value"],
                       "raw_duration_unit_label": r["raw_duration_unit_label"]} for r in diagnostic["rows"]])
        if hardware:
            source_field = 'actual_total_bytes_by_fixed_GPU_elapsed_Bps' if unit == 'TB/s' else 'NCU_reported_DRAM_percent'
            chart.pop('pending_only', None)
            chart.pop('pending_reason', None)
            chart.update(title=hardware['capture_id'] + ' current H800 hardware diagnostic — unreviewed',
                aggregation='Five complete own64/63 hardware ranges; profiled diagnostic, not ordinary timing or formal performance. Historical P2 retained separately; reported and fixed-elapsed normalization remain distinct; >60% not accepted.',
                rows=[{'label': f"current {hardware['capture_id']} GPU{r['physical_index']} {r['prompt_length']} /63",
                       'value': r[source_field] / scale, 'status': 'measured',
                       'measurement_ids': [hardware['capture_id'] + '-range-' + str(r['action_index'])],
                       'binding': hardware['binding'], 'scope': hardware['scope'],
                       'duration_ms': r['gpu_elapsed_duration_ns'] / 1e6,
                       'raw_duration_value': r['required_raw_metrics']['gpu__time_duration.sum']['value'],
                       'raw_duration_unit_label': r['elapsed_unit_evidence']['API_label_preserved'],
                       'actual_CSV_duration_unit': r['elapsed_unit_evidence']['actual_CSV_unit'],
                       'fixed_peak_Bps': r['fixed_peak_Bps'],
                       'fixed_elapsed_peak_percent': r['fixed_GPU_elapsed_bytes_by_fixed_peak_percent']}
                      for r in hardware['rows']])
        charts[name] = chart
    fixed_chart = template_data(skill, inputs, directory, 'backends', 'Current fixed elapsed/peak HBM diagnostic — unreviewed')
    fixed_chart.update(metric='Actual DRAM bytes / full profiled GPU elapsed / fixed clock-bus peak', unit='%',
                       aggregation='Hardware range bytes and elapsed; fixed observed clock/bus ceiling. No ordinary performance point or accepted >60% claim.',
                       rows=[{'label': f"current {hardware['capture_id']} GPU{r['physical_index']} {r['prompt_length']} /63",
                              'value': r['fixed_GPU_elapsed_bytes_by_fixed_peak_percent'], 'status': 'measured',
                              'measurement_ids': [hardware['capture_id'] + '-range-' + str(r['action_index'])],
                              'binding': hardware['binding'], 'scope': hardware['scope'], 'fixed_peak_Bps': r['fixed_peak_Bps']}
                             for r in hardware['rows']] if hardware else [{'label': 'Current fixed elapsed HBM', 'value': None, 'reason': 'Source-matching validated hardware sidecar unavailable'}])
    if not hardware:
        fixed_chart.update(pending_only=True, pending_reason='No source-matching validated hardware bytes/time/peak sidecar; historical reports remain separate.')
    charts['hbm-fixed-elapsed'] = fixed_chart
    status = template_data(skill, inputs, directory, "operators", "Current working state — measurements and gaps")
    status.update(pending_only=True, pending_reason="Status rows are evidence labels, not measured operator durations.", rows=[
        {"label": "Numerical", "value": None, "reason": data["numerical"]},
        {"label": "Event DEVICE", "value": None, "reason": f"{event['device_tokens_per_second']:.3f} TPS / {event['device']['median_ms']:.6f} ms median; review pending" if event else f"{native_event['tokens_per_second']:.3f} TPS / {native_event['mean_step_ns']/1e6:.6f} ms mean; {native_event['capture_id']}; review pending" if native_event else "pending"},
        {"label": "Ordinary HOST", "value": None, "reason": f"{host['host_tokens_per_second']:.3f} TPS / {host['host']['median_ms']:.6f} ms median; no harness events" if host else f"{native_host['tokens_per_second']:.3f} TPS / {native_host['mean_step_ns']/1e6:.6f} ms mean; {native_host['capture_id']}" if native_host else "pending"},
        {"label": "Native trace", "value": None, "reason": data["trace"].get("decision", "activation pending")},
        {"label": "GPU / HBM", "value": None, "reason": data["native"].get("status", "pending")},
        {"label": "Dashboard flow", "value": None, "reason": data["phase"]}])
    if diagnostic:
        error = diagnostic["error"]
        stage = error["primary_error"]["stage"]
        probe_status = "; ".join(p["probe_id"] + (" full5 own64/63 passed" if p["passed_scope"] else " failed exit" + str(p["exit_code"])) for p in diagnostic["probes"])
        error_status = (f"P3 SourceCounters first marked prompt{stage['prompt_length']}/decode{stage['decode_index']} "
                        f"{error['primary_error']['class']}; exit{error['actual_NCU_exit']}; "
                        f"supervisor={json.dumps(diagnostic['frozen_supervisor_error_flag'])} / raw={json.dumps(diagnostic['actual_NCU_error'])}; historical SourceCounters failure retained; SC excluded from new capture scope; original fault PC unknown")
        cu_status = "; ".join(f"{p['probe_id']}: original-visible regs{p['actual_original_function']['attributes']['registers']} "
                               f"local{p['actual_original_function']['attributes']['local_bytes']} resident{p['actual_original_function']['resident_blocks_per_SM']} "
                               f"launch{p['actual_launch']['grid'][0]}x{p['actual_launch']['block'][0]}/shared{p['actual_launch']['dynamic_shared']}/params{p['native_parameter_count']}"
                               for p in diagnostic["probes"]) + "; profiler-private patched resources/PC unknown"
        status["rows"].extend([{"label": "Threeprobe NUM", "value": None, "reason": probe_status},
                               {"label": "P3 actual error", "value": None, "reason": error_status},
                               {"label": "CUDA scope", "value": None, "reason": cu_status}])
    charts["status"] = status
    status["rows"].extend({"label":key,"value":None,"reason":value["reason"]} for key,value in progress.items())
    current_ncu = next((r for r in data["native"].get("rows", [])
                        if r["binding"]["matching_current"] and r.get("prompt_length") == 128), None)
    ncu_label = (f"Current {current_ncu['capture_id']} NCU /63: {current_ncu['DRAM_bytes_per_second']/1e12:.8f} TB/s; "
                 f"fixed-peak bytes/time {fixed_peak_ratio(current_ncu):.6f}%; NCU reported {current_ncu['reported_dram_pct_peak_sustained_elapsed']:.2f}%; review pending") if current_ncu else "Current full-range NCU evidence pending"
    controls_label = (f"Current mean HOST {native_host['tokens_per_second']:.3f} TPS; event DEVICE {native_event['tokens_per_second']:.3f} TPS; separate cohorts, review pending") if native_host and native_event else "Current matched native timing controls pending"
    current_pair = next((p for p in reversed(data.get('paired_performance', []))
                         if p['revision_primary']['runtime_manifest_sha256'] == data['source_binding']['runtime_manifest_sha256']), None)
    if current_pair:
        comparisons = [c for c in current_pair['contrasts'] if c['prompt_length'] == 128 and
                       (c['boundary'] == 'DEVICE' or c['cohort'] == 'ordinary_no_events')]
        controls_label = 'Current paired primary: ' + '; '.join(
            f"{c['boundary']} {c['cohort']} gain {100*c['mean_gain']:.6f}%, 95%CI [{100*c['two_sided_95pct_interval'][0]:.6f}%, {100*c['two_sided_95pct_interval'][1]:.6f}%]" +
            (' inconclusive (CI includes zero)' if c['two_sided_95pct_interval'][0] <= 0 <= c['two_sided_95pct_interval'][1] else '')
            for c in comparisons)
    if diagnostic:
        primary_diagnostic = next(r for r in diagnostic["rows"] if r["prompt_length"] == 128)
        ncu_label = (f"P2 diagnostic /63: NCU reported {primary_diagnostic['reported_dram_pct_peak_sustained_elapsed']:.8f}%; "
                     f"NCU bandwidth {primary_diagnostic['DRAM_bytes_per_second']/1e12:.8f} TB/s; {primary_diagnostic['duration_ms']:.8f} ms; "
                     f"source/runtime/UUID binding: {diagnostic['binding']}; unreviewed, not ordinary utilization")
    if hardware:
        primary_hardware = next(r for r in hardware['rows'] if r['prompt_length'] == 128)
        ncu_label = (f"Current {hardware['capture_id']} /63: reported {primary_hardware['NCU_reported_DRAM_percent']:.8f}%; "
                     f"fixed elapsed/peak {primary_hardware['fixed_GPU_elapsed_bytes_by_fixed_peak_percent']:.8f}%; "
                     f"{primary_hardware['actual_total_bytes_by_fixed_GPU_elapsed_Bps']/1e12:.8f} TB/s; source/native: {hardware['binding']}; "
                     "unreviewed profiled hardware, not ordinary performance; old P2 remains historical")
    for chart in charts.values():
        chart["provenance"] = [{"path": os.path.relpath(inputs.work / row["path"], directory), "sha256": row["sha256"]} for row in inputs.archive_pins.values()]
        chart["working_status"] = {"dashboard_latest_row_utc": data["dashboard_latest_utc"], "dashboard_sha256": inputs.pins.get("dashboard.md", {}).get("sha256"),
                                   "dashboard_rows": data["dashboard"].get("rows", []),
                                   "input_errors": inputs.errors, "acceptance": data["acceptance"],
                                   "visible_context": [f"Selected source: {data['selected_source']} | device: {data['selected_device']} | NUM: {data['numerical']}",
                                       "Dashboard SHA256: " + inputs.pins.get("dashboard.md", {}).get("sha256", "unavailable"),
                                       "Latest actual dashboard row: " + data['dashboard_latest_utc'] + " | actor: " + data['agent_id'],
                                       "Historical devices: " + str([device_identity(r) for r in data['historical_events'][-2:]]),
                                       "Runtime manifest SHA256: " + str(data['source_binding']['runtime_manifest_sha256']),
                                       "Generated program SHA256: " + str(data['source_binding']['generated_source_sha256']),
                                       controls_label, ncu_label,
                                       "Native origins: " + "; ".join(sorted(set(
                                           f"{r['binding']['source'][:28]} / GPU{r['binding']['gpu']} {r['binding']['uuid'][:16]} / " +
                                           ("current" if r['binding']['matching_current'] else "historical")
                                           for r in data['native'].get('rows', []))))]}
        if chart is charts["backend-comparison"] or chart is charts["backend-device-comparison"]:
            flash_cohorts = backend_states.get("flashrt", {}).get("current_cohorts", {})
            flash = (flash_cohorts.get("ordinary_no_events") if chart["measurement_channel"] == "HOST"
                     else flash_cohorts.get("event_instrumented"))
            if flash:
                chart["working_status"]["visible_context"].append(flash["review_qualification"])
        if chart is charts["backend-comparison"]:
            chart["working_status"]["visible_context"].extend([
                "HOST diagnostics: " + "; ".join(f"{r['label']} {r['value']:.6f} TPS / {r['measurement_ids']}" for r in host_rows if r.get("value") is not None),
                "HOST boundary: full63 synchronous decode + per-step token delivery; FlagMega P1 logs/guards retained; backend capture dates/configs are different.",
                "GPU3 UUID: " + str(diagnostic['binding']['uuid'] if diagnostic else next((v.get('physical_gpu_uuid') for v in backend_states.values() if v.get('physical_gpu_uuid')), 'pending')) + "; source/version/boundary pins are in the immutable chart packet; no formal winner."])
        if chart is charts["graph-trace"] or chart is charts["operator-latency"]:
            chart["working_status"]["visible_context"].extend(value["reason"] for value in progress.values())
            if leaders and leaders.get("compact"):
                compact = leaders["compact"]
                chart["working_status"]["visible_context"].extend([
                    f"Actual enabled recording: {compact['frame_count']} frames / {compact['timestamp_count']} timestamps; 0 point-order errors; checksums/count/sentinels/SM-role mappings validated by unchanged decoder; interpretation review pending.",
                    "Observed leader-window means (nominal ns, excludes unobserved prefix/tail/D2H): " + "; ".join(
                        f"P{row['prompt_length']} {row['observed_leader_window_delta']['mean']:.3f}" for row in sorted(compact['contexts'],key=lambda row:row['prompt_length'])),
                    "Source9919 / cubin23aa / GPU3 " + leaders['primary']['physical_uuid'] + "; native footer/drop fields absent; all12-warp activity, actual clock resolution/alignment and perturbation pending."])
        if chart is status:
            chart["attempts_table"] = [{"id": j.get("id", j.get("job_id", "unknown")), "source_id": j.get("source_id", j.get("scope", "source unavailable")),
                                        "status": j.get("status", "pending"), "exit_code": j.get("exit_code"),
                                        "flow_steps": j.get("flow_steps"), "owner": j.get("owner")} for j in data["jobs"]]
    if data.get("tir_views"):
        view = data["tir_views"]
        for role, name in (("graph", "graph-trace"), ("operators", "operator-latency")):
            existing = charts[name]
            captured = dict(view["charts"][role])
            if not view["matching_selected_source"]:
                captured["title"] = "Historical source " + captured["title"]
                captured["cohort"] = "Selected " + str(data["selected_source"]) + ": trace pending; historical GPU0 native1099/d4b0 caller capture"
                if role == "graph":
                    captured["title"] = "Current trace pending | Historical source TIR caller overview"
            captured["template_binding"] = existing["template_binding"]
            captured["provenance"] = captured["provenance"] + existing["provenance"]
            captured["working_status"] = dict(existing["working_status"])
            captured["working_status"]["visible_context"] = [
                "TIR diagnostic matching_selected_source=" + str(view["matching_selected_source"]).lower() +
                "; " + ("selected diagnostic source" if view["matching_selected_source"] else "historical measured source; selected ordinary source has no recapture"),
                "Selected working source: " + str(data["selected_source"]) + "; current trace " + ("captured diagnostic" if view["matching_selected_source"] else "pending"),
                "Actual TIR diagnostic source: " + view["descriptor"]["diagnostic_artifact"]["sha256"],
                "Native: " + str(view["descriptor"]["native"]),
                "Latest dashboard: " + data["dashboard_latest_utc"] + " / " + inputs.pins["dashboard.md"]["sha256"],
                "CTA0 one actual SM; full623 first-step role calls admitted; raw timer units; physical calibration/host alignment/full315/perturbation pending."]
            charts[name] = captured
    return charts


def render_graph_presentation(chart, renderer, input_path):
    """Keep canonical overview geometry; expose a dense real capture with a same-clock detail."""
    if chart.get("scope") != "observed_TIR_callers":
        return renderer.render(chart)
    renderer.validate(chart, input_path)
    ns = "http://www.w3.org/2000/svg"
    tag = lambda name: "{" + ns + "}" + name
    ET.register_namespace("", ns)
    root = ET.fromstring(renderer.render(chart))
    callsites = sorted({event["callsite"] for row in chart["rows"] for event in row["intervals"]})
    # Categorical TIR identity colors only: preserve unknown hardware kind and dashed envelopes.
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
               "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39",
               "#7b4173", "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d", "#006d2c"]
    mapping = {site: (f"K{i + 1:02d}", palette[i % len(palette)]) for i, site in enumerate(callsites)}

    def text(parent, x, y, value, size=11, **attributes):
        element = ET.SubElement(parent, tag("text"), {"x": str(x), "y": str(y), "font-size": str(size),
                                "fill": "#0f172a", **attributes})
        element.text = value

    def identify(panel, labels=False):
        annotations = ET.SubElement(panel, tag("g"), {"font-family": "Inter,DejaVu Sans,sans-serif"})
        for rect in list(panel.iter(tag("rect"))):
            title = rect.find(tag("title"))
            site = title.text.split(" / layer ", 1)[0] if title is not None and title.text else None
            if site not in mapping:
                continue
            code, color = mapping[site]
            rect.set("fill", color)
            rect.set("fill-opacity", ".23")
            rect.set("stroke", color)
            rect.set("data-tir-callsite", site)
            rect.set("data-callsite-code", code)
            if labels and float(rect.get("width", 0)) >= 28:
                text(annotations, float(rect.get("x")) + float(rect.get("width")) / 2,
                     float(rect.get("y")) + 13, code, 9, **{"text-anchor": "middle"})

    identify(root)
    root.set("data-graph-panel", "complete-overview")
    height = int(root.get("height"))
    layer_events = [event for row in chart["rows"] for event in row["intervals"] if event.get("layer") == 0]
    if layer_events:
        detail = copy.deepcopy(chart)
        for row in detail["rows"]:
            row["intervals"] = [event for event in row["intervals"] if event.get("layer") == 0]
            if row["status"] == "captured" and not row["intervals"]:
                row["status"] = "unavailable"
        window = [min(e["start_raw_units"] for e in layer_events), max(e["end_raw_units"] for e in layer_events)]
        detail.update(title="Layer 0 detail | Same CTA/SM capture and all12 role tracks",
                      observed_window_raw_units=window,
                      invocation=f"Layer0; selected actual window {window[0]}..{window[1]} raw units",
                      cohort="Detail of the overview dataset; original caller endpoints retained; colors identify TIR callsites only",
                      working_status={"visible_context": [
                          "Selected layer0 only; full28 overview above retains every captured interval and the original full observed window.",
                          "Dashed envelopes are inclusive physical caller spans; unknown hardware kind remains unknown; no engine-activity claim."]})
        renderer.validate(detail, input_path)
        panel = ET.fromstring(renderer.render(detail))
        panel.set("y", str(height + 12))
        panel.set("data-graph-panel", "layer0-detail")
        meta = ET.Element(tag("metadata"), {"id": "layer0-detail-provenance"})
        meta.text = json.dumps({"presentation_only": True, "selection": {"layer": 0},
                              "observed_window_raw_units": window,
                              "overview_observed_window_raw_units": chart["observed_window_raw_units"],
                              "shared_span_dataset": chart["shared_span_dataset"], "timebase": chart["timebase"],
                              "interval_ids": [e["id"] for e in layer_events], "physical_role_tracks": len(chart["rows"])})
        panel.insert(0, meta)
        identify(panel, labels=True)
        root.append(panel)
        height += int(panel.get("height")) + 12
    legend = ET.SubElement(root, tag("g"), {"font-family": "Inter,DejaVu Sans,sans-serif", "data-graph-panel": "callsite-legend"})
    text(legend, 24, height + 25, "Actual TIR callsite colors | K codes label readable detail bars", 16)
    text(legend, 24, height + 46, "Identity colors only; dashed bars are inclusive caller envelopes. Hardware kind remains unknown; clocks remain raw/un-calibrated.")
    for i, site in enumerate(callsites):
        x, y = 24 + (i % 2) * 735, height + 67 + (i // 2) * 20
        code, color = mapping[site]
        ET.SubElement(legend, tag("rect"), {"x": str(x), "y": str(y - 11), "width": "16", "height": "12", "fill": color})
        text(legend, x + 23, y, code + " " + site, 10)
    height += 91 + ((len(callsites) + 1) // 2) * 20
    root.set("height", str(height))
    root.set("viewBox", f"0 0 1500 {height}")
    return ET.tostring(root, encoding="unicode") + "\n"


def reconcile_deleted_status(work, agent_id):
    """Honor the explicit user deletion without relaxing other managed-file checks."""
    policy_path = work / "benchmark/current-figure-policy.json"
    if not policy_path.exists():
        return None
    policy = json.loads(policy_path.read_text())
    deletion_pin = policy["root_status_deletion"]
    deletion_path = work / deletion_pin["path"]
    if not deletion_path.resolve().is_relative_to(work.resolve()) or sha(deletion_path) != deletion_pin["sha256"]:
        raise ValueError("Explicit status deletion receipt changed or missing")
    deletion = json.loads(deletion_path.read_text())
    if (policy.get("schema") != "flagmega.current-figure-policy/v1" or policy.get("publish_root_status") is not False or
            deletion.get("schema") != "flagmega.root-explicit-user-deletion/v1" or deletion.get("agent_id") != "/root" or
            deletion.get("user_request") != "删除figures/status.svg"):
        raise ValueError("No valid explicit user status-deletion policy")
    pointer = work / ".local/views/root-figures.json"
    if not pointer.exists():
        return None
    with (pointer.parent / "root-figures.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(pointer.read_text())
        managed = previous.get("managed_files", previous.get("files", []))
        removed = [row for row in managed if row["path"] == "figures/status.svg"]
        if not removed:
            if (work/"figures/status.svg").exists():
                raise ValueError("Unmanaged status file appeared after explicit deletion")
            return None
        if len(removed) != 1 or removed[0]["sha256"] != deletion["prior"]["sha256"] or (work/"figures/status.svg").exists():
            raise ValueError("Explicit deleted status identity/absence mismatch")
        for row in managed:
            if row["path"] != "figures/status.svg" and sha(work/row["path"]) != row["sha256"]:
                raise ValueError("Another managed figure changed during deletion reconciliation")
        archive = work / ".local/working/figures" / ("explicit-status-deletion-" + str(time.time_ns()))
        archive.mkdir(parents=True)
        (archive/"prior-root-view.json").write_bytes(pointer.read_bytes())
        (archive/"prior-root-manifest.json").write_bytes((work/"figures/manifest.json").read_bytes())
        view = {key:value for key,value in previous.items() if key != "managed_files"}
        view.update(sync_id=archive.name, generation=previous["generation"] + 1, archive=str(archive.relative_to(work)),
                    observed_at_utc=utc(), agent_id=agent_id, explicit_user_deletion=deletion_pin)
        view["files"] = [row for row in view["files"] if row["path"] != "figures/status.svg"]
        dump(archive/"root-manifest.json", view)
        view["managed_files"] = view["files"] + [{"path":"figures/manifest.json","sha256":sha(archive/"root-manifest.json"),"generated_metadata":True}]
        dump(archive/"root-view.json", view)
        receipt = {"schema":"flagmega.explicit-user-deletion-reconcile/v1","actor":agent_id,"observed_utc":utc(),
                   "deletion_receipt":deletion_pin,"only_removed_path":"figures/status.svg","other_managed_files_verified":True,
                   "prior_view_sha256":sha(archive/"prior-root-view.json"),"new_view_sha256":sha(archive/"root-view.json"),
                   "status_file_absent":True,"no_payload_or_performance_mutation":True}
        dump(archive/"receipt.json", receipt)
        atomic(work/"figures/manifest.json", (archive/"root-manifest.json").read_text())
        atomic(pointer, (archive/"root-view.json").read_text())
        return str((archive/"receipt.json").relative_to(work))


def regular_working_pin(work, pin):
    """Bind publication to a regular target; retain declared paths as origins."""
    declared = Path(pin["path"])
    origin = declared if declared.is_absolute() else work / declared
    target = origin.resolve(strict=True)
    base = work.resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise ValueError("working input target is outside workspace or not a file")
    before = target.stat()
    if sha(target) != pin["sha256"]:
        raise ValueError("working input identity changed during path binding")
    after = target.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or
            origin.resolve(strict=True) != target):
        raise ValueError("working input target changed during path binding")
    return {"path": str(target.relative_to(base)), "sha256": pin["sha256"]}


def publication_input_pin(inputs, pin):
    # A captured management input is published from the same immutable bytes
    # used by its parser; native/source inputs retain their original live guard.
    origin = pin['path']
    selected = inputs.archive_pins[origin] if origin in inputs.captured else pin
    return regular_working_pin(inputs.work, selected)


def record_completed_render(work, frame, agent_id):
    receipt = frame / 'render-receipt.json'
    marker = {'schema': 'flagmega.completed-live-render/v1', 'agent_id': agent_id,
              'snapshot_id': frame.name, 'receipt': {'path': str(receipt.relative_to(work)),
              'sha256': sha(receipt), 'bytes': receipt.stat().st_size}}
    dump(work / '.local/reports/live-watcher/completed-render.json', marker)
    return marker


def previous_render_snapshots(inputs):
    """Only a complete figures+readme commit advances the reusable input cache."""
    marker = inputs.read('.local/reports/live-watcher/completed-render.json')
    bootstrap = marker is None
    if bootstrap:
        status = inputs.read('.local/reports/live-watcher/status.json') or {}
        completed = status.get('last_success')
        if not completed:
            return {}
        target = (inputs.work / completed['receipt']).resolve(strict=True)
        marker = {'schema': 'flagmega.completed-live-render/v1', 'agent_id': status['agent_id'],
                  'snapshot_id': completed['snapshot_id'], 'receipt': {'path': str(target.relative_to(inputs.work)),
                  'sha256': sha(target), 'bytes': target.stat().st_size}}
    if marker.get('schema') != 'flagmega.completed-live-render/v1':
        raise ValueError('completed render marker schema mismatch')
    receipt = pinned_workspace_object(inputs, marker['receipt'])
    frame = (inputs.work / marker['receipt']['path']).resolve(strict=True).parent
    if (frame.parent != (inputs.work / '.local/reports').resolve() or not frame.name.startswith('live-') or
            receipt.get('schema') != 'flagmega.live-render/v1' or receipt.get('status') != 'synced' or
            receipt.get('snapshot_id') != marker['snapshot_id'] or frame.name != marker['snapshot_id']):
        raise ValueError('previous render input reuse requires a complete render receipt')
    for kind in ('figures', 'readme'):
        path = Path(receipt['sync_receipts'][kind]).resolve(strict=True)
        sync = inputs.read(str(path.relative_to(inputs.work)))
        if (not path.is_relative_to(inputs.work / '.local/working' / kind) or
                sync.get('schema') != 'flagmega.working-sync-receipt/v1' or sync.get('status') != 'synced' or
                sync.get('kind') != kind or sync.get('sync_id') != frame.name + '-' + kind):
            raise ValueError('complete render must bind both successful working syncs')
    for output in receipt['outputs']:
        source = (frame / output['path']).resolve(strict=True)
        if not source.is_relative_to(frame) or sha(source) != output['sha256']:
            raise ValueError('completed render output pin changed')
    snapshots = frame / 'input-snapshots'
    candidates = {}
    for pin in receipt['input_pins']:
        path = inputs.work / pin['path']
        if path.is_relative_to(snapshots):
            target = path.resolve(strict=True)
            if path.is_symlink() or target != path or not target.is_file():
                raise ValueError('previous immutable render input must be a regular snapshot')
            candidates[str(path.relative_to(snapshots))] = {'path': target, 'sha256': pin['sha256']}
    if bootstrap:
        record_completed_render(inputs.work, frame, marker['agent_id'])
    return candidates


def archive_render_inputs(inputs, directory, workspace, candidates):
    inputs.archive_pins = {}
    for original, pin in list(inputs.pins.items()):
        source = inputs.work / original
        snapshot = source if source.is_relative_to(directory) else directory / 'input-snapshots' / original
        if snapshot != source:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if original in inputs.captured:
                snapshot.write_bytes(inputs.captured[original])
            else:
                prior = candidates.get(original)
                candidate = prior['path'] if prior and prior['sha256'] == pin['sha256'] else None
                workspace.copy_immutable_snapshot(source, snapshot, immutable_candidate=candidate,
                                                  expected_sha256=pin['sha256'])
        if sha(snapshot) != pin['sha256']:
            raise ValueError('Input changed before immutable provenance snapshot')
        inputs.archive_pins[original] = {'path': str(snapshot.relative_to(inputs.work)), 'sha256': sha(snapshot)}


def render_live(work, skill, agent_id=None):
    if not agent_id:
        raise ValueError("Supply actual --agent-id; fixture invocations must name their fixture actor")
    lock_dir = work / ".local/reports/live-watcher"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sid = "live-" + str(time.time_ns())
        directory = work / ".local/reports" / sid
        directory.mkdir()
        deletion_reconciliation = reconcile_deleted_status(work, agent_id)
        data, inputs = collect(work, skill)
        data["agent_id"] = agent_id
        figure_policy = inputs.read("benchmark/current-figure-policy.json")
        if figure_policy:
            inputs.read(figure_policy["root_status_deletion"]["path"])
        inputs.pins["benchmark/render_results.py"] = {"path": "benchmark/render_results.py", "sha256": sha(__file__)}
        renderer = skill / "scripts/render_workspace_figures.py"
        script_hash = sha(renderer)
        renderer_snapshot = directory / "template-sources" / renderer.name
        atomic(renderer_snapshot, renderer.read_text())
        inputs.pins[str(renderer_snapshot.relative_to(work))] = {"path": str(renderer_snapshot.relative_to(work)), "sha256": sha(renderer_snapshot)}
        support = renderer.parent / "render_kernel_swimlane.py"
        atomic(directory / "template-sources" / support.name, support.read_text())
        support_snapshot = directory / "template-sources" / support.name
        inputs.pins[str(support_snapshot.relative_to(work))] = {"path": str(support_snapshot.relative_to(work)), "sha256": sha(support_snapshot)}
        for name, pin in inputs.external.items():
            snapshot = directory / "template-sources" / name
            snapshot.write_bytes(Path(pin["path"]).read_bytes())
            if sha(snapshot) != pin["sha256"]:
                raise ValueError("Dashboard parser changed while rendering")
            inputs.pins[str(snapshot.relative_to(work))] = {"path": str(snapshot.relative_to(work)), "sha256": pin["sha256"]}
        candidates = previous_render_snapshots(inputs)
        sys.path.insert(0, str(skill / 'scripts'))
        workspace_spec = importlib.util.spec_from_file_location('immutable_workspace_snapshots', skill / 'scripts/workspace.py')
        workspace = importlib.util.module_from_spec(workspace_spec)
        workspace_spec.loader.exec_module(workspace)
        archive_render_inputs(inputs, directory, workspace, candidates)
        inputs.pins.update({p["path"]: p for p in inputs.archive_pins.values()})
        # Dashboard can link the previous published figure manifest. Preserve
        # those exact bytes historically; the publication itself replaces it.
        for published in ("figures/manifest.json", "readme.manifest.json", "readme.md",
                          ".local/reports/live-watcher/completed-render.json", ".local/reports/live-watcher/status.json"):
            inputs.pins.pop(published, None)
        sys.path.insert(0, str(renderer.parent))
        spec = importlib.util.spec_from_file_location("live_maintained_renderer", renderer)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        charts = build_charts(data, inputs, skill, directory)
        public_charts = {name: chart for name, chart in charts.items() if name != "status"}
        for name, chart in public_charts.items():
            path = directory / (name + ".json")
            dump(path, chart)
            module.validate(chart, path)
            atomic(directory / (name + ".svg"), render_graph_presentation(chart, module, path) if name == "graph-trace" else module.render(chart))
        dump(directory / "data.json", data)
        timestamp = utc()
        e, h = primary(data["events"]), primary(data["host"])
        ne = next((c for c in reversed(data["native_controls"]) if c.get("prompt_length") == 128 and c["cohort"] == "event_instrumented"), None)
        nh = next((c for c in reversed(data["native_controls"]) if c.get("prompt_length") == 128 and c["cohort"] == "ordinary_no_events"), None)
        et = f"{e['device_tokens_per_second']:.6f} TPS / {e['device']['median_ms']:.6f} ms median" if e else f"{ne['tokens_per_second']:.6f} TPS / {ne['mean_step_ns']/1e6:.6f} ms mean ({ne['capture_id']})" if ne else "pending"
        ht = f"{h['host_tokens_per_second']:.6f} TPS / {h['host']['median_ms']:.6f} ms median" if h else f"{nh['tokens_per_second']:.6f} TPS / {nh['mean_step_ns']/1e6:.6f} ms mean ({nh['capture_id']})" if nh else "pending"
        current_pair = next((p for p in reversed(data.get('paired_performance', []))
                             if p['revision_primary']['runtime_manifest_sha256'] == data['source_binding']['runtime_manifest_sha256']), None)
        if current_pair:
            current_ids = set(current_pair['revision_primary']['measurement_ids'])
            candidate = next(p['point_id'].split(' ')[0] for p in current_pair['points'] if set(p['measurement_ids']) == current_ids)
            et = f"{current_pair['revision_primary']['value']:.6f} TPS / three rotated rounds"
            host_point = next(p for p in current_pair['points'] if p['boundary'] == 'HOST' and p['point_id'].startswith(candidate + ' '))
            ht = f"{host_point['value']:.6f} TPS / three rotated ordinary rounds"
        performance_state = ("原三轮匹配计时与独立收益审核已完成；event开销控制与最终验收仍待完成。" if current_pair and current_pair['review'] else
                             "原三轮匹配计时已完成，独立收益审核状态待匹配；event开销控制与最终验收仍待完成。" if current_pair else
                             "独立性能审核、event开销控制和多轮匹配仍待完成。")
        sync_ids = {kind: sid + "-" + kind for kind in ("figures", "readme")}
        num_text = "五种原始上下文的64个输出全部匹配独立 native vLLM 参考。" if data["num_receipt"] and data["num_receipt"].get("NUM_qualified") else "当前选定source的NUM证据待匹配。"
        hbm_rows = data["native"].get("rows", [])
        hbm_text = "Actual native counters pending."
        if hbm_rows:
            hbm_text = "\n".join(f"- {'current' if r['binding']['matching_current'] else 'historical'} {r['binding']} / {r['prompt_length']} / {r['scope']}: {r['DRAM_bytes_per_second']/1e12:.4f} TB/s；NCU reported {r['reported_dram_pct_peak_sustained_elapsed']:.2f}%；{'producer clock-checked fixed' if 'fixed_peak_Bps' in r else 'provisional fixed'} {fixed_peak_ratio(r):.2f}%；{r['isolation']}。" for r in hbm_rows)
        diagnostic = data.get("latest_diagnostic")
        diagnostic_text = ""
        counter_note = "Reported DRAM percentage使用NCU硬件单位归一化；GPU3 c0014的bytes/time/固定峰值分母已有producer时钟核对，独立审核仍待办。历史GPU2分母与隔离另列；这些counter执行不能替代ordinary decode利用率。"
        if diagnostic:
            prefix = "Latest working" if diagnostic["binding"]["matching_selected_runtime"] else "Historical source"
            diagnostic_text = prefix + " threeprobe（producer diagnostics，独立审核待办）：\n\n" + "\n".join("- " + row["reason"] for row in charts["status"]["rows"] if row["label"] in ("Threeprobe NUM", "P3 actual error", "CUDA scope"))
            hbm_text = "\n".join(f"- {prefix} P2 GPU{r['binding']['gpu']} / prompt{r['prompt_length']} /63: NCU reported **{r['reported_dram_pct_peak_sustained_elapsed']:.8f}%**；native reported {r['DRAM_bytes_per_second']/1e12:.8f} TB/s；duration **{r['duration_ms']:.8f} ms**；{r['metric_count']} metrics；raw duration {r['raw_duration_value']} / API label `{r['raw_duration_unit_label']}`。" for r in diagnostic['rows'])
            counter_note = ("P2 values are unreviewed NCU-only diagnostics, not ordinary utilization or formal performance. "
                            "Raw duration API label `s` stores nanoseconds here; pinned bytes/bandwidth and raw/1e9 dimensional proof confirms 140–148 ms. "
                            "Historical P3 has no usable SourceCounters report. SourceCounters/PC/SW are excluded from the new requested capture; genuine GPU/WG timing is being prepared and remains unmeasured. Source binding: " + str(diagnostic["binding"]) + ".")
        hardware = data.get('current_hardware')
        if hardware:
            historical_hbm = hbm_text
            hbm_text = '\n'.join(f"- Current {hardware['capture_id']} GPU{r['physical_index']} / prompt{r['prompt_length']} /63: reported **{r['NCU_reported_DRAM_percent']:.8f}%**; fixed elapsed/peak **{r['fixed_GPU_elapsed_bytes_by_fixed_peak_percent']:.8f}%**; {r['actual_total_bytes_by_fixed_GPU_elapsed_Bps']/1e12:.8f} TB/s; {r['gpu_elapsed_duration_ns']/1e6:.8f} ms; raw API unit `{r['elapsed_unit_evidence']['API_label_preserved']}`, base CSV `ns`." for r in hardware['rows'])
            hbm_text += '\n\nHistorical hardware evidence retained:\n' + historical_hbm
            counter_note = (f"Current {hardware['capture_id']} is one complete hardware10 collection; all five own64/63 trajectories passed. "
                            f"Time-weighted full-five fixed elapsed/peak {hardware['time_weighted']['fixed_peak_percent']:.8f}%. "
                            "Cache/clock controls none. Bytes/time belongs to profiled ranges, not ordinary timing; no formal gain/HBM60/goal acceptance. "
                            "SourceCounters/PC/SW/PM excluded. Mapping, original CU/graph21, units and sampled isolation are pinned; unsampled activity and hardware-busy attribution remain limited. " + str(hardware['binding']))
            if hardware.get('boundary_diagnostics'):
                boundary = hardware['boundary_diagnostics']
                counter_note += (f" Maximum observed monitor gap {boundary['maximum_observed_poll_gap_ns']/1e9:.9f} s. " +
                                 boundary['interpretation'])
        current_control_text = "\n".join(f"- 当前匹配native control {c['capture_id']} / GPU{c['physical_gpu_index']} {c['physical_gpu_uuid']} / prompt{c.get('prompt_length')}: {c['cohort']}；{c.get('tokens_per_second', 'pending')} TPS；{c.get('mean_step_ns', 'pending')} ns **mean**（不是median）；{c.get('status', 'review pending')}。" for c in data['native_controls']) or "当前匹配native controls待完成。"
        comparator_text = "\n".join("- " + r["label"] + ": " + (f"**HOST {r['value']:.6f} TPS**；" if r.get("value") is not None else "") + r.get("reason", "measurement review pending") + "; boundary: " + r.get("boundary", "pending") for r in charts["backend-comparison"]["rows"] if r["label"] != "FlagMega DEVICE")
        device_comparator_text = "\n".join("- " + row["label"] + f": **DEVICE {row['value']:.6f} TPS**；" + row["reason"] + "; measurement: " + ", ".join(row.get("measurement_ids", []))
                                            for row in charts["backend-device-comparison"]["rows"] if row.get("value") is not None)
        capture_text = ("已捕获315帧/46,650,240个leader时间戳；primary128/index0实际CTA0 consumer/producer publication/service envelopes已用于graph/operator图。完整GPU起止、全部warp、busy/drop/时钟校准/扰动仍待验证。" if data.get("census_leaders") else "") + ("现有图是 P1 实际 HOST API 时间线，含异步调用/同步/传回的host envelope，GPU内部/逐算子/WG仍未捕获；hardware-only GPU采集准备中，不要求SourceCounters，等待root新采集请求与device租约。" if diagnostic and diagnostic.get("HOST_timeline") and not data.get("census_leaders") else "") + ("真实captured roles：" + ", ".join(data["captures"]) if data["captures"] else "已显示实际 recorded role-leader 图；完整模型/全部warp活动仍pending" if data.get("census_leaders") else "Graph/operator/WG硬件时钟和活动仍缺有效捕获") + "；native trace状态：" + data["trace"].get("decision", "pending")
        if data.get("tir_views"):
            view = data["tir_views"]["descriptor"]
            capture_text = (("Current" if data["tir_views"]["matching_selected_source"] else "Historical source") +
                            " TIR caller diagnostic (matching_selected_source=" + str(data["tir_views"]["matching_selected_source"]).lower() +
                            ")：primary128/index0、完整623个role-call（567leaf+56parents）严格配对；graph/latency同一CTA0/SM数据集，producer/consumer多warp轨道；父调用折叠且不计入排序。Native " + str(view["native"]) + "；raw globaltimer units，真实resolution、跨SM/host校准、perturbation和full315仍pending；不是exclusive engine busy或ordinary性能。")

        progress_text = "\n\n".join(value["reason"] for value in data.get("current_progress", {}).values())
        paired_text = ''
        for experiment in data.get('paired_performance', []):
            paired_text += '\n最新真实三轮配对计时（各 subject 均保留567个计时间隔，HOST/DEVICE分开）：\n\n'
            paired_text += '\n'.join('- ' + point['point_id'] + f": **{point['value']:.9f} TPS**；" + point['review_status'] for point in experiment['points'])
            paired_text += '\n\n' + '\n'.join(f"- {c['boundary']} {c['cohort']} candidate vs {c['baseline']}: paired耗时收益 {100*c['mean_gain']:.6f}%；原95%CI [{100*c['two_sided_95pct_interval'][0]:.6f}%, {100*c['two_sided_95pct_interval'][1]:.6f}%]。" + (' inconclusive：CI含0，不能宣称显著收益。' if c['two_sided_95pct_interval'][0] <= 0 <= c['two_sided_95pct_interval'][1] else '') for c in experiment['contrasts'] if c['prompt_length'] == 128 and (c['boundary'] == 'DEVICE' or c['cohort'] == 'ordinary_no_events'))
            paired_text += '\n\n全部ordinary HOST轮次及原CI保留；historical compiler/mapped library未知；实际observed compute-only监控不证明continuous0.25s隔离。完整logits/state、fair5、315trace、HBM60与winning milestone仍pending。 [原始配对包](' + experiment['packet']['path'] + ')。\n'
        text = f'''# Qwen3-1.7B BF16 / NVIDIA H800

当前工作视图采用 skill 的 maintained workspace-figure templates；独立审核决定验收，不阻止最新状态展示。

选定source：{data['selected_source']}，选定device：{data['selected_device']}。主场景128→64：匹配当前source/device的预分配 CUDA event 完整设备 cohort（event instrumented）为 **{et}**；ordinary no-event HOST cohort 为 **{ht}**。完整设备边界包含输入、28层、完整151936词表 LM head、greedy、状态更新和 token 传回，排除 prefill/首输出。{num_text} {performance_state}这些 cohort 不同于 NCU instrumented replay。

{current_control_text}

{progress_text}

{paired_text}

Graph trace采用canonical模板保留完整overview，并从同一真实数据集显示layer0细节及21个callsite映射。历史raw caller窗口不代表当前source的完整模型trace；当前trace采集状态见图首。

![Graph trace](figures/graph-trace.svg)

![Operator latency](figures/operator-latency.svg)

![Milestones](figures/performance-vs-milestone.svg)

![Iterations](figures/performance-vs-iteration.svg)

![Backend comparison](figures/backend-comparison.svg)

{comparator_text}

已有 HOST 诊断数字展示在上图，完整 source / boundary / capture cohort 保存在图的 source 包。不同日期、配置和observer开销尚未匹配，不能据此宣称胜出。DEVICE comparison单独显示：

![DEVICE comparison](figures/backend-device-comparison.svg)

{device_comparator_text}

{diagnostic_text}

实际counter证据（各行保持原scope/device，隔离与分母资格单列）：

{hbm_text}

{counter_note} 验收状态：{data['acceptance']}。

![HBM throughput diagnostic](figures/hbm-throughput.svg)

![NCU reported percentage diagnostic](figures/hbm-ncu-reported.svg)

{capture_text}。[采集前置与边界](.local/agents/dispatch/fix_current_figures_user_feedback/capture-prerequisites.md)。不生成虚假的时间轴或零延迟。资源记录：{data['resources']}；CPU/UI继续更新。

[当前artifact/vectorized/distributed IR](.local/views/root-artifacts.json) · [历史GPU2测量](.local/agents/dispatch/measure_repaired_baseline/handoff.json) · [当前GPU3签署诊断](.local/agents/dispatch/profile_idle_h800/handoff.json) · [唯一管理表](dashboard.md)

渲染时间：{timestamp}（不是测量时间）。Dashboard最新行时间：{data['dashboard_latest_utc']}。Dashboard SHA256：`{inputs.pins.get('dashboard.md', {}).get('sha256', 'unavailable')}`。本次只读管理表快照：

{data['dashboard'].get('table', 'dashboard pending')}

Working-sync：[figures receipt](.local/working/figures/{sync_ids['figures']}/receipt.json) · [readme receipt](.local/working/readme/{sync_ids['readme']}/receipt.json) · [figure manifest](figures/manifest.json) · [readme manifest](readme.manifest.json)。模板binding、renderer SHA和raw provenance保存在immutable [.local/reports/{sid}](.local/reports/{sid}/render-receipt.json)。

重现：`python benchmark/render_results.py --live --agent-id {agent_id}`。启动：`python benchmark/render_results.py --start-watcher --agent-id {agent_id}`。监听状态：`python benchmark/render_results.py --watch-status`。停止：`python benchmark/render_results.py --stop-watcher --agent-id {agent_id}`。独立fixture须指定其真实fixture actor。
'''
        atomic(directory / "readme.md", text)
        # Signed evidence and archived chart origins keep their declared paths.
        # Only the live publication pins use the current regular-file location.
        working_pins = [publication_input_pin(inputs, pin) for pin in inputs.pins.values()]
        working_manifest = regular_working_pin(work, data["source_binding"]["source_manifest"])
        input_origins = [{"declared_path": pin["path"], **regular}
                         for pin, regular in zip(inputs.pins.values(), working_pins)]
        dump(directory / "working-input-path-bindings.json", {
            "schema": "flagmega.live-input-path-bindings/v1", "inputs": input_origins,
            "selected_manifest_origin": data["source_binding"]["source_manifest"],
            "selected_manifest": working_manifest})
        receipts = {}
        for kind in ("figures", "readme"):
            pointer = work / ".local/views" / ("root-" + kind + ".json")
            files = [name + ".svg" for name in public_charts] if kind == "figures" else ["readme.md"]
            record = {"schema": "flagmega.working-sync/v1", "sync_id": sync_ids[kind], "kind": kind,
                      "agent_name": "graph profiler", "agent_id": agent_id, "source_id": str(data["selected_source"] or sid),
                      "source_manifest": working_manifest,
                      "status": {"phase": data["phase"], "acceptance": data["acceptance"], "input_errors": inputs.errors,
                                 "flow": data["dashboard"], "dashboard_sha256": inputs.pins.get("dashboard.md", {}).get("sha256"),
                                 "selected_source": data["selected_source"], "selected_device": data["selected_device"],
                                 "native": data["native"].get("status", "pending"), "template_renderer_sha256": script_hash},
                      "base_view_sha256": sha(pointer) if pointer.exists() else None,
                      "input_pins": working_pins,
                      "files": [{"source": str((directory / name).relative_to(work)), "path": "figures/" + name if kind == "figures" else "readme.md",
                                 "sha256": sha(directory / name)} for name in files]}
            record_path = directory / ("sync-" + kind + ".json")
            dump(record_path, record)
            result = subprocess.run([sys.executable, str(skill / "scripts/workspace.py"), "sync", "--work-dir", str(work),
                                     "--agent", "graph profiler", "--agent-id", agent_id, "--kind", kind, "--record", str(record_path)], capture_output=True, text=True)
            dump(directory / ("sync-" + kind + "-command.json"), {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise RuntimeError("Working sync failed: " + result.stderr)
            receipts[kind] = result.stdout.strip()
        receipt = {"schema": "flagmega.live-render/v1", "snapshot_id": sid, "status": "synced", "generated_utc": timestamp,
                   "renderer": {"path": str(renderer), "sha256": script_hash}, "template_bindings": {name: chart["template_binding"] for name, chart in public_charts.items()},
                   "input_pins": list(inputs.pins.values()), "sync_receipts": receipts,
                   "explicit_status_deletion_reconciliation": deletion_reconciliation,
                   "outputs": [{"path": name + ".svg", "sha256": sha(directory / (name + ".svg"))} for name in public_charts]}
        dump(directory / "render-receipt.json", receipt)
        record_completed_render(work, directory, agent_id)
        return {"snapshot_id": sid, "receipt": str((directory / "render-receipt.json").relative_to(work))}


def process_identity(pid):
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        argv = [v.decode() for v in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]]
        return {"pid": pid, "start_ticks": int(tail[19]), "ppid": int(tail[1]), "session": int(tail[3]),
                "exe": os.readlink(f"/proc/{pid}/exe"), "argv": argv}
    except (OSError, ValueError, IndexError):
        return None


def watch_status(work):
    path = work / ".local/reports/live-watcher/status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    identity = process_identity(status.get("pid", -1))
    alive = bool(identity and identity["start_ticks"] == status.get("start_ticks") and
                 str(Path(__file__).resolve()) in identity["argv"] and "--watch" in identity["argv"])
    return {"alive": alive, "observed_utc": utc(), "identity": identity, "status": status}


def fingerprint(work, skill):
    paths = {work / "dashboard.md", work / ".local/task.yaml", work / ROOT_ARTIFACT_VIEW, Path(__file__).resolve(),
             work / "benchmark/current-diagnostic-input.json", work / "benchmark/current-figure-policy.json",
             work / "benchmark/current-native-backend-input.json"}
    paths.update((skill / "assets/workspace-figures").glob("*.json"))
    paths.update([skill / "scripts/render_workspace_figures.py", skill / "scripts/render_kernel_swimlane.py", skill / "scripts/workspace.py", skill / "scripts/dashboard.py"])
    dispatch = work / ".local/agents/dispatch"
    try:
        registry = json.loads((work / "benchmark/current-native-backend-input.json").read_text())
        for entry in registry["cohorts"]:
            for key in ("report", "job_receipt", "control_config", "source_manifest", "source_verdict"):
                path = work / entry[key]["path"]
                if path.resolve().is_relative_to(work.resolve()): paths.add(path)
            config = json.loads((work / entry["control_config"]["path"]).read_text())
            paths.update(work / pin["path"] for pin in config["source_pins"])
    except (OSError, ValueError, TypeError, KeyError):
        pass  # Explicit descriptor stays watched even if a dependency is missing.
    for assignment, filenames in {
        "execute_hardware_trace_full_ring_num_gpu3_v2":["summary.json","handoff.json"],
        "review_runtime_v3_actual_implementation_graph_contract":["approval.json"],
        "implement_flashrt_sm90_cmake_and_native_build_v1":["handoff-partial-000.json","job.json"],
        "execute_hardware_trace_census_recording_gpu3":["primary-128-step0-intervals.json","decoded-315-compact.json","capture-summary.json"]}.items():
        paths.update(dispatch / assignment / name for name in filenames)
    paths.update((dispatch / "prepare_remaining_hardware_trace_phase_bindings/full-census-recording").glob("**/terminal.json"))
    paths.update(live_measurements(work))
    for pattern in ("**/figure-graph.json", "**/figure-operators.json", "**/native-diagnostic-data-*.json",
                    "**/manifest.json", "**/artifact.json", "**/generated_kernels.py", "**/results/*.json",
                    "**/profiles/**/*", "**/figures/**/*.json"):
        paths.update(evidence_files(work, pattern))
    for pattern in (".local/jobs/**/receipt.json", ".local/agents/dispatch/**/handoff*.json",
                    ".local/agents/dispatch/**/jobs/**/receipt.json"):
        paths.update(work.glob(pattern))
    # Hardware sidecars live in their authentic assignment, outside profiles/.
    # Watch the exact data/report/source pins so drift cannot leave a stale view.
    for path in list(paths):
        if not path.name.startswith('handoff') or path.suffix != '.json':
            continue
        try:
            receipt = json.loads(path.read_text())
            if (not isinstance(receipt, dict) or receipt.get('schema') != 'flagmega.own-scoped-handoff/v1' or
                    not isinstance(receipt.get('metrics'), dict) or 'report' not in receipt):
                continue
            for key in ('metrics', 'report', 'source_provenance'):
                origin = Path(receipt[key]['path'])
                if not origin.is_absolute():
                    origin = work / origin
                if origin.resolve().is_relative_to(work.resolve()):
                    paths.add(origin)
        except (OSError, ValueError, TypeError, KeyError):
            pass  # The handoff itself remains watched when invalid.
    # These frozen progress receipts pin facts outside the timing/capture tree.
    # Watch their exact dependencies so a missing/drifted summary is visible,
    # without treating CPU or disabled-recording NUM output as a measurement.
    for assignment in ("execute_hardware_trace_full_census_num_gpu3", "implement_native_reference_runtime_v3",
                       "implement_flashrt_dense_bf16_python_v1"):
        handoff = work / ".local/agents/dispatch" / assignment / "handoff.json"
        paths.add(handoff)
        try:
            receipt = json.loads(handoff.read_text())
            if not isinstance(receipt, dict): continue
            for key in ("summary", "source_manifest", "selfcheck_receipt", "real_checkpoint_headers", "CPU_regression_receipt"):
                pin = receipt.get(key, {})
                if not isinstance(pin, dict) or not pin.get("path"): continue
                path = Path(pin["path"])
                if not path.is_absolute(): path = work / path
                if path.resolve().is_relative_to(work.resolve()): paths.add(path)
        except (OSError, ValueError, TypeError):
            pass  # Missing/invalid handoff itself remains watched.
    try:
        policy = json.loads((work / "benchmark/current-figure-policy.json").read_text())
        paths.add(work / policy["root_status_deletion"]["path"])
    except (OSError, ValueError, TypeError, KeyError):
        pass
    # Diagnostic inputs include frozen helper/error/proof files outside the
    # profiling tree. Track their exact pins so drift refreshes visible status.
    try:
        descriptor = json.loads((work / "benchmark/current-diagnostic-input.json").read_text())
        pins = [descriptor[key] for key in ("handoff", "dimensional_proof", "P1_stages", "P1_observations_source") if key in descriptor]
        for pin in pins:
            path = work / pin["path"]
            if not path.resolve().is_relative_to(work.resolve()):
                parts = path.parts
                path = work.joinpath(*parts[parts.index(".local"):])
            paths.add(path)
            if pin == descriptor["handoff"]:
                handoff = json.loads(path.read_text())
                pins.extend(value for value in handoff.values() if isinstance(value, dict) and "path" in value and "sha256" in value)
    except (OSError, ValueError, TypeError, KeyError):
        pass  # Missing/invalid descriptor and declared pin paths stay watched.
    # Raw chart pins may live outside profiles/, and source selection can name
    # any contained build directory. Watch those exact dependencies too.
    descriptors = {work / ROOT_ARTIFACT_VIEW} | evidence_files(work, "**/figure-graph.json") | evidence_files(work, "**/figure-operators.json")
    for path in descriptors:
        try:
            data = json.loads(path.read_text())
            for pin in data.get("provenance", []):
                raw = (path.parent / pin["path"]).resolve()
                if raw.is_relative_to(work.resolve()):
                    paths.add(raw)
            if path == work / ROOT_ARTIFACT_VIEW:
                for pin in data.get("files", []):
                    paths.add(work / pin["path"])
                selected = data.get("source_manifest", {})
                if selected.get("path"):
                    manifest = work / selected["path"]
                    if manifest.resolve().is_relative_to(work.resolve()):
                        paths.add(manifest)
                        selected_data = json.loads(manifest.read_text())
                        runtime = selected_data.get("input_identities", {}).get("runtime_artifact")
                        if runtime:
                            artifact = work / runtime / "artifact.json"
                            if artifact.resolve().is_relative_to(work.resolve()):
                                paths.add(artifact)
        except (OSError, ValueError, TypeError, KeyError):
            pass  # Missing/invalid descriptor itself remains a watched input.
    items = []
    for path in sorted(paths):
        try:
            value = path.stat()
            items.append((str(path), value.st_size, value.st_mtime_ns))
        except FileNotFoundError:
            items.append((str(path), None, None))
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def watch(work, skill, agent_id, interval):
    if not agent_id:
        raise ValueError("Watcher needs the actual caller --agent-id")
    directory = work / ".local/reports/live-watcher"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "watcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = True
        def stop(*_):
            nonlocal running
            running = False
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        executed_sha = sha(__file__)
        executed_sources = {str(path): sha(path) for path in [Path(__file__).resolve(),
            skill / "scripts/dashboard.py", skill / "scripts/workspace.py", skill / "scripts/render_workspace_figures.py",
            skill / "scripts/render_kernel_swimlane.py", *sorted((skill / "assets/workspace-figures").glob("*.json"))]}
        status = {"schema": "flagmega.live-watcher/v1", **process_identity(os.getpid()), "agent_id": agent_id,
                  "started_utc": utc(), "interval_seconds": interval, "cpu_only": True,
                  "executed_script_sha256": executed_sha, "executed_sources": executed_sources,
                  "management_input": "dashboard.md", "state": "running", "last_success": None}
        json_write = lambda: dump(directory / "status.json", status)
        previous = None
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            while running:
                if any(not Path(path).is_file() or sha(path) != value for path, value in executed_sources.items()):
                    status["state"] = "source_changed_restart_required"
                    break
                current = fingerprint(work, skill)
                if current != previous:
                    command = [sys.executable, str(Path(__file__).resolve()), "--live", "--work-dir", str(work),
                               "--skill-dir", str(skill), "--agent-id", agent_id]
                    result = subprocess.run(command, capture_output=True, text=True)
                    status.update(last_exit_code=result.returncode, last_stderr=result.stderr[-5000:], last_stdout=result.stdout[-5000:],
                                  last_event_fingerprint=current, last_attempt_utc=utc(), state="running" if result.returncode == 0 else "render_failed_retrying")
                    print(json.dumps({"time": utc(), "argv": command, "exit_code": result.returncode,
                                      "stdout": result.stdout, "stderr": result.stderr}), flush=True)
                    if result.returncode == 0:
                        previous = current
                        status["last_success"] = json.loads(result.stdout)
                        status["last_success_utc"] = utc()
                status["heartbeat_utc"] = utc()
                json_write()
                time.sleep(interval)
        finally:
            if status["state"] != "source_changed_restart_required":
                status["state"] = "stopped"
            status["ended_utc"] = utc()
            json_write()


def start_watcher(work, skill, agent_id, interval):
    if not agent_id:
        raise ValueError("Watcher launch needs actual --agent-id")
    old = watch_status(work)
    if old["alive"]:
        if old["status"].get("agent_id") != agent_id:
            raise ValueError("Existing service belongs to a different actor")
        return old
    directory = work / ".local/reports/live-watcher"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--watch", "--work-dir", str(work),
               "--skill-dir", str(skill), "--agent-id", agent_id, "--interval", str(interval)]
    with (directory / "watcher.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   close_fds=True, start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    _OWNED_CHILDREN[process.pid] = process
    for _ in range(100):
        result = watch_status(work)
        if result["alive"] and result["identity"]["pid"] == process.pid and result["status"].get("last_success"):
            dump(directory / ("launch-" + str(time.time_ns()) + ".json"), {"schema": "flagmega.watcher-launch/v1",
                 "argv": command, "started_utc": utc(), "detached_session": True, "observed": result,
                 "log": str(directory / "watcher.log")})
            return result
        if process.poll() is not None:
            raise RuntimeError("Watcher exited; inspect retained watcher.log")
        time.sleep(.1)
    raise RuntimeError("Watcher has no verified successful first refresh; inspect status/log and same PID")


def stop_watcher(work, agent_id):
    result = watch_status(work)
    if not result["alive"]:
        return result
    if not agent_id or result["status"].get("agent_id") != agent_id:
        raise ValueError("Only the explicitly named owned watcher actor may stop this service")
    owned_pid = result["identity"]["pid"]
    os.kill(owned_pid, signal.SIGTERM)
    for _ in range(100):
        result = watch_status(work)
        if not result["alive"]:
            child = _OWNED_CHILDREN.pop(owned_pid, None)
            if child:
                child.wait(timeout=5)
            return result
        time.sleep(.1)
    raise RuntimeError("Owned watcher has not terminated; preserve and inspect the same PID")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=["graph", "kernel"])
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--agent-id", help="actual independent caller identity; fixtures must name their fixture actor")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--start-watcher", action="store_true")
    parser.add_argument("--watch-status", action="store_true")
    parser.add_argument("--stop-watcher", action="store_true")
    parser.add_argument("--interval", type=float, default=2)
    args, forwarded = parser.parse_known_args(argv)
    skill = args.skill_dir or find_skill()
    if args.interval < .25:
        parser.error("watch interval must be at least .25 seconds")
    if args.watch_status:
        print(json.dumps(watch_status(args.work_dir.resolve())))
        return 0
    if args.stop_watcher:
        print(json.dumps(stop_watcher(args.work_dir.resolve(), args.agent_id)))
        return 0
    if args.watch:
        watch(args.work_dir.resolve(), skill, args.agent_id, args.interval)
        return 0
    if args.start_watcher:
        print(json.dumps(start_watcher(args.work_dir.resolve(), skill, args.agent_id, args.interval)))
        return 0
    if args.live:
        print(json.dumps(render_live(args.work_dir.resolve(), skill, args.agent_id)))
        return 0
    if not args.renderer:
        parser.error("select --live or --renderer")
    filename = "render_workspace_figures.py" if args.renderer == "graph" else "render_kernel_swimlane.py"
    return subprocess.run([sys.executable, str(skill / "scripts" / filename), *forwarded], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
