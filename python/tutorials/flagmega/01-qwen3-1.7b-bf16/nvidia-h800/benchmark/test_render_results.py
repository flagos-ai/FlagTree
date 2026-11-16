"""Working-view regressions: stale sources, missing evidence and actor identity."""
import json
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import subprocess
import sys
import time
from unittest.mock import patch
from types import SimpleNamespace
import render_results as renderer


class LiveViewContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        (self.work / "benchmark").mkdir()
        shutil.copyfile(renderer.__file__, self.work / "benchmark/render_results.py")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, value):
        path = self.work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_artifact_metadata_selection_uses_external_current_view(self):
        self.dashboard('External artifact metadata fixture')
        self.write('artifacts/manifest.json', {'source_id':'obsolete-root-metadata'})
        self.write('.local/views/root-artifacts.json', {'source_id':'actual-current-view', 'files':[]})
        data, inputs = renderer.collect(self.work)
        self.assertEqual(data['selected_source'], 'actual-current-view')
        self.assertIn('.local/views/root-artifacts.json', inputs.pins)
        self.assertNotIn('artifacts/manifest.json', inputs.pins)

    def test_latest_artifact_view_selects_pinned_whole_model(self):
        self.write('.local/views/model-stage.json', {'source_id':'actual-model-source'})
        model = {'path':'.local/views/model-stage.json', 'sha256':renderer.sha(self.work/'.local/views/model-stage.json'),
                 'bytes':(self.work/'.local/views/model-stage.json').stat().st_size}
        self.write('.local/views/latest-stage.json', {'schema':'flagmega.latest-artifacts/v1',
                    'whole_model_source_manifest':model, 'working_operator_revision_id':'r016'})
        published = {'source_id':'model-plus-standalone', 'source_manifest':{
            'path':'.local/views/latest-stage.json', 'sha256':renderer.sha(self.work/'.local/views/latest-stage.json'),
            'bytes':(self.work/'.local/views/latest-stage.json').stat().st_size}}
        selected = renderer.current_model_artifacts(renderer.Inputs(self.work), published)
        self.assertEqual(selected['source_id'], 'actual-model-source')
        self.assertEqual(selected['source_manifest'], model)
        self.write('.local/views/model-stage.json', {'source_id':'changed-source'})
        with self.assertRaisesRegex(ValueError, 'full pin changed'):
            renderer.current_model_artifacts(renderer.Inputs(self.work), published)

    def test_revision_table_keeps_unmeasured_rows_out_of_full_model_curve(self):
        scripts = renderer.find_skill()/'scripts'
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location('revision_figures', scripts/'render_workspace_figures.py')
            figures = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(figures)
        finally:
            sys.path.remove(str(scripts))
        data = json.loads((renderer.find_skill()/'assets/workspace-figures/history.json').read_text())
        data['milestones'] = []
        data['rows'] = [{'label':'model', 'x':0, 'point_id':'baseline', 'value':100, 'status':'qualified', 'main_improvement':'起始基线'},
                        {'label':'model', 'x':1, 'point_id':'r001', 'value':110, 'status':'qualified', 'main_improvement':'整模融合'}]
        missing = lambda rid: {'point_id':rid, 'value':None, 'model_status':'整模未测 / 未集成',
                               'main_improvement':'未集成内核实验：共享输入暂存'}
        data['revision_table'] = [data['rows'][0], missing('r000'), data['rows'][1], missing('r002')]
        figures.validate(data, self.work/'history.json')
        svg = figures.render(data)
        self.assertEqual(svg.count('class="performance-point-marker"'), 2)
        self.assertEqual(svg.count('class="performance-point-table-row"'), 4)
        self.assertIn('整模未测 / 未集成', svg)
        self.assertIn('主要优化点', svg)
        data['revision_table'][-1]['value'] = 110
        with self.assertRaisesRegex(ValueError, 'table.*point'):
            figures.validate(data, self.work/'history.json')

    def dashboard(self, description, evidence=""):
        (self.work / "dashboard.md").write_text(
            "| ID | 时间（UTC，记录/完成） | 状态 | Flow步骤 | 做了什么 / 正在做什么 / 还要做什么 | 负责人 | 证据 | 前置条件 / 下一步 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            f"| F001 | {renderer.utc()} | 进行中 | 2 | {description} | actual fixture | {evidence} | CPU fixture only |\n")

    def test_current_artifact_excludes_newer_timestamp_from_old_source(self):
        self.write("artifacts/manifest.json", {"source_id": "b002"})
        self.dashboard("new-source-CPU-build")
        self.write(".local/baseline/profiles/graph/old/measurement.json", {
            "schema": "flagmega.model-measurement/v1", "backend": "flagmega", "status": "measured",
            "timestamp_utc": "2099-01-01", "cohort": "event_instrumented",
            "artifact": {"path": str(self.work / ".local/baseline/build/b001/runtime-artifact")}, "scenarios": []})
        data, _ = renderer.collect(self.work)
        self.assertIsNone(data["events"], "old-source timing cannot become current b002 timing")

    def test_missing_evidence_does_not_claim_NUM_or_NCU_values(self):
        self.dashboard("fixture-pending")
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/missing-evidence")
        text = (self.work / ".local/reports" / result["snapshot_id"] / "readme.md").read_text()
        self.assertNotIn("全部匹配独立 native vLLM", text)
        self.assertNotIn("1.5075 TB/s", text)
        self.assertNotIn("reported44.97%", text)

    def test_explicit_fixture_actor_is_preserved(self):
        self.dashboard("fixture-actor-test")
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/reviewer/live-view")
        directory = self.work / ".local/reports" / result["snapshot_id"]
        record = json.loads((directory / "sync-figures.json").read_text())
        self.assertEqual(record["agent_id"], "/fixture/reviewer/live-view")

    def test_previous_publication_is_snapshotted_without_self_dependency(self):
        self.write("figures/manifest.json", {"fixture": "actual prior publication bytes"})
        self.dashboard("Prior output link is historical input", "[prior](figures/manifest.json)")
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/prior-output")
        directory = self.work / ".local/reports" / result["snapshot_id"]
        record = json.loads((directory / "sync-readme.json").read_text())
        pins = {p["path"] for p in record["input_pins"]}
        self.assertNotIn("figures/manifest.json", pins)
        self.assertTrue(any(p.endswith("input-snapshots/figures/manifest.json") for p in pins))

    def test_archived_chart_survives_real_dashboard_and_wrapper_changes(self):
        self.dashboard("fixture-before")
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/history")
        directory = self.work / ".local/reports" / result["snapshot_id"]
        self.dashboard("fixture-after-real-edit")
        (self.work / "benchmark/render_results.py").write_text("# changed actual fixture source\n")
        result = subprocess.run([sys.executable, str(directory / "template-sources/render_workspace_figures.py"),
                                 str(directory / "graph-trace.json"), "--output", str(directory / "replayed.svg")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((directory / "replayed.svg").read_bytes(), (directory / "graph-trace.svg").read_bytes())

    def test_same_opaque_id_different_runtime_bytes_excludes_real_old_GPU_data(self):
        original_work = Path(renderer.__file__).resolve().parents[1]
        report = json.loads((original_work / ".local/baseline/profiles/graph/flagmega-b001-events-000/measurement.json").read_text())
        self.write(".local/task.yaml", {"environment": {"gpu_index": 2}, "fixture": "same label and device, different actual runtime"})
        self.write("artifacts/manifest.json", {"source_id": "b001", "runtime_manifest_sha256": "actual-new-runtime"})
        self.write(".local/baseline/profiles/graph/old/measurement.json", report)
        data, _ = renderer.collect(self.work)
        self.assertIsNone(data["events"])
        self.assertEqual(data["historical_events"][0]["artifact"]["manifest_sha256"], report["artifact"]["manifest_sha256"])

    def test_old_real_native_packet_is_explicitly_historical_for_new_source(self):
        original_work = Path(renderer.__file__).resolve().parents[1]
        packet = json.loads((original_work / ".local/baseline/figures/kernel/native-diagnostic-data-001.json").read_text())
        self.write("artifacts/manifest.json", {"source_id": "fixture-C-new-source"})
        self.write(".local/baseline/results/native-diagnostic-data-001.json", packet)
        data, _ = renderer.collect(self.work)
        self.assertTrue(data["native"]["rows"])
        self.assertTrue(all(not row["binding"]["matching_current"] for row in data["native"]["rows"]))
        self.assertEqual(data["native_controls"], [])

    def test_dashboard_authenticates_current_GPU3_without_management_JSON(self):
        original_work = Path(renderer.__file__).resolve().parents[1]
        packet = json.loads((original_work / ".local/baseline/results/native-diagnostic-data-002.json").read_text())
        code = "artifacts/generated_kernels.py"
        original_code = next(original_work / item["path"] for item in json.loads((original_work / "artifacts/manifest.json").read_text())["files"] if Path(item["path"]).name == "generated_kernels.py")
        (self.work / "artifacts").mkdir()
        shutil.copyfile(original_code, self.work / code)
        runtime = self.work / ".local/baseline/build/b001/runtime-artifact/artifact.json"
        runtime.parent.mkdir(parents=True)
        shutil.copyfile(original_work / ".local/baseline/build/b001/runtime-artifact/artifact.json", runtime)
        self.write("artifacts/manifest.json", {"source_id": "b001", "runtime_manifest_sha256": packet["runtime_manifest_sha256"], "files": [{"path": code, "sha256": renderer.sha(self.work / code)}]})
        self.write(".local/baseline/results/native-diagnostic-data-002.json", packet)
        self.dashboard("Actual copied native source/device evidence; fixture management row", "[native](.local/baseline/results/native-diagnostic-data-002.json)")
        data, inputs = renderer.collect(self.work)
        self.assertEqual(data["selected_device"]["index"], "3")
        self.assertEqual(data["selected_device"]["uuid"], packet["physical_gpu_uuid"])
        self.assertEqual(len(data["native_controls"]), 10)
        self.assertEqual(data["source_binding"]["source_manifest"]["sha256"], packet["runtime_manifest_sha256"])
        self.assertNotIn(".local/state.json", inputs.pins)
        self.assertFalse((self.work / ".local/state.json").exists())

    def test_archive_copies_never_become_live_measurements_or_events(self):
        original_work = Path(renderer.__file__).resolve().parents[1]
        report = json.loads((original_work / ".local/baseline/profiles/graph/flagmega-b001-events-000/measurement.json").read_text())
        self.write(".local/baseline/profiles/graph/real/measurement.json", report)
        self.dashboard("Actual fixture archive discovery check")
        before = renderer.fingerprint(self.work, renderer.find_skill())
        self.write(".local/reports/archived/input-snapshots/.local/baseline/profiles/graph/real/measurement.json", report)
        self.assertEqual(renderer.fingerprint(self.work, renderer.find_skill()), before)
        data, _ = renderer.collect(self.work)
        self.assertEqual(len(data["reports"]), 1)

    def test_dashboard_only_change_keeps_one_curve_point_and_summary_row(self):
        self.test_dashboard_authenticates_current_GPU3_without_management_JSON()
        results = []
        for description in ("fixture pending review", "fixture review status updated, no performance change"):
            self.dashboard(description, "[actual native](.local/baseline/results/native-diagnostic-data-002.json)")
            with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
                results.append(renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/point-table"))
        for result in results:
            directory = self.work/".local/reports"/result["snapshot_id"]
            for name in ("performance-vs-milestone", "performance-vs-iteration"):
                chart = json.loads((directory/(name+".json")).read_text())
                svg = (directory/(name+".svg")).read_text()
                self.assertNotIn("attempts_table", chart)
                self.assertEqual(svg.count('class="performance-point-marker"'), 1)
                self.assertEqual(svg.count('class="performance-point-table-row"'), 1)
                self.assertNotIn("Milestone improvements", svg)
                self.assertNotIn("Actual working source / validation", svg)

    def test_source_NUM_publication_without_timing_retains_actual_native_history(self):
        self.test_dashboard_authenticates_current_GPU3_without_management_JSON()
        original_work = Path(renderer.__file__).resolve().parents[1]
        packet = json.loads((original_work / ".local/baseline/results/native-diagnostic-data-002.json").read_text())
        relative = packet["event_process"]["path"].split(".local/", 1)[1]
        process_path = self.work / ".local" / relative
        process_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_work / ".local" / relative, process_path)
        old = json.loads((original_work / ".local/baseline/profiles/graph/flagmega-b001-events-000/measurement.json").read_text())
        self.write(".local/baseline/profiles/graph/old/measurement.json", old)
        actual = next(c for c in packet["controls"] if c.get("cohort") == "event_instrumented" and c.get("prompt_length") == 128)

        def charts():
            with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
                result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/native-history")
            directory = self.work / ".local/reports" / result["snapshot_id"]
            return [(json.loads((directory / (name + ".json")).read_text()), (directory / (name + ".svg")).read_text())
                    for name in ("performance-vs-milestone", "performance-vs-iteration")]

        before = charts()
        manifest = json.loads((self.work / "artifacts/manifest.json").read_text())
        manifest["source_id"] = "b001+trace-v2-current-000"
        manifest["fixture_scope"] = "source/NUM metadata publication only; actual copied timing unchanged"
        self.write("artifacts/manifest.json", manifest)
        real_num = json.loads((original_work / ".local/agents/dispatch/review_trace_v2_admission_NUM/handoff.json").read_text())
        self.write(".local/agents/dispatch/real-NUM/handoff.json", real_num)
        self.dashboard("new native NUM metadata only, no new timing", "[NUM](.local/agents/dispatch/real-NUM/handoff.json)")
        for (_, old_svg), (chart, svg) in zip(before, charts()):
            numeric = [r for r in chart["rows"] if r.get("value") is not None]
            self.assertEqual([r["value"] for r in numeric], [actual["tokens_per_second"]])
            self.assertIn(actual["capture_id"], numeric[0]["measurement_ids"])
            self.assertIn("Historical", numeric[0]["label"])
            self.assertIn("GPU3", numeric[0]["label"])
            self.assertEqual(numeric[0]["source_id"], actual["source_id"])
            self.assertEqual(numeric[0]["runtime_manifest_sha256"], actual["runtime_manifest_sha256"])
            self.assertEqual(numeric[0]["binding"]["uuid"], actual["physical_gpu_uuid"])
            self.assertEqual(svg.count('class="performance-point-marker"'), 1)
            self.assertEqual(svg.count('class="performance-point-table-row"'), 1)
            self.assertEqual(old_svg.count('class="performance-point-marker"'), 1)
        data, _ = renderer.collect(self.work)
        self.assertIsNone(data["events"])
        self.assertEqual(data["native_controls"], [])

    def test_dashboard_JSON_array_is_opaque_evidence_not_device_receipt(self):
        self.write(".local/baseline/results/capability-array.json", [{"actual": "declared nonreceipt CPU evidence"}])
        self.dashboard("actual dashboard accepts JSON array evidence", "[array](.local/baseline/results/capability-array.json)")
        data, inputs = renderer.collect(self.work)
        self.assertEqual(data["selected_device"]["index"], "unselected")
        self.assertIsNone(data["events"])
        self.assertIn(".local/baseline/results/capability-array.json", inputs.pins)

    def copy_native_backend_state_receipts(self):
        original_work = Path(renderer.__file__).resolve().parents[1]
        directory = ".local/agents/dispatch/reference_gpu_num_gpu3"
        handoff = json.loads((original_work / directory / "handoff.json").read_text())
        summary = json.loads((original_work / handoff["results"]["path"]).read_text())
        manifest = json.loads((original_work / handoff["raw_evidence_manifest"]["path"]).read_text())
        files = [directory + "/handoff.json", handoff["results"]["path"], handoff["raw_evidence_manifest"]["path"]]
        files += [item["report"]["path"] for item in summary["successful_backends"]]
        for item in summary["failed_backends"]:
            files += [item["causal_evidence"]["path"]]
            files += [pin["path"] for pin in manifest["files"] if pin["path"].endswith(item["job"] + "/measurement.failed.json")]
        flash = ".local/agents/dispatch/approve_flashrt_v1_graph_profiler/handoff.json"
        for relative in files + [flash]:
            target = self.work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_work / relative, target)
        self.dashboard("Actual copied native runtime receipts; no new GPU timing",
                       f"[native]({directory}/handoff.json) [PLAN]({flash})")
        return summary

    def test_native_backend_receipts_refresh_NUM_HOST_without_DEVICE_bars(self):
        summary = self.copy_native_backend_state_receipts()
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/backend-state")
        directory = self.work / ".local/reports" / result["snapshot_id"]
        chart = json.loads((directory / "backend-device-comparison.json").read_text())
        rows = {r["label"]: r for r in chart["rows"]}
        text = (directory / "readme.md").read_text()
        for backend, label in [("tilefoundry", "TileFoundry"), ("llama.cpp", "llama.cpp")]:
            measured = next(s for s in summary["successful_backends"] if s["backend"] == backend)
            primary = next(s for s in measured["scenarios"] if s["prompt_length"] == 128)
            reason = rows[label]["reason"]
            self.assertIn("960 own outputs exact", reason)
            self.assertIn(f"HOST {primary['HOST_TPS_unreviewed_diagnostic']:.6f}", reason)
            self.assertIn("diagnostic", reason)
            self.assertIn("DEVICE", reason)
            self.assertIn("pending", reason)
            self.assertIn(reason, text)
        self.assertIn("UUID", rows["vLLM"]["reason"])
        self.assertIn("cuda_graph_runner", rows["SGLang"]["reason"])
        self.assertIn("PLAN-only", rows["FlashRT"]["reason"])
        self.assertTrue(all(row["value"] is None for row in chart["rows"]), "HOST diagnostics cannot enter DEVICE bars")
        self.assertNotIn("GPU unmeasured", rows["llama.cpp"]["reason"])
        self.assertNotIn("compiled native pending", rows["TileFoundry"]["reason"])


    def test_available_native_HOST_numbers_have_explicit_HOST_chart_scope(self):
        summary = self.copy_native_backend_state_receipts()
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/HOST-presentation")
        directory = self.work / ".local/reports" / result["snapshot_id"]
        chart = json.loads((directory / "backend-comparison.json").read_text())
        self.assertEqual(chart["measurement_channel"], "HOST")
        rows = {r["label"]:r for r in chart["rows"]}
        for backend,label in [("tilefoundry","TileFoundry"),("llama.cpp","llama.cpp")]:
            expected = next(x for x in summary["successful_backends"] if x["backend"] == backend)
            primary = next(x for x in expected["scenarios"] if x["prompt_length"] == 128)
            self.assertEqual(rows[label]["value"], primary["HOST_TPS_unreviewed_diagnostic"])
            self.assertEqual(rows[label]["measurement_channel"], "HOST")
            self.assertIn("diagnostic", rows[label]["reason"])
            self.assertIn("boundary", rows[label])
        self.assertFalse(chart["formal_comparison_qualified"])
        self.assertTrue(all(row["value"] is None for row in json.loads((directory/"backend-device-comparison.json").read_text())["rows"]))

    def test_deleted_status_never_enters_current_publication_or_readme(self):
        self.dashboard("explicit no-status current publication fixture")
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/no-status")
        directory = self.work/".local/reports"/result["snapshot_id"]
        record = json.loads((directory/"sync-figures.json").read_text())
        self.assertNotIn("figures/status.svg", [x["path"] for x in record["files"]])
        self.assertNotIn("figures/status.svg", (directory/"readme.md").read_text())
        self.assertFalse((directory/"status.svg").exists())

    def test_real_HOST_stage_pairs_preserve_whole_window_and_GPU_unknown(self):
        self.copy_three_probe_receipts()
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/actual-HOST-stages")
        directory = self.work/".local/reports"/result["snapshot_id"]
        graph = json.loads((directory/"graph-trace.json").read_text())
        self.assertEqual(graph["measurement_channel"], "HOST")
        self.assertEqual(graph["window_ns"], [711862347417093,711862475462999])
        self.assertTrue(all(len(row["intervals"]) == 63 for row in graph["rows"][:-1]))
        self.assertTrue(all(span["evidence"] == "inflight_envelope" for row in graph["rows"][:-1] for span in row["intervals"]))
        self.assertEqual(graph["rows"][-1]["status"], "unavailable")
        self.assertEqual(graph["rows"][-1]["intervals"], [])
        self.assertTrue(json.loads((directory/"operator-latency.json").read_text())["pending_only"])
        self.assertIn("SourceCounters not required",json.loads((directory/"operator-latency.json").read_text())["pending_reason"])
        descriptor = json.loads((self.work/"benchmark/current-diagnostic-input.json").read_text())
        with (self.work/descriptor["P1_stages"]["path"]).open("a") as stream:
            stream.write("\n")
        data, inputs = renderer.collect(self.work)
        self.assertIsNone(data["latest_diagnostic"])
        self.assertTrue(inputs.errors)

    def test_explicit_status_deletion_reconciles_only_one_managed_file(self):
        original = Path(renderer.__file__).resolve().parents[1]
        before = original/".local/agents/dispatch/fix_current_figures_user_feedback/failure-before"
        pointer = json.loads((before/".local/views/root-figures.json").read_text())
        for row in pointer["managed_files"]:
            if row["path"] == "figures/status.svg":
                continue
            target = self.work/row["path"]
            target.parent.mkdir(parents=True,exist_ok=True)
            source = before/row["path"] if (before/row["path"]).exists() else original/row["source"]
            shutil.copyfile(source,target)
        self.write(".local/views/root-figures.json",pointer)
        policy = json.loads((original/"benchmark/current-figure-policy.json").read_text())
        self.write("benchmark/current-figure-policy.json",policy)
        target = self.work/policy["root_status_deletion"]["path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original/policy["root_status_deletion"]["path"],target)
        # Unrelated drift remains rejected before any deletion metadata commit.
        other = self.work/"figures/backend-comparison.svg"
        original_bytes = other.read_bytes()
        other.write_bytes(original_bytes + b"\n")
        with self.assertRaisesRegex(ValueError,"Another managed figure"):
            renderer.reconcile_deleted_status(self.work,"/fixture/explicit-deletion")
        other.write_bytes(original_bytes)
        receipt = renderer.reconcile_deleted_status(self.work,"/fixture/explicit-deletion")
        self.assertTrue(receipt)
        after = json.loads((self.work/".local/views/root-figures.json").read_text())
        self.assertNotIn("figures/status.svg",[row["path"] for row in after["managed_files"]])
        self.assertFalse((self.work/"figures/status.svg").exists())
        self.assertIsNone(renderer.reconcile_deleted_status(self.work,"/fixture/explicit-deletion"))
        for row in after["managed_files"]:
            self.assertEqual(renderer.sha(self.work/row["path"]),row["sha256"])

    def test_changed_native_report_is_not_admitted_as_frozen_backend_state(self):
        summary = self.copy_native_backend_state_receipts()
        relative = summary["successful_backends"][0]["report"]["path"]
        with (self.work / relative).open("a") as stream:
            stream.write("\n")
        data, inputs = renderer.collect(self.work)
        self.assertNotIn("tilefoundry", data.get("native_backend_states", {}))
        self.assertTrue(any(error["path"] == relative for error in inputs.errors), "Changed hash-bound report must be visible")

    def copy_three_probe_receipts(self):
        self.test_dashboard_authenticates_current_GPU3_without_management_JSON()
        original_work = Path(renderer.__file__).resolve().parents[1]
        relative = ".local/agents/dispatch/execute_ncu_001_diagnostic_gpu3/handoff.json"
        handoff = json.loads((original_work / relative).read_text())
        summary = json.loads(Path(handoff["summary"]["path"]).read_text())
        pins = [handoff[key] for key in ["manifest", "summary", "actual_error", "P2_helper_inspection", "P2_report"]]
        pins += [row["process_packet"] for row in summary["rows"] if row["probe_id"].startswith(("P1", "P2"))]
        shutil.copyfile(original_work / "benchmark/current-diagnostic-input.json", self.work / "benchmark/current-diagnostic-input.json")
        paths = [original_work / relative, original_work / ".local/agents/dispatch/diagnose_source_counters_illegal_memory/P2-values-and-dimensional-proof-v1.json"]
        paths += [Path(pin["path"]) for pin in pins]
        descriptor = json.loads((original_work/"benchmark/current-diagnostic-input.json").read_text())
        paths += [original_work/descriptor[key]["path"] for key in ["P1_stages","P1_observations_source"]]
        old_packet = json.loads((original_work / ".local/baseline/results/native-diagnostic-data-002.json").read_text())
        paths += [original_work / ".local" / old_packet["event_process"]["path"].split(".local/", 1)[1]]
        for source in paths:
            target = self.work / ".local" / str(source).split(".local/", 1)[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        self.dashboard("Actual copied threeprobe diagnosis; no new measured performance point", f"[threeprobe]({relative})")
        return json.loads(Path(handoff["P2_helper_inspection"]["path"]).read_text())

    def test_three_probe_latest_HBM_scopes_preserve_history(self):
        helper = self.copy_three_probe_receipts()
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/threeprobe")
        directory = self.work / ".local/reports" / result["snapshot_id"]
        chart = json.loads((directory / "hbm-ncu-reported.json").read_text())
        expected = [a["required_metrics"]["dram__throughput.avg.pct_of_peak_sustained_elapsed"]["value"] for a in helper["actions"]]
        self.assertEqual([row["value"] for row in chart["rows"]], expected)
        self.assertTrue(all("P2" in row["label"] for row in chart["rows"]))
        text = (directory / "readme.md").read_text()
        self.assertIn("45.08299559", text)
        self.assertIn("144.77872", text)
        self.assertIn("SourceCounters", text)
        self.assertIn("supervisor=false / raw=true", text)
        self.assertIn("not ordinary utilization", text)
        for name in ["performance-vs-milestone", "performance-vs-iteration"]:
            svg = (directory / (name + ".svg")).read_text()
            self.assertEqual(svg.count('class="performance-point-marker"'), 1)
            self.assertEqual(svg.count('class="performance-point-table-row"'), 1)
            points = json.loads((directory / (name + ".json")).read_text())["rows"]
            self.assertEqual([p["value"] for p in points], [495.1419730500993])

    def test_three_probe_changed_helper_is_visible_and_not_admitted(self):
        self.copy_three_probe_receipts()
        path = self.work / ".local/agents/dispatch/execute_ncu_001_diagnostic_gpu3/analysis/P2-helper-inspection.json"
        fingerprint_before = renderer.fingerprint(self.work, renderer.find_skill())
        with path.open("a") as stream:
            stream.write("\n")
        self.assertNotEqual(renderer.fingerprint(self.work, renderer.find_skill()), fingerprint_before)
        data, inputs = renderer.collect(self.work)
        self.assertIsNone(data.get("latest_diagnostic"))
        self.assertTrue(any(error["path"].endswith("P2-helper-inspection.json") for error in inputs.errors))

    def test_three_probe_different_runtime_bytes_stay_historical(self):
        self.copy_three_probe_receipts()
        path = self.work / ".local/baseline/build/b001/runtime-artifact/artifact.json"
        with path.open("a") as stream:
            stream.write("\n")
        data, _ = renderer.collect(self.work)
        self.assertFalse(data["latest_diagnostic"]["binding"]["matching_selected_runtime"])
        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/historical-HOST")
        directory = self.work/".local/reports"/result["snapshot_id"]
        self.assertIn("Historical",json.loads((directory/"graph-trace.json").read_text())["title"])
        self.assertIn("Historical",json.loads((directory/"backend-comparison.json").read_text())["rows"][0]["label"])

        with patch.object(renderer.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="fixture-sync-only", stderr="")):
            result = renderer.render_live(self.work, renderer.find_skill(), agent_id="/fixture/threeprobe-historical")
        chart = json.loads((self.work / ".local/reports" / result["snapshot_id"] / "hbm-ncu-reported.json").read_text())
        self.assertTrue(all(row["label"].startswith("historical P2") for row in chart["rows"]))

    def test_detached_watcher_updates_real_dashboard_and_stops_owned_PID(self):
        actor = "/fixture/managed-live-watcher"
        self.dashboard("fixture-live-before")
        observed = renderer.start_watcher(self.work, renderer.find_skill(), actor, .25)
        self.assertTrue(observed["alive"])
        self.assertEqual(observed["identity"]["session"], observed["identity"]["pid"])
        try:
            pointer = self.work / ".local/views/root-figures.json"
            before = json.loads(pointer.read_text())
            self.dashboard("fixture-live-after-real-dashboard-edit")
            for _ in range(100):
                after = json.loads(pointer.read_text())
                if after["generation"] > before["generation"]:
                    break
                time.sleep(.1)
            self.assertGreater(after["generation"], before["generation"])
            self.assertEqual(after["status"]["flow"]["rows"][0]["description"], "fixture-live-after-real-dashboard-edit")
        finally:
            stopped = renderer.stop_watcher(self.work, actor)
            self.assertFalse(stopped["alive"])


if __name__ == "__main__":
    unittest.main()
