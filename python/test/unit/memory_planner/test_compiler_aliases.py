"""Compiler alias lifetime regressions; fixture declarations are synthetic.

The separate real-schema class reads installed Torch on CPU. Missing Torch is
a hard failure when F08_REQUIRE_TORCH=1; local fixture tests are not that proof.
"""
import copy
import os
import unittest
from unittest import mock

from test_compiler_analysis import analysis, adapter, manifest_for, node, ref, tensor_metadata, tuple_manifest
import importlib

aliases = importlib.import_module(analysis.__package__ + ".compiler_aliases")
intervals = importlib.import_module(analysis.__package__ + ".lifetime_intervals")


def seal(record):
    record["record_sha256"] = aliases._digest({k: v for k, v in record.items() if k != "record_sha256"})


def fixture_facts(manifest):
    records = {}
    for target, (name, overload) in aliases.VIEW_RULES.items():
        alias = {"before": ["a"], "after": ["a"], "is_write": False}
        records[target] = {
            "status": "available", "name": name, "overload": overload,
            "text": "synthetic unit-test declaration", "is_mutable": False,
            "arguments": [{"name": "self", "type": "Tensor", "alias": copy.deepcopy(alias)}],
            "returns": [{"name": "", "type": "Tensor", "alias": copy.deepcopy(alias)}],
        }
        seal(records[target])
    return {"schema_version": aliases.FACT_VERSION,
            "source": "installed_pytorch_dispatcher_schema", "torch_version": "test-fixture",
            "manifest_sha256": adapter.manifest_identity(manifest)[1], "targets": records}


def view_manifest(target="aten.view.default", *, output_view=False):
    return manifest_for([
        node("x", "placeholder", "x", value=tensor_metadata()),
        node("view", "call_function", target, [ref("x"), [4]], tensor_metadata()),
        node("early", "call_function", "aten.neg.default", [ref("x")], tensor_metadata()),
        node("late", "call_function", "aten.neg.default", [ref("view")], tensor_metadata()),
        node("output", "output", "output", [ref("view" if output_view else "late")]),
    ])


def group_for(report, leaf):
    return next(g for g in report["graphs"][0]["alias_groups"] if leaf in g["members"])


class CompilerAliasTests(unittest.TestCase):
    def run_analysis(self, manifest):
        return aliases.analyze_compiler_aliases(manifest, fixture_facts(manifest))

    def test_view_late_consumer_extends_whole_group(self):
        report = self.run_analysis(view_manifest())
        group = group_for(report, "root/x:value")
        self.assertEqual(group["members"], ["root/x:value", "root/view:value"])
        self.assertEqual((group["lifetime_start"], group["lifetime_end"]), (0, 3))
        self.assertIn("graph_input_or_state", group["protected_reasons"])
        self.assertFalse(group["safe_to_release"])

    def test_all_four_view_rules_and_symbolic_layout(self):
        for target in aliases.VIEW_RULES:
            with self.subTest(target=target):
                manifest = view_manifest(target)
                manifest["graphs"][0]["nodes"][1]["metadata"]["value"]["layout"] = "symbolic_strided"
                report = self.run_analysis(manifest)
                self.assertEqual(len(report["graphs"][0]["must_alias_edges"]), 1)

    def test_output_view_protects_entire_group(self):
        group = group_for(self.run_analysis(view_manifest(output_view=True)), "root/x:value")
        self.assertTrue(group["is_graph_output"])
        self.assertIn("graph_output", group["protected_reasons"])
        self.assertEqual(group["lifetime_end"], 4)

    def test_slice_expand_permute_chain_keeps_root_until_last_consumer(self):
        manifest = manifest_for([
            node("x", "placeholder", "x", value=tensor_metadata()),
            node("slice", "call_function", "aten.slice.Tensor", [ref("x"), 0, 1, 3], tensor_metadata()),
            node("expand", "call_function", "aten.expand.default", [ref("slice"), [4]], tensor_metadata()),
            node("permute", "call_function", "aten.permute.default", [ref("expand"), [0]], tensor_metadata()),
            node("late", "call_function", "aten.neg.default", [ref("permute")], tensor_metadata()),
            node("output", "output", "output", [ref("late")]),
        ])
        group = group_for(self.run_analysis(manifest), "root/x:value")
        self.assertEqual(len(group["members"]), 4)
        self.assertEqual(group["lifetime_end"], 4)
        self.assertIsNone(group["capacity_bytes"])  # view shape is not storage capacity
        self.assertFalse(group["safe_to_release"])

    def test_self_keyword_is_supported_without_changing_call(self):
        manifest = view_manifest()
        view = manifest["graphs"][0]["nodes"][1]
        view["args"] = []
        view["kwargs"] = {"self": ref("x"), "size": [4]}
        # users stay identical because the same node reference is in kwargs.
        self.assertEqual(len(self.run_analysis(manifest)["graphs"][0]["must_alias_edges"]), 1)

    def test_clone_same_shape_does_not_imply_alias_or_disjointness(self):
        report = self.run_analysis(view_manifest("aten.clone.default"))
        self.assertEqual(group_for(report, "root/x:value")["members"], ["root/x:value"])
        self.assertTrue(report["groups_may_alias_each_other"])

    def test_missing_schema_does_not_merge(self):
        manifest = view_manifest()
        facts = fixture_facts(manifest)
        facts["targets"].clear()
        report = aliases.analyze_compiler_aliases(manifest, facts)
        self.assertEqual(len(report["graphs"][0]["must_alias_edges"]), 0)
        self.assertEqual(report["graphs"][0]["diagnostics"][0]["code"], "view_schema_missing_or_unproven")

    def test_changed_manifest_rejects_old_facts(self):
        manifest = view_manifest()
        facts = fixture_facts(manifest)
        manifest["source"]["changed"] = True
        with self.assertRaisesRegex(adapter.CompilerIRAnalysisError, "operator_schema_manifest_mismatch"):
            aliases.analyze_compiler_aliases(manifest, facts)

    def test_changed_schema_digest_rejected(self):
        manifest = view_manifest()
        facts = fixture_facts(manifest)
        facts["targets"]["aten.view.default"]["overload"] = "dtype"
        with self.assertRaisesRegex(adapter.CompilerIRAnalysisError, "operator_schema_record_mismatch"):
            aliases.analyze_compiler_aliases(manifest, facts)

    def test_wrong_schema_name_overload_write_and_wildcard_refuse_merge(self):
        for change in ("name", "overload", "mutable", "wildcard", "alias_mismatch"):
            with self.subTest(change=change):
                manifest = view_manifest()
                facts = fixture_facts(manifest)
                record = facts["targets"]["aten.view.default"]
                if change in {"name", "overload"}:
                    record[change] = "unknown"
                elif change == "mutable":
                    record["is_mutable"] = True
                elif change == "wildcard":
                    for item in record["arguments"] + record["returns"]:
                        item["alias"]["before"] = item["alias"]["after"] = ["*"]
                else:
                    record["returns"][0]["alias"]["after"] = ["b"]
                seal(record)
                report = aliases.analyze_compiler_aliases(manifest, facts)
                self.assertEqual(report["graphs"][0]["must_alias_edges"], [])
                if change == "mutable":
                    self.assertEqual(report["graphs"][0]["schema_observed_mutation_nodes"], ["view"])

    def test_device_dtype_or_sparse_metadata_does_not_merge(self):
        for key, value in (("dtype", "int64"), ("device", "cuda:0"), ("layout", "sparse")):
            manifest = view_manifest()
            manifest["graphs"][0]["nodes"][1]["metadata"]["value"][key] = value
            report = self.run_analysis(manifest)
            self.assertEqual(report["graphs"][0]["must_alias_edges"], [])

    def test_tuple_leaves_distinct_projection_hold_preserved(self):
        manifest = tuple_manifest()
        report = self.run_analysis(manifest)
        values = group_for(report, "root/max:[0]")
        indices = group_for(report, "root/max:[1]")
        self.assertIn("root/values:value", values["members"])
        self.assertIn("root/indices:value", indices["members"])
        self.assertNotEqual(values["alias_group_id"], indices["alias_group_id"])
        self.assertGreaterEqual(values["lifetime_end"], 4)  # tuple last projection

    def test_input_is_not_modified_and_report_deterministic(self):
        manifest = view_manifest()
        manifest["source"]["mask_literal"] = float("-inf")
        before = copy.deepcopy(manifest)
        result = self.run_analysis(manifest)
        self.assertEqual(result, self.run_analysis(manifest))
        self.assertEqual(result["manifest_sha256"], adapter.manifest_identity(manifest)[1])
        self.assertEqual(manifest, before)

    def test_unknown_effects_not_discharged_by_schema(self):
        manifest = view_manifest()
        report = self.run_analysis(manifest)
        self.assertIn("late", report["graphs"][0]["nodes_with_unproven_effects"])
        self.assertIsNone(report["peak_bytes"])
        self.assertFalse(report["runtime_applied"])

    def test_enabled_entry_includes_alias_report_but_keeps_storage_blocked(self):
        manifest = view_manifest()
        report = analysis.analyze_compiler_manifest(manifest, enabled=True, operator_schemas=fixture_facts(manifest))
        self.assertIn("alias_analysis", report)
        self.assertIsNone(report["storage_analysis"]["peak_bytes"])
        self.assertEqual(report["reuse_planning"]["status"], "blocked")
        self.assertTrue(all(v["storage_id"] is None and not v["safe_to_release"]
                            for v in report["ir_analysis"]["graphs"][0]["values"]))

    def test_disabled_path_does_not_read_facts(self):
        self.assertIsNone(analysis.analyze_compiler_manifest(None, enabled=False, operator_schemas=object()))

    def test_invalid_schema_structure_is_diagnostic(self):
        manifest = view_manifest()
        facts = fixture_facts(manifest)
        record = facts["targets"]["aten.view.default"]
        record["arguments"] = ["not-an-argument"]
        seal(record)
        with self.assertRaisesRegex(adapter.CompilerIRAnalysisError, "invalid_operator_schema_record"):
            aliases.analyze_compiler_aliases(manifest, facts)

    def test_interval_union_keeps_closed_endpoints(self):
        self.assertEqual(intervals.union_lifetime_intervals([(2, 4), (4, 9), (1, 1)]), (1, 9))
        for bad in ([], [(4, 2)], [(True, 2)], [(-1, 3)]):
            with self.assertRaises(ValueError):
                intervals.union_lifetime_intervals(bad)


class InstalledTorchSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError as error:
            if os.environ.get("F08_REQUIRE_TORCH") == "1":
                raise RuntimeError("installed Torch required; do not report synthetic schemas as installed evidence") from error
            raise unittest.SkipTest("Torch unavailable; installed schemas not checked")
        cls.torch = torch

    def test_installed_four_views_read_schemas_without_device_or_operator_execution(self):
        for target in aliases.VIEW_RULES:
            manifest = view_manifest(target)
            with mock.patch.object(self.torch.cuda, "_lazy_init", side_effect=AssertionError("no device")):
                facts = aliases.collect_operator_schemas(manifest)
                report = aliases.analyze_compiler_aliases(manifest, facts)
            self.assertEqual(len(report["graphs"][0]["must_alias_edges"]), 1, facts["targets"].get(target))

    def test_installed_schema_replay_is_identical(self):
        manifest = view_manifest()
        facts = aliases.collect_operator_schemas(manifest)
        self.assertEqual(aliases.analyze_compiler_aliases(manifest, facts),
                         aliases.analyze_compiler_aliases(manifest, json.loads(json.dumps(facts))))


if __name__ == "__main__":
    unittest.main()
