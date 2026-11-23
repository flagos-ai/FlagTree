"""Compiler-IR use/definition tests; no model, torch, runtime, or device.

The fixtures describe IR semantics only.  None of these tests prove physical
storage identity, a device completion order, or authorization to reuse memory.
"""

from __future__ import annotations

import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "python" / "flagtree" / "memory_planner"
PACKAGE_NAME = "flagtree_compiler_analysis_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules[PACKAGE_NAME] = package
for name in ("model_ir_adapter", "compiler_analysis"):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{name}", PACKAGE_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

adapter = sys.modules[f"{PACKAGE_NAME}.model_ir_adapter"]
analysis = sys.modules[f"{PACKAGE_NAME}.compiler_analysis"]


def tensor_metadata(dtype="float32", shape=(4,), *, requires_grad=False):
    return {
        "kind": "tensor",
        "dtype": dtype,
        "shape": [{"kind": "static", "value": size} for size in shape],
        "stride": ["1"] * len(shape),
        "layout": "contiguous",
        "device": "cpu",
        "requires_grad": requires_grad,
    }


def ref(name):
    return {"node": name}


def node(name, op, target, args=(), value=None):
    return {
        "name": name,
        "op": op,
        "target": target,
        "args": list(args),
        "kwargs": {},
        "users": [],
        "metadata": {"value": copy.deepcopy(value)},
    }


def references(value):
    if isinstance(value, dict):
        if set(value) == {"node"}:
            return [value["node"]]
        return [item for child in value.values() for item in references(child)]
    if isinstance(value, (tuple, list)):
        return [item for child in value for item in references(child)]
    return []


def manifest_for(nodes, *, control_flow=()):
    nodes = copy.deepcopy(nodes)
    for item in nodes:
        item["users"] = []
    by_name = {item["name"]: item for item in nodes}
    for item in nodes:
        for name in set(references([item["args"], item["kwargs"]])):
            if name in by_name:
                by_name[name]["users"].append(item["name"])
    for item in nodes:
        item["users"].sort()
    return {
        "schema_version": 1,
        "format": "PyTorch ExportedProgram",
        "dialect": "Core ATen with structured higher-order operators when present",
        "graphs": [{
            "name": "root", "node_count": len(nodes), "nodes": nodes,
            "control_flow": list(control_flow),
        }],
        "control_flow": list(control_flow),
        "source": {"kind": "compiler_ir_test"},
    }


def linear_manifest():
    return manifest_for([
        node("x", "placeholder", "x", value=tensor_metadata()),
        node("silu", "call_function", "aten.silu.default", [ref("x")], tensor_metadata()),
        node("mul", "call_function", "aten.mul.Tensor", [ref("silu"), ref("silu")], tensor_metadata()),
        node("output", "output", "output", [ref("mul")]),
    ])


def tuple_manifest():
    return manifest_for([
        node("x", "placeholder", "x", value=tensor_metadata(shape=(2, 4))),
        node("max", "call_function", "aten.max.dim", [ref("x"), 1],
             [tensor_metadata(shape=(2,)), tensor_metadata("int64", (2,))]),
        node("values", "call_function", "builtins.getitem", [ref("max"), 0], tensor_metadata(shape=(2,))),
        node("use_values", "call_function", "aten.neg.default", [ref("values")], tensor_metadata(shape=(2,))),
        node("indices", "call_function", "builtins.getitem", [ref("max"), 1], tensor_metadata("int64", (2,))),
        node("use_indices", "call_function", "aten.add.Tensor", [ref("indices"), ref("indices")], tensor_metadata("int64", (2,))),
        node("output", "output", "output", [[ref("use_values"), ref("use_indices")]]),
    ])


def graph_values(report):
    return {item["source_tensor_id"]: item for item in report["graphs"][0]["values"]}


class CompilerIRAnalysisTest(unittest.TestCase):
    def test_negative_infinity_mask_literal_is_metadata_not_a_model_change(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["nodes"][1]["kwargs"]["fill_value"] = float("-inf")
        before = copy.deepcopy(manifest)
        report = analysis.analyze_compiler_manifest(manifest, enabled=True)
        self.assertEqual(manifest, before)
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["runtime_applied"])
        tagged = copy.deepcopy(manifest)
        tagged["graphs"][0]["nodes"][1]["kwargs"]["fill_value"] = {"kind": "f08_float_literal", "value": "-inf"}
        self.assertNotEqual(report["manifest_sha256"], adapter.adapt_model_ir(tagged)["manifest_sha256"])

    def assert_code(self, code, manifest):
        with self.assertRaises(adapter.CompilerIRAnalysisError) as cm:
            adapter.adapt_model_ir(manifest)
        self.assertEqual(cm.exception.code, code)

    def test_default_off_does_not_even_inspect_the_manifest(self):
        # A disabled compiler feature must not change legacy compilation.
        self.assertIsNone(analysis.analyze_compiler_manifest(object()))

    def test_use_def_and_returned_value_are_traceable_without_storage_claims(self):
        report = adapter.adapt_model_ir(linear_manifest())
        values = graph_values(report)
        self.assertEqual(set(values), {"root/x:value", "root/silu:value", "root/mul:value"})
        self.assertEqual(values["root/silu:value"]["consumer_nodes"], ["mul"])
        self.assertGreater(values["root/silu:value"]["last_reference_index"],
                           values["root/silu:value"]["definition_index"])
        self.assertTrue(values["root/mul:value"]["is_graph_output"])
        self.assertTrue(values["root/x:value"]["protected_reasons"])
        self.assertTrue(all(item["safe_to_release"] is False for item in values.values()))
        self.assertEqual(report["analysis_stage"], "model_ir")
        self.assertEqual(report["order_status"], "logical_ir_sequence_not_observed_kernel_schedule")
        self.assertIsNone(report["storage_peak_bytes"])

    def test_analysis_returns_plain_metadata_without_mutating_its_input(self):
        original = linear_manifest()
        before = copy.deepcopy(original)
        result = analysis.analyze_compiler_manifest(original, enabled=True)
        self.assertEqual(original, before)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))
        self.assertFalse(result["runtime_applied"])
        self.assertFalse(result["optimization_applied"])

    def test_same_shape_does_not_invent_storage_identity_or_peak(self):
        result = analysis.analyze_compiler_manifest(linear_manifest(), enabled=True)
        self.assertEqual(result["storage_analysis"]["status"], "blocked_missing_compiler_storage_facts")
        self.assertIsNone(result["storage_analysis"]["peak_bytes"])
        self.assertEqual(result["reuse_planning"]["status"], "blocked")
        self.assertTrue(result["reuse_planning"]["blockers"])
        self.assertEqual(result["inplace_legality"]["status"], "not_run_missing_alias_and_effects")

    def test_stage1_rejects_even_a_claimed_matching_storage_fact(self):
        manifest = linear_manifest()
        fingerprint = adapter.adapt_model_ir(manifest)["manifest_sha256"]
        with self.assertRaises(adapter.CompilerIRAnalysisError) as cm:
            analysis.analyze_compiler_manifest(manifest,
                storage_facts={"manifest_sha256": fingerprint, "storages": []}, enabled=True)
        self.assertEqual(cm.exception.code, "storage_facts_not_supported_in_stage1")

    def test_max_two_leaves_keep_the_tuple_until_last_projection(self):
        result = adapter.adapt_model_ir(tuple_manifest())
        values = graph_values(result)
        self.assertEqual(values["root/max:[0]"]["metadata"]["dtype"], "float32")
        self.assertEqual(values["root/max:[1]"]["metadata"]["dtype"], "int64")
        self.assertIn("values", values["root/max:[0]"]["consumer_nodes"])
        self.assertNotIn("indices", values["root/max:[0]"]["consumer_nodes"])
        self.assertIn("indices", values["root/max:[1]"]["consumer_nodes"])
        later_projection = values["root/indices:value"]["definition_index"]
        self.assertGreaterEqual(values["root/max:[0]"]["retained_through_index"], later_projection)
        self.assertGreaterEqual(values["root/max:[1]"]["retained_through_index"], later_projection)
        aliases = {(item["source"], item["target"]) for item in result["graphs"][0]["projection_aliases"]}
        self.assertIn(("root/max:[0]", "root/values:value"), aliases)
        self.assertIn(("root/max:[1]", "root/indices:value"), aliases)

    def test_projection_alias_carries_graph_output_protection_to_source_leaf(self):
        original = tuple_manifest()
        nodes = original["graphs"][0]["nodes"]
        nodes[-1]["args"] = [ref("values")]
        result = adapter.adapt_model_ir(manifest_for(nodes))
        values = graph_values(result)
        self.assertTrue(values["root/values:value"]["is_graph_output"])
        # The tuple leaf is not itself a direct output node reference, but its
        # projected alias escapes through the graph output and must be protected.
        self.assertIn("graph_output", values["root/max:[0]"]["protected_reasons"])

    def test_dynamic_getitem_consumes_all_tuple_leaves_without_alias_proof(self):
        result = adapter.adapt_model_ir(manifest_for([
            node("x", "placeholder", "x", value=tensor_metadata()),
            node("index", "placeholder", "index", value=tensor_metadata("int64", ())),
            node("pair", "call_function", "example.pair", [ref("x")], [tensor_metadata(), tensor_metadata()]),
            node("item", "call_function", "builtins.getitem", [ref("pair"), ref("index")], tensor_metadata()),
            node("output", "output", "output", [ref("item")]),
        ]))
        values = graph_values(result)
        self.assertIn("item", values["root/pair:[0]"]["consumer_nodes"])
        self.assertIn("item", values["root/pair:[1]"]["consumer_nodes"])
        self.assertEqual(result["graphs"][0]["projection_aliases"], [])

    def test_unknown_tuple_consumer_keeps_all_leaves(self):
        result = adapter.adapt_model_ir(manifest_for([
            node("x", "placeholder", "x", value=tensor_metadata()),
            node("pair", "call_function", "example.pair", [ref("x")], [tensor_metadata(), tensor_metadata("int64")]),
            node("read_pair", "call_function", "example.consume_pair", [ref("pair")], tensor_metadata()),
            node("output", "output", "output", [ref("read_pair")]),
        ]))
        values = graph_values(result)
        for leaf in ("[0]", "[1]"):
            self.assertIn("read_pair", values[f"root/pair:{leaf}"]["consumer_nodes"])

    def test_nested_projection_keeps_full_source_path_and_output_protection(self):
        # A real container key named "value" must not be confused with the
        # root-Tensor fallback leaf path also spelled "value".
        leaf = tensor_metadata()
        result = adapter.adapt_model_ir(manifest_for([
            node("x", "placeholder", "x", value=leaf),
            node("pack", "call_function", "example.pack", [ref("x")],
                 [{"value": copy.deepcopy(leaf)}, tensor_metadata("int64")]),
            node("nested", "call_function", "builtins.getitem", [ref("pack"), 0],
                 {"value": copy.deepcopy(leaf)}),
            node("item", "call_function", "builtins.getitem", [ref("nested"), "value"], leaf),
            node("output", "output", "output", [ref("item")]),
        ]))
        values = graph_values(result)
        aliases = {(item["source"], item["target"]) for item in result["graphs"][0]["projection_aliases"]}
        self.assertIn(("root/pack:[0].value", "root/nested:value"), aliases)
        self.assertIn(("root/nested:value", "root/item:value"), aliases)
        self.assertIn("graph_output", values["root/pack:[0].value"]["protected_reasons"])

    def test_projection_layout_mismatch_does_not_claim_an_alias(self):
        manifest = tuple_manifest()
        # The compiler cannot prove a projection is the same IR tensor when
        # the destination metadata contradicts the selected tuple leaf.
        manifest["graphs"][0]["nodes"][2]["metadata"]["value"]["stride"] = ["2"]
        result = adapter.adapt_model_ir(manifest)
        aliases = {(item["source"], item["target"]) for item in result["graphs"][0]["projection_aliases"]}
        self.assertNotIn(("root/max:[0]", "root/values:value"), aliases)
        self.assertTrue(result["diagnostics"] or result["graphs"][0]["diagnostics"])
        values = graph_values(result)
        self.assertIn("values", values["root/max:[1]"]["consumer_nodes"])

    def test_missing_metadata_is_diagnosed_instead_of_inventing_empty_output(self):
        manifest = linear_manifest()
        del manifest["graphs"][0]["nodes"][1]["metadata"]
        result = adapter.adapt_model_ir(manifest)
        self.assertTrue(result["diagnostics"] or result["graphs"][0]["diagnostics"])
        self.assertIsNone(result["storage_peak_bytes"])

    def test_control_flow_is_not_reported_as_proven_global_storage_liveness(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["control_flow"] = [{"node": "silu", "target": "higher_order.cond"}]
        manifest["control_flow"] = list(manifest["graphs"][0]["control_flow"])
        result = adapter.adapt_model_ir(manifest)
        self.assertTrue(result["diagnostics"] or result["graphs"][0]["diagnostics"])
        self.assertIsNone(result["storage_peak_bytes"])
        self.assertTrue(all(not item["safe_to_release"] for item in graph_values(result).values()))

    def test_duplicate_graph_is_rejected(self):
        manifest = linear_manifest()
        manifest["graphs"].append(copy.deepcopy(manifest["graphs"][0]))
        self.assert_code("duplicate_graph", manifest)

    def test_duplicate_node_is_rejected(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["nodes"].insert(1, copy.deepcopy(manifest["graphs"][0]["nodes"][0]))
        manifest["graphs"][0]["node_count"] = len(manifest["graphs"][0]["nodes"])
        self.assert_code("duplicate_node", manifest)

    def test_unknown_reference_is_rejected(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["nodes"][1]["args"] = [ref("absent")]
        self.assert_code("unknown_reference", manifest)

    def test_forward_reference_and_cycle_are_rejected(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["nodes"][1]["args"] = [ref("mul")]
        self.assert_code("non_topological_reference", manifest)

    def test_forged_users_are_rejected(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["nodes"][1]["users"] = []
        self.assert_code("users_mismatch", manifest)

    def test_changed_graph_changes_independently_computed_fingerprint(self):
        old = linear_manifest()
        changed = copy.deepcopy(old)
        changed["graphs"][0]["nodes"][1]["target"] = "aten.neg.default"
        self.assertNotEqual(adapter.adapt_model_ir(old)["manifest_sha256"],
                            adapter.adapt_model_ir(changed)["manifest_sha256"])

    def test_analysis_is_deterministic_and_does_not_import_device_libraries(self):
        real_import = builtins.__import__

        def no_device_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "triton"}:
                raise AssertionError("compiler manifest analysis tried a device-library import")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=no_device_import):
            first = analysis.analyze_compiler_manifest(linear_manifest(), enabled=True)
            second = analysis.analyze_compiler_manifest(linear_manifest(), enabled=True)
        self.assertEqual(first, second)

    def test_non_json_live_object_is_rejected(self):
        manifest = linear_manifest()
        manifest["graphs"][0]["nodes"][1]["metadata"]["value"] = object()
        with self.assertRaises(adapter.CompilerIRAnalysisError):
            adapter.adapt_model_ir(manifest)


if __name__ == "__main__":
    unittest.main()
