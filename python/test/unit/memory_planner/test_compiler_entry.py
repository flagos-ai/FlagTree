"""Cover the public entry without requiring a vendor backend.

The first two cases use a controlled stand-in for the Torch execution boundary.
The real FX case exports and reloads a CPU graph and calls torch.compile using
its eager backend, which avoids requiring Triton kernels or a device.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from test_compiler_aliases import fixture_facts, view_manifest


def modules():
    # The existing FlagTree package imports Triton's version at module import.
    # CPU-only tests don't need anything else from Triton.
    try:
        import flagtree  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "triton":
            raise
        triton = ModuleType("triton")
        triton.__version__ = "cpu-test"
        sys.modules["triton"] = triton
        try:
            import flagtree  # noqa: F401
        finally:
            del sys.modules["triton"]
    from flagtree.memory_planner import compiler_aliases
    from model_import_adapters import compiled_execution, unified_ir
    return compiled_execution, unified_ir, compiler_aliases


class TestPublicEntry(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.entry, self.ir, self.aliases = modules()

    def _run_standin(self, enabled):
        class Graph:
            def __init__(self):
                self.retargeted = False

            def named_modules(self):
                return [("", self)]

            def to(self, _):
                return self

            def __call__(self):
                return 1

        source, device_graph = Graph(), Graph()
        torch = ModuleType("torch")
        torch.fx = ModuleType("torch.fx")
        torch.fx.GraphModule = Graph
        torch.no_grad = nullcontext
        torch.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        torch.utils = SimpleNamespace(_pytree=SimpleNamespace(tree_flatten=lambda item: ([item], None)))
        torch.Tensor = np.ndarray

        def compile_graph(graph, *, fullgraph):
            self.assertTrue(fullgraph)
            self.assertIs(graph, device_graph)
            self.assertTrue(graph.retargeted)
            return graph

        def summarize(name, graph):
            self.assertIs(graph, device_graph)
            self.assertTrue(graph.retargeted)
            return view_manifest()["graphs"][0]

        torch.compile = mock.Mock(side_effect=compile_graph)
        graph_file = self.root / "fixture.pt2"
        graph_file.write_bytes(b"test-only serialization stand-in")
        with (
            mock.patch.dict(sys.modules, {"torch": torch, "torch.fx": torch.fx}),
            mock.patch.object(self.entry, "load_core_aten",
                              side_effect=[SimpleNamespace(module=lambda: source),
                                           SimpleNamespace(module=lambda: device_graph)]),
            mock.patch.object(self.entry, "load_example_inputs", return_value=((), {})),
            mock.patch.object(self.entry, "_retarget_factory_devices",
                              side_effect=lambda graph, _: setattr(graph, "retargeted", True) or 1),
            mock.patch.object(self.entry, "compare_outputs", return_value=[]),
            mock.patch.object(self.ir, "_graph_module_summary", side_effect=summarize) as summary,
            mock.patch.object(self.aliases, "collect_operator_schemas", side_effect=fixture_facts) as schemas,
        ):
            flags = {"memory_analysis": True} if enabled else {}
            result = self.entry.run_exported_program_on_device(
                graph_file, self.root / "inputs.npz", {}, self.root / "out", device="cpu", **flags)
        torch.compile.assert_called_once_with(device_graph, fullgraph=True)
        self.assertEqual(summary.call_count, int(enabled))
        self.assertEqual(schemas.call_count, int(enabled))
        return result

    def test_default_off_keeps_original_entry_and_reports(self):
        result = self._run_standin(False)
        self.assertEqual(result["status"], "passed")
        self.assertNotIn("memory_analysis", result)
        self.assertFalse((self.root / "out/memory-analysis.json").exists())
        self.assertFalse((self.root / "out/memory-manifest.json").exists())

    def test_opt_in_analyzes_device_graph_and_keeps_planning_blocked(self):
        result = self._run_standin(True)
        self.assertEqual(result["status"], "passed")
        report = json.loads(Path(result["memory_analysis"]["report_path"]).read_text())
        self.assertFalse(result["memory_analysis"]["allocation_or_runtime_changed"])
        self.assertFalse(report["runtime_applied"])
        self.assertFalse(report["optimization_applied"])
        self.assertEqual(report["reuse_planning"]["status"], "blocked")
        self.assertEqual(report["inplace_legality"]["status"], "not_run_missing_alias_and_effects")
        self.assertIsNone(report["storage_analysis"]["peak_bytes"])
        self.assertTrue(report["alias_analysis"]["graphs"][0]["must_alias_edges"])

    def test_real_cpu_fx_and_compile_see_the_same_graph(self):
        try:
            import torch
        except ImportError:
            if os.environ.get("F08_REQUIRE_TORCH") == "1":
                raise
            self.skipTest("PyTorch is unavailable")

        class Model(torch.nn.Module):
            def forward(self, x):
                base = torch.neg(x)
                view = base.view(2, 3)
                permute = view.permute(1, 0)
                return permute.clone()

        x = torch.arange(6, dtype=torch.float32)
        program = torch.export.export(Model().eval(), (x,), strict=True)
        graph_file, input_file = self.root / "model.pt2", self.root / "input.npz"
        torch.export.save(program, graph_file)
        np.savez(input_file, arg_0=x.numpy())
        input_schema = {"positional": ["arg_0"], "keyword": []}
        real_compile = torch.compile
        summarized = []
        compiled = []
        real_summary = self.ir._graph_module_summary

        def summarize(name, graph):
            summarized.append(graph)
            return real_summary(name, graph)

        def compile_graph(graph, *, fullgraph):
            compiled.append(graph)
            return real_compile(graph, backend="eager", fullgraph=fullgraph)

        with (mock.patch.object(self.ir, "_graph_module_summary", side_effect=summarize),
              mock.patch.object(torch, "compile", side_effect=compile_graph)):
            result = self.entry.run_exported_program_on_device(
                graph_file, input_file, input_schema, self.root / "out",
                device="cpu", memory_analysis=True)
        self.assertEqual(result["status"], "passed")
        self.assertIs(summarized[0], compiled[0])
        report = json.loads(Path(result["memory_analysis"]["report_path"]).read_text())
        graph = report["ir_analysis"]["graphs"][0]
        alias_graph = report["alias_analysis"]["graphs"][0]
        base = next(value for value in graph["values"] if value["node"] == "neg")
        group = next(group for group in alias_graph["alias_groups"]
                     if base["source_tensor_id"] in group["members"])
        self.assertGreater(group["lifetime_end"], base["last_reference_index"])
        self.assertFalse(group["safe_to_release"])
        self.assertIn("neg", alias_graph["nodes_with_unproven_effects"])
        self.assertFalse(report["runtime_applied"])
        self.assertFalse(report["optimization_applied"])
