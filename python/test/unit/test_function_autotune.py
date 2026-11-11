"""Function autotuning unit tests and an opt-in single-device smoke test.

Run with ``python -m unittest discover -s python/test/unit -p test_function_autotune.py``.
By default, device interfaces are mocked; no PyTorch or Triton build is required.
Set ``_FLAGTREE_TEST_DEVICE_AUTOTUNE=1`` to also run the real-device test, which
requires installed FlagTree/PyTorch and a working accelerator benchmark backend.
"""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


# Load tune.py in a temporary package to bypass flagtree initialization while preserving relative imports.
def _load_tune():
    source = Path(__file__).resolve().parents[2] / "flagtree" / "tune.py"
    package = ModuleType("_function_autotune_under_test")
    package.__path__ = [str(source.parent)]
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.tune", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package.__name__: package}):
        spec.loader.exec_module(module)
    return module


class LocalWorldSizeTest(unittest.TestCase):

    def test_local_world_size_reads_the_launcher_setting(self):
        tune = _load_tune()
        for local_world_size in (1, 2, 4):
            with self.subTest(local_world_size=local_world_size), patch.dict(
                    os.environ, {"LOCAL_WORLD_SIZE": str(local_world_size)}):
                self.assertEqual(tune.get_local_world_size(), local_world_size)

    def test_missing_or_invalid_local_world_size_is_rejected(self):
        tune = _load_tune()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LOCAL_WORLD_SIZE must be set"):
                tune.get_local_world_size()
        for value in ("", "invalid", "0", "-1", "1.5"):
            with self.subTest(value=value), patch.dict(os.environ, {"LOCAL_WORLD_SIZE": value}):
                with self.assertRaisesRegex(ValueError, "LOCAL_WORLD_SIZE must be a positive integer"):
                    tune.get_local_world_size()


class LocalAutotuneTest(unittest.TestCase):

    def setUp(self):
        self.tune = _load_tune()
        cache = tempfile.TemporaryDirectory(prefix="function-autotune-local-")
        self.addCleanup(cache.cleanup)
        environment = patch.dict(os.environ, {"FLAGTREE_AUTOTUNE_CACHE_DIR": cache.name})
        environment.start()
        self.addCleanup(environment.stop)
        identity = patch.object(self.tune, "_get_cache_environment", return_value={"backend": "test"})
        identity.start()
        self.addCleanup(identity.stop)

    def use_benchmarker(self, benchmark):
        # Stub this isolated tuner without replacing the real driver or rolling back module imports.
        benchmark_patch = patch.object(self.tune.AutoTuner, "do_bench", property(lambda _: benchmark))
        benchmark_patch.start()
        self.addCleanup(benchmark_patch.stop)

    def test_prune_select_cache_and_argument_forwarding(self):
        calls, benchmarks, keys, pruned = [], [], [], []

        def key_fn(size, *, offset):
            keys.append((size, offset))
            return size, offset

        def prune_fn(config, size, *, offset):
            pruned.append((config["block"], size, offset))
            return config["block"] <= size

        def benchmark(call, *, quantiles):
            self.assertEqual(quantiles, [0.5])
            block = call()[1]
            benchmarks.append(block)
            return [100 / block]

        self.use_benchmarker(benchmark)

        @self.tune.autotune(
            config_space=[{"block": 16}, {"block": 32}, {"block": 64}],
            key_fn=key_fn,
            prune_fn=prune_fn,
        )
        def operation(size, *, block, offset):
            calls.append((size, block, offset))
            return size + offset, block

        self.assertEqual(operation(32, offset=7), (39, 32))
        self.assertEqual(operation(32, offset=7), (39, 32))
        self.assertEqual(benchmarks, [16, 32])
        self.assertEqual(len(pruned), 3)
        self.assertEqual(operation(64, offset=7), (71, 64))
        self.assertEqual(benchmarks, [16, 32, 16, 32, 64])
        self.assertEqual(len(operation.best_configs), 2)
        self.assertEqual(keys, [(32, 7), (32, 7), (64, 7)])
        self.assertEqual(len(calls), 8)

    def test_disabled_tuning_uses_first_config(self):

        def unexpected(*args, **kwargs):
            self.fail("disabled tuning must not evaluate the key, pruning, or benchmark")

        self.use_benchmarker(unexpected)

        @self.tune.autotune(
            [{"block": 16}, {"block": 32}],
            unexpected,
            unexpected,
        )
        def operation(value, block):
            return value, block

        self.assertEqual(operation(5, autotune=False), (5, 16))
        self.assertEqual(operation.best_configs, {})

    def test_local_keyword_override_requires_opt_in(self):

        @self.tune.autotune([{"block": 16}], lambda: None)
        def operation(block):
            return block

        with self.assertRaises(ValueError):
            operation(block=8, autotune=False)
        with self.assertWarns(UserWarning):
            self.assertEqual(operation(block=8, autotune=False, autotune_allow_arg_overwrite=True), 8)

    def test_local_arbitrary_config_objects_preserve_identity(self):
        config = object()
        self.use_benchmarker(lambda call, **kw: call() and 1)

        @self.tune.autotune([{"config": config}], lambda: None)
        def operation(config):
            return config

        self.assertIs(operation(), config)
        self.assertIs(operation.best_configs[None]["config"], config)

    def test_empty_pruning_and_unhashable_key_do_not_execute(self):

        def unexpected(*args, **kwargs):
            self.fail("invalid inputs must fail before execution")

        self.use_benchmarker(unexpected)
        empty = self.tune.autotune(
            [{"block": 16}],
            lambda: None,
            lambda config: False,
        )(unexpected)
        with self.assertRaises(ValueError):
            empty()
        self.assertEqual(empty.best_configs, {})

        unhashable = self.tune.autotune([{"block": 16}], lambda: [])(unexpected)
        with self.assertRaises(TypeError):
            unhashable()
        self.assertEqual(unhashable.best_configs, {})

    def test_invalid_and_failed_candidates_are_not_selected(self):

        def benchmark(call, **kwargs):
            block = call()
            if block == 1:
                raise RuntimeError("candidate failure")
            return {2: float("nan"), 3: float("inf"), 4: -1, 5: 2}[block]

        self.use_benchmarker(benchmark)

        @self.tune.autotune([{"block": b} for b in range(1, 6)], lambda: None)
        def operation(block):
            return block

        self.assertEqual(operation(), 5)


@unittest.skipUnless(
    os.environ.get("_FLAGTREE_TEST_DEVICE_AUTOTUNE") == "1",
    "set _FLAGTREE_TEST_DEVICE_AUTOTUNE=1 to run the real-device smoke test")
class LocalDeviceAutotuneTest(unittest.TestCase):

    def test_host_function_tunes_and_reuses_config(self):
        import torch
        from flagtree.tune import autotune
        from triton.runtime.driver import driver

        active = driver.active
        device = torch.device(active.get_active_torch_device())
        self.assertNotEqual(device.type, "cpu", "This smoke test requires an accelerator device")
        device_interface = active.get_device_interface()

        x_cpu = torch.arange(65536, dtype=torch.float32)
        y_cpu = torch.full_like(x_cpu, 3)
        x, y = x_cpu.to(device), y_cpu.to(device)
        output = torch.empty_like(x)
        configs = [{"chunks": 1}, {"chunks": 2}]
        calls = []

        # Use the real driver benchmark; disable disk caching so the first call must tune.
        @autotune(configs, key_fn=lambda x, y, out: x.numel(), cache_dir=False)
        def vector_add(x, y, out, *, chunks):
            calls.append(chunks)
            chunk_size = (x.numel() + chunks - 1) // chunks
            for start in range(0, x.numel(), chunk_size):
                end = start + chunk_size
                torch.add(x[start:end], y[start:end], out=out[start:end])
            return out

        result = vector_add(x, y, output)
        device_interface.synchronize()
        self.assertIs(result, output)
        torch.testing.assert_close(result.cpu(), x_cpu + y_cpu, rtol=0, atol=0)
        self.assertEqual(set(calls), {1, 2})
        self.assertGreater(len(calls), 2)  # warmup + tuning calls > 2
        self.assertEqual(set(vector_add.best_configs), {x.numel()})
        winner = vector_add.best_configs[x.numel()]
        self.assertIn(winner, configs)

        # Same workload key, changed values: reuse the winner but still execute the function.
        x_cpu.add_(7)
        x.copy_(x_cpu)
        output.fill_(float("nan"))
        calls.clear()
        result = vector_add(x, y, output)
        device_interface.synchronize()
        self.assertIs(result, output)
        torch.testing.assert_close(result.cpu(), x_cpu + y_cpu, rtol=0, atol=0)
        self.assertEqual(calls, [winner["chunks"]])
        self.assertIs(vector_add.best_configs[x.numel()], winner)


if __name__ == "__main__":
    unittest.main()
