from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from triton.backends.ascend import testing


class TestNPUCallTiming(unittest.TestCase):

    def setUp(self):
        self.clock = 0.0
        self.calls = 0
        self.callback = None
        owner = self

        class Event:

            def __init__(self, enable_timing):
                assert enable_timing

            def record(self, stream=None):
                if not hasattr(self, "time"):
                    owner.clock += 5.0  # Lazy event initialization must be untimed.
                self.time = owner.clock

            def elapsed_time(self, end):
                return end.time - self.time

        class Monitor:

            def start(self, callback):
                owner.callback = callback

            def stop(self):
                owner.callback = None

        self.buffer = SimpleNamespace(zero_=self.clear_cache)
        self.torch = SimpleNamespace(npu=SimpleNamespace(Event=Event, synchronize=Mock(), current_stream=Mock(
            return_value=object())))
        self.modules = patch.dict("sys.modules", {"torch": self.torch, "torch_npu": SimpleNamespace()})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        monitor = patch.object(testing, "KernelMonitor", Monitor)
        monitor.start()
        self.addCleanup(monitor.stop)
        environment_check = patch.object(testing, "_check_mspti_env")
        environment_check.start()
        self.addCleanup(environment_check.stop)
        environment = patch.dict("os.environ", {"TRITON_BENCH_METHOD": "default"})
        environment.start()
        self.addCleanup(environment.stop)

    def clear_cache(self):
        self.clock += 7.0

    def mixed_task_call(self):
        self.calls += 1
        # 100 ms of DSA work is absent from KernelMonitor's 40 ms kernel record.
        start = self.clock
        self.clock += 140.0
        if self.callback is not None:
            self.callback(
                SimpleNamespace(name="vector_kernel", start=int((start + 100) * 1e6), end=int(self.clock * 1e6)))

    def test_default_covers_non_kernel_work(self):
        result = testing.do_bench_npu(self.mixed_task_call, warmup=2, active=3)
        self.assertEqual(result, 140.0)
        self.assertEqual(self.calls, 6)

    def test_explicit_kernel_timing_retains_monitor(self):
        result = testing.do_bench_npu(self.mixed_task_call, warmup=2, active=3, timing_method="kernel")
        self.assertEqual(result, 40.0)

    def test_list_returns_one_complete_call_time_per_function(self):

        def short_call():
            self.clock += 25.0

        result = testing.do_bench_npu([self.mixed_task_call, short_call], warmup=0, active=2)
        self.assertEqual(result, [140.0, 25.0])

    def test_overlapping_device_tasks_are_not_summed(self):

        def concurrent_call():
            self.clock += 20.0

        self.assertEqual(testing.do_bench_npu(concurrent_call, warmup=0, active=2), 20.0)

    def test_cache_clear_is_outside_timed_interval(self):
        with patch.object(testing.runtime, "driver",
                          SimpleNamespace(active=SimpleNamespace(get_empty_cache_for_benchmark=lambda: self.buffer))):
            result = testing.do_bench_npu(self.mixed_task_call, warmup=1, active=3, clear_l2_cache=True)
        self.assertEqual(result, 140.0)

    def test_filtered_kernel_keeps_profiler_path(self):
        with patch.object(testing, "do_bench_npu_profiler", return_value=9.0) as profiler:
            result = testing.do_bench_npu(self.mixed_task_call, target_kernel_name="vector_kernel")
        self.assertEqual(result, 9.0)
        profiler.assert_called_once()

    def test_requested_profiler_artifacts_keep_profiler_path(self):
        with patch.object(testing, "do_bench_npu_profiler", return_value=9.0) as profiler:
            result = testing.do_bench_npu(self.mixed_task_call, prof_dir="results", keep_res=True)
        self.assertEqual(result, 9.0)
        profiler.assert_called_once()

    def test_auto_preserves_npu_autotuner_kernel_timing(self):
        with patch.dict("os.environ", {"TRITON_BENCH_METHOD": "npu"}):
            result = testing.do_bench_npu([self.mixed_task_call], warmup=2, active=3)
        self.assertEqual(result, 40.0)

    def test_explicit_events_can_measure_autotuner_calls(self):
        with patch.dict("os.environ", {"TRITON_BENCH_METHOD": "npu"}):
            result = testing.do_bench_npu([self.mixed_task_call], warmup=2, active=3, timing_method="event")
        self.assertEqual(result, 140.0)

    def test_explicit_events_rejects_kernel_filter(self):
        with self.assertRaises(ValueError):
            testing.do_bench_npu(self.mixed_task_call, target_kernel_name="kernel", timing_method="event")

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            testing.do_bench_npu(self.mixed_task_call, timing_method="unknown")

    def test_invalid_iteration_counts(self):
        for warmup, active in ((-1, 3), (0, 0), (0, -1)):
            with self.subTest(warmup=warmup, active=active), self.assertRaises(ValueError):
                testing.do_bench_npu(self.mixed_task_call, warmup=warmup, active=active)


if __name__ == "__main__":
    unittest.main()
