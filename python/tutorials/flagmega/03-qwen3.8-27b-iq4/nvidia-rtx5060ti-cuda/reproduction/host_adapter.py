"""Keep the original complete decode graph inside the owned plan capture.

Only capture ownership/teardown change. The base eager launch, numerical
checks, state reset, prompt injection and timed replay/token-return loop remain
the original benchmark implementation. Phase durations below are host wall
observations outside the timed decode samples, not isolated GPU costs.
"""
from time import perf_counter


def adapt_decode_context(base, *, phase_timings=None):
    class CounterEpochDecodeContext(base):
        def __init__(self, *args, **kwargs):
            start = perf_counter()
            super().__init__(*args, **kwargs)
            if phase_timings is not None:
                phase_timings['load_prepare_seconds'] = perf_counter() - start

        def capture(self):
            self.reset()
            self.stream.synchronize()
            slot = self.values['paged_state.slot_mapping']
            length = self.values['paged_state.seq_lens']
            input_ids, token = self.values['input_ids'], self.token
            start = perf_counter()
            self.graph = self.runtime.capture_cpp_programs(
                self.arguments, stream=self.stream,
                before=lambda: slot.copy_(length),
                after=lambda: input_ids.copy_(token.reshape_as(input_ids)),
                keepalive=(slot, length, input_ids, token),
            )
            if phase_timings is not None:
                phase_timings['capture_host_seconds'] = perf_counter() - start
            return self.graph

        def close(self):
            # Retirement also owns quarantined captures which raised before
            # capture_cpp_programs returned a wrapper. It synchronizes the real
            # stream and destroys the real graph before releasing any roots.
            self.runtime.retire_cpp_program_graphs()
            self.runtime.close(synchronize=False)
            self.graph = None

    return CounterEpochDecodeContext


def install_benchmark_adapter(benchmark, execution_binding):
    """Attach explicit provenance without changing the original run body."""
    original_context, original_run = benchmark.DecodeContext, benchmark.run

    def run(args, report, save):
        report['execution_adapter'] = execution_binding
        phases = report.setdefault('execution_phase_seconds', {})
        report['execution_phase_limits'] = (
            'load_prepare includes original context construction and runtime.prepare; '
            'capture_host includes the owned full-graph capture. These are outside '
            'the original host-token samples and are not pure GPU costs. '
            'Per-replay host enqueue is included in the unchanged timer and is '
            'not separately instrumented or subtracted.'
        )
        benchmark.DecodeContext = adapt_decode_context(original_context, phase_timings=phases)
        try:
            return original_run(args, report, save)
        finally:
            benchmark.DecodeContext = original_context

    benchmark.run = run
