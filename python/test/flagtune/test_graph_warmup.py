from types import SimpleNamespace

import pytest

from triton.flagtune.runtime import graph_benchmark


class AsyncClock:
    """Submission is free; only device synchronization advances the clock."""

    def __init__(self, seconds_per_launch):
        self.now = 0.0
        self.seconds_per_launch = seconds_per_launch
        self.pending = 0
        self.batches = []

    def launch(self):
        self.pending += 1

    def synchronize(self):
        self.batches.append(self.pending)
        self.now += self.pending * self.seconds_per_launch
        self.pending = 0


@pytest.mark.parametrize("seconds_per_launch", [0.001, 0.5, 1.7])
def test_warmup_sizes_batches_from_completed_device_work(monkeypatch, seconds_per_launch):
    clock = AsyncClock(seconds_per_launch)
    monkeypatch.setattr(graph_benchmark, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    graph_benchmark._timed_warmup(clock.launch, 5, clock)

    assert clock.batches[:2] == [0, 1]
    assert clock.pending == 0
    assert clock.now >= 0.005
    assert clock.now <= max(0.005, seconds_per_launch) + 1e-12
    if seconds_per_launch > 0.005:
        assert clock.batches == [0, 1]


def test_fast_warmup_batches_launches_and_reaches_requested_duration(monkeypatch):
    clock = AsyncClock(0.00002)
    monkeypatch.setattr(graph_benchmark, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    graph_benchmark._timed_warmup(clock.launch, 5, clock)

    assert clock.pending == 0
    assert 0.005 <= clock.now < 0.00503
    assert 50 in clock.batches
    assert max(clock.batches) <= 50
    assert sum(clock.batches) >= 250


def test_zero_warmup_does_not_launch_or_synchronize():
    clock = AsyncClock(1.0)
    graph_benchmark._timed_warmup(clock.launch, 0, clock)
    assert clock.batches == []
    assert clock.pending == 0


def test_warmup_propagates_launch_failure(monkeypatch):
    clock = AsyncClock(0.001)
    monkeypatch.setattr(graph_benchmark, "time", SimpleNamespace(perf_counter=lambda: clock.now))

    def launch():
        raise RuntimeError("device launch failed")

    with pytest.raises(RuntimeError, match="device launch failed"):
        graph_benchmark._timed_warmup(launch, 5, clock)
    assert clock.batches == [0]
