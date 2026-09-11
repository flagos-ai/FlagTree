"""Stable Cost Model failure boundaries; no GPU kernels are needed."""

import pytest

from triton.flagtune.runtime.errors import (
    BenchmarkError,
    ContractExecutionError,
    ModelValidationError,
    flagtune_error_boundary,
    flagtune_errors,
)


def test_boundary_preserves_existing_category_and_original_cause():
    original = ValueError("broken contract")
    with pytest.raises(ContractExecutionError) as caught:
        with flagtune_error_boundary(ModelValidationError):
            with flagtune_error_boundary(ContractExecutionError):
                raise original
    assert caught.value.phase == "postload"
    assert caught.value.__cause__ is original


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_boundary_does_not_swallow_process_control(exception):
    with pytest.raises(exception):
        with flagtune_error_boundary(ModelValidationError):
            raise exception()


def test_decorator_normalizes_source_error():
    original = FileNotFoundError("package missing")

    @flagtune_errors(ModelValidationError)
    def fail():
        raise original

    with pytest.raises(ModelValidationError) as caught:
        fail()
    assert caught.value.__cause__ is original


@pytest.mark.parametrize("samples", [[], [float("inf")], [float("nan")]])
def test_proposer_rejects_nonfinite_benchmark(samples):
    from triton.flagtune.runtime.proposer import _benchmark_candidate
    with pytest.raises(BenchmarkError):
        _benchmark_candidate(lambda *args: samples, {"BLOCK": 1})


def test_proposer_does_not_swallow_benchmark_exception():
    from triton.flagtune.runtime.proposer import _benchmark_candidate
    original = RuntimeError("kernel compile error")

    def fail(*args):
        raise original

    with pytest.raises(BenchmarkError) as caught:
        _benchmark_candidate(fail, {"BLOCK": 1})
    assert caught.value.__cause__ is original
