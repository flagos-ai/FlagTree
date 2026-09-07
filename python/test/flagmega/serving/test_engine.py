import pytest

from triton.flagmega.serving import TextGenerationEngine


def test_real_prefix_is_consumed_and_final_output_is_not_yet_cached(backend):
    engine = TextGenerationEngine(backend)
    result = engine.generate([1, 2, 3], max_new_tokens=3)
    assert result.token_ids == (6, 12, 24)
    assert engine.cached_tokens == (1, 2, 3, 6, 12)
    assert backend.tokens == list(engine.cached_tokens)
    assert result.metrics.prefill_tokens == 3
    assert len(result.metrics.decode_step_ms) == 2
    assert result.to_dict()["prompt_tokens"] == [1, 2, 3]
    assert result.to_dict()["prompt_token_count"] == 3


@pytest.mark.parametrize("prompt", [[1, 2, 3, 6, 12, 24, 7], [1, 2, 9], [1, 2, 3]])
def test_reused_rewritten_and_shortened_prefix_match_fresh_generation(backend, prompt):
    engine = TextGenerationEngine(backend)
    engine.generate([1, 2, 3], max_new_tokens=3)
    continued = engine.generate(prompt, max_new_tokens=3)
    fresh = TextGenerationEngine(type(backend)()).generate(prompt, max_new_tokens=3)
    assert continued.token_ids == fresh.token_ids
    assert continued.metrics.cached_tokens > 0
    assert continued.metrics.prefill_tokens >= 1


def test_eos_first_token_has_no_decode_interval_and_ignore_eos_is_explicit(backend):
    engine = TextGenerationEngine(backend, eos_token_ids=[3])
    result = engine.generate([1, 2], max_new_tokens=4)
    assert result.finish_reason == "eos"
    assert result.token_ids == (3,)
    assert result.metrics.to_dict()["decode_tokens_per_second"] is None
    assert "decode: 0 tokens" in result.metrics.format()
    engine.reset()
    assert len(engine.generate([1, 2], max_new_tokens=4, ignore_eos=True).token_ids) == 4


@pytest.mark.parametrize("prompt,count", [([], 2), ([True], 2), ([32], 2), ([1], 0), ([1], True), ([1]*63, 2)])
def test_invalid_request_does_not_mutate_existing_state(backend, prompt, count):
    engine = TextGenerationEngine(backend)
    engine.generate([2], max_new_tokens=2)
    original = engine.cached_tokens
    with pytest.raises(ValueError):
        engine.generate(prompt, max_new_tokens=count)
    assert engine.cached_tokens == original


def test_interrupted_or_reentrant_callback_invalidates_uncertain_state(backend):
    engine = TextGenerationEngine(backend)
    def interrupt(token):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        engine.generate([1, 2], max_new_tokens=2, on_token=interrupt)
    assert engine.cached_tokens == ()
    assert backend.length == 0
    def recurse(token):
        engine.generate([1], max_new_tokens=1)
    with pytest.raises(RuntimeError, match="reentrant"):
        engine.generate([1, 2], max_new_tokens=2, on_token=recurse)
    assert backend.length == 0


def test_external_backend_mutation_is_detected(backend):
    engine = TextGenerationEngine(backend)
    backend.append(1)
    with pytest.raises(RuntimeError, match="disagree"):
        engine.generate([1], max_new_tokens=1)
