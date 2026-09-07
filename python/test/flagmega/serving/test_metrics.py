from triton.flagmega.serving.metrics import GenerationMetrics, summarize


def test_denominators_exclude_first_token_from_decode_and_cached_tokens_from_prefill():
    metrics = GenerationMetrics(10, 7, 3, 3, 30., 35., (2., 4.), 42.)
    report = metrics.to_dict()
    assert report["prefill_tokens_per_second"] == 100
    assert report["decode_tokens"] == 2
    assert report["decode_tokens_per_second"] == 2000/6
    assert report["output_tokens_per_second"] == 3000/42
    assert report["decode_latency_ms"]["median"] == 3
    assert "6.00 ms" in metrics.format()
    assert summarize([])["median"] is None
