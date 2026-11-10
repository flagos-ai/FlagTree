def finite(v):
    if isinstance(v, float):
        assert math.isfinite(v)
    elif isinstance(v, dict):
        for x in v.values():
            finite(x)
    elif isinstance(v, list):
        for x in v:
            finite(x)

def stats(samples):
    assert all((type(x) in (int, float) and math.isfinite(x) and (x > 0) for x in samples))
    values = sorted(samples)
    index = (len(values) - 1) * 0.95
    p95 = values[math.floor(index)] + (values[math.ceil(index)] - values[math.floor(index)]) * (index - math.floor(index))
    mean = statistics.mean(samples)
    return dict(sample_count=len(samples), mean_ms=mean, median_ms=statistics.median(samples), p95_ms=p95, max_ms=max(samples), tokens_per_second=1000 / mean)
