def distribution(xs):
    return dict(mean_ms=statistics.mean(xs), sample_stddev_ms=statistics.stdev(xs), range_ms=[min(xs), max(xs)])
