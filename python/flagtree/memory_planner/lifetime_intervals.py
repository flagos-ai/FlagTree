"""Closed logical lifetime intervals for compiler view groups."""


def union_lifetime_intervals(intervals):
    """Cover all alias members with one closed logical lifetime interval.

    The caller owns the event coordinate system and alias evidence.
    """
    pairs = list(intervals)
    if not pairs or any(
        not isinstance(pair, (tuple, list)) or len(pair) != 2
        or any(type(index) is not int or index < 0 for index in pair)
        or pair[0] > pair[1] for pair in pairs
    ):
        raise ValueError("expected nonempty, ordered, nonnegative lifetime intervals")
    return min(pair[0] for pair in pairs), max(pair[1] for pair in pairs)
