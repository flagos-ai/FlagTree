"""The serving contract is exact greedy tokens, not a cosine threshold."""


def require_same_tokens(actual, expected, *, context):
    actual, expected = list(actual), list(expected)
    if len(actual) != len(expected):
        raise AssertionError(f"{context}: token sequence length {len(actual)} != {len(expected)}")
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        if value != reference:
            raise AssertionError(f"{context}: first divergent token at {index}: {value} != {reference}")
