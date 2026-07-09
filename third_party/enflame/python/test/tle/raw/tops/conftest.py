"""Prevent pytest from importing non-test scripts as test modules."""

collect_ignore_glob = [
    "01-*.py",
    "02-*.py",
    "03-*.py",
]
