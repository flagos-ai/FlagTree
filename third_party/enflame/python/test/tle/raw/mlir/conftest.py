"""Prevent pytest from importing standalone scripts as test modules."""

collect_ignore_glob = [
    "01-*.py",
    "02-*.py",
]
