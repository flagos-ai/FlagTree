# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Lossless diagnostic path components, independent from IR symbol spelling."""

from urllib.parse import quote


def encode_file_component(name: str) -> str:
    if not name:
        raise ValueError("A diagnostic path component cannot be empty.")
    # Escaping literal percent makes the encoding injective, including names
    # that already look encoded. Leading/trailing dots must not hide a file.
    value = quote(name, safe="-_.")
    if value.startswith("."):
        value = "%2E" + value[1:]
    if value.endswith("."):
        value = value[:-1] + "%2E"
    return value


__all__ = ["encode_file_component"]
