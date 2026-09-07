# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT


def single_spec(**updates):
    value = {
        "kind": "single",
        "name": "weight_descriptor",
        "source": "rdata",
        "offset_bytes": 0,
        "dtype": "float32",
        "shape": (0, 4),
        "strides": (4, 1),
        "block_shape": (1, 4),
        "source_shape_axes": ((0,), ()),
        "padding": "zero",
    }
    value.update(updates)
    return value


def table_spec(**updates):
    value = {
        "kind": "table",
        "name": "weight_descriptor_table",
        "source": "rdata",
        "dtype": "float32",
        "block_shape": (1, 4),
        "padding": "zero",
        "swizzle_mode": 0,
        "entry_size_bytes": 128,
        "entries": ({
            "offset_bytes": 0,
            "shape": (2, 4),
            "strides": (4, 1),
            "source_shape_axes": ((), ()),
        },),
    }
    value.update(updates)
    return value


__all__ = ["single_spec", "table_spec"]
