# flagtree tle
from importlib import import_module

from .core import load
from .tile import (
    tensor_view_type,
    tensor_view,
    make_partition_view,
    make_strided_view,
    make_gather_scatter_view,
    tile_load,
    tile_store,
)

from . import gpu, raw

__all__ = [
    "load",
    "dsa",
    "tensor_view_type",
    "tensor_view",
    "tile_load",
    "tile_store",
    "make_partition_view",
    "make_strided_view",
    "make_gather_scatter_view",
]


def __getattr__(name):
    if name == "dsa":
        module = import_module(f"{__name__}.dsa")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
