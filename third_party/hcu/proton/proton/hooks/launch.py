# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ..state import enter_state, exit_state
from triton.compiler import LazyDict
from .hook import Hook
from triton._C.libproton import proton as libproton
from contextvars import ContextVar

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"

op_name = ContextVar("op_name", default=None)
id = ContextVar("id", default=None)


class LaunchHook(Hook):
    # Highest priority
    priority = 100
    # This is a singleton class
    _instance = None
    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LaunchHook, cls).__new__(cls)
        return cls._instance

    def init_handle(self, module, function, name: str, metadata_group: dict, hash: str) -> None:
        pass

    def activate(self):
        pass

    def deactivate(self):
        pass

    def enter(self, metadata: LazyDict) -> None:
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        lazy_metadata = metadata.get()
        exit_state()
        fn_metrics = {k: lazy_metadata[k] for k in LaunchHook.metrics if k in lazy_metadata}
        op_name.set(lazy_metadata["name"])
        id.set(libproton.record_scope())
        libproton.enter_op(id.get(), lazy_metadata["name"])
        libproton.add_metrics(id.get(), fn_metrics)

    def exit(self, metadata: LazyDict) -> None:
        libproton.exit_op(id.get(), op_name.get())
