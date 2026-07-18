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

from triton._C.libproton import proton as libproton
from .flags import flags
from functools import wraps

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class state:
    """
    A context manager and decorator for entering and exiting a state.

    Usage:
        context manager:
        ```python
        with proton.state("test0"):
            foo[1,](x, y)
        ```

        decorator:
        ```python
        @proton.state("test0")
        def foo(x, y):
            ...
        ```

    Args:
        name (str): The name of the state.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        if not flags.profiling_on:
            return self
        libproton.enter_state(self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not flags.profiling_on:
            return
        libproton.exit_state()

    def __call__(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if flags.profiling_on:
                libproton.enter_state(self.name)
            ret = func(*args, **kwargs)
            if flags.profiling_on:
                libproton.exit_state()
            return ret

        return wrapper


class metadata_state(state):

    def __init__(self) -> None:
        super().__init__(COMPUTE_METADATA_SCOPE_NAME)


def enter_state(name: str) -> None:
    libproton.enter_state(name)


def exit_state() -> None:
    libproton.exit_state()
