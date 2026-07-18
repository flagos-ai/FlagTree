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

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]


def apply_with_path(value: Any, fn: Callable[[ObjPath, Any], None], _path=None) -> None:
    if _path is None:
        _path = ()

    from triton._utils import is_iterable
    if is_iterable(value):
        for idx, item in enumerate(value):
            apply_with_path(item, fn, _path=(*_path, idx))
    else:
        fn(_path, value)


def _tuple_create(arg, contents):
    # NamedTuples and tuples have different construction semantics. NamedTuple
    # has a constructor that takes individual arguments, while tuple takes an
    # iterable. Both have type "tuple" making it difficult to distinguish
    # between them, but only NamedTuple has "_fields" and apparently this is how
    # everyone does the check.
    return type(arg)(*contents) if hasattr(arg, "_fields") else type(arg)(contents)
