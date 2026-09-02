# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class scoped_dict(Generic[K, V]):
    stack: list[dict[K, V]] = field(default_factory=list)

    def __init__(self, d: dict[K, V] | None = None) -> None:
        self.stack = [d or {}]

    def __getitem__(self, key: K) -> V:
        for d in reversed(self.stack):
            if key in d:
                return d[key]
        raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        self.stack[-1][key] = value

    def __contains__(self, key: K) -> bool:
        return any(key in d for d in reversed(self.stack))

    def setdefault(self, key: K, value: V) -> V:
        return self.stack[-1].setdefault(key, value)

    @contextmanager
    def scope(self, d: dict[K, V] | None = None) -> Generator[None, None, None]:
        self.stack.append(d or {})
        try:
            yield
        finally:
            self.stack.pop()
