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

import ast
from typing import Optional
from ..errors import TritonError


class CompilationError(TritonError):
    """Base class for all errors raised during compilation"""
    source_line_count_max_in_message = 12

    def _format_message(self) -> str:
        node = self.node
        if self.src is None:
            source_excerpt = " <source unavailable>"
        else:
            if hasattr(node, 'lineno'):
                source_excerpt = self.src.split('\n')[:node.lineno][-self.source_line_count_max_in_message:]
                if source_excerpt:
                    source_excerpt.append(' ' * node.col_offset + '^')
                    source_excerpt = '\n'.join(source_excerpt)
                else:
                    source_excerpt = " <source empty>"
            else:
                source_excerpt = self.src

        message = "at {}:{}:\n{}".format(node.lineno, node.col_offset, source_excerpt) if hasattr(
            node, 'lineno') else source_excerpt
        if self.error_message:
            message += '\n' + self.error_message
        return message

    def __init__(self, src: Optional[str], node: ast.AST, error_message: Optional[str] = None):
        self.src = src
        self.node = node
        self.error_message = error_message
        self.message = self._format_message()

    def __str__(self):
        return self.message

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return type(self), (self.src, self.node, self.error_message)


class CompileTimeAssertionFailure(CompilationError):
    """Specific exception for failed tests in `static_assert` invocations"""
    pass


class UnsupportedLanguageConstruct(CompilationError):
    pass
