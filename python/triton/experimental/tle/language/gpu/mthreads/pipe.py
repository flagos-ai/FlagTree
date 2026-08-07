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


def is_backend_builder(builder) -> bool:
    # The mthreads-only libtriton can be loaded while backend-independent TLE
    # frontend tests use a synthetic builder. Gate the restricted contract on
    # a backend-local native capability so those public tests retain the
    # portable pipe model.
    return hasattr(builder, "mark_musa_tle_auto_shared_layout")


def validate_pipe_options(scope, readers, one_shot, fields) -> None:
    if scope != "cta":
        raise ValueError("initial mthreads tle.pipe supports only scope='cta'")
    if len(fields) != 1:
        raise ValueError("initial mthreads tle.pipe requires exactly one payload field")
    if readers is not None:
        raise ValueError("initial mthreads tle.pipe supports only the default SPSC reader")
    if one_shot:
        raise ValueError("initial mthreads tle.pipe does not support one_shot=True")
