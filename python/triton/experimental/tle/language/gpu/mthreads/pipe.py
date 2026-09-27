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


def validate_pipe_options(scope, readers, one_shot, fields, builder=None) -> None:
    if scope != "cta":
        raise ValueError("initial mthreads tle.pipe supports only scope='cta'")
    if not fields:
        raise ValueError("mthreads tle.pipe requires at least one payload field")
    # Version 2 extends the grouped-completion contract to three payloads.
    # Keep the version check at the Python boundary so a newer package paired
    # with an older native libtriton fails closed before emitting IR that the
    # old LowerPipe pass cannot consume.
    multifield_version = getattr(builder, "mthreads_tle_multifield_pipe_version", 0)
    # The original mthreads LowerPipe accepted exactly one payload.  Version
    # 1 adds grouped completion for two fields; version 2 extends that to
    # three.  Treat an absent marker as version 0 so a newer Python frontend
    # cannot emit multi-field IR for an old native pass.
    max_fields = 3 if multifield_version >= 2 else 2 if multifield_version >= 1 else 1
    if len(fields) > max_fields:
        max_fields_name = {1: "one", 2: "two", 3: "three"}[max_fields]
        raise ValueError(
            f"mthreads tle.pipe supports at most {max_fields_name} payload fields"
        )
    if readers is not None:
        raise ValueError("initial mthreads tle.pipe supports only the default SPSC reader")
    if one_shot:
        one_shot_version = getattr(builder, "mthreads_tle_one_shot_pipe_version", 0)
        if one_shot_version < 1:
            raise ValueError(
                "mthreads TLE one_shot pipes require native one_shot capability"
            )
