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

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

import triton.language.core as tl
from triton._C.libtriton import ir

from .. import types as gpu_types

if TYPE_CHECKING:
    from ...semantic import TLESemantic


def is_backend_builder(builder) -> bool:
    # iluvatar_tle pipe ops keep the field-only operand list; the shared model
    # threads an SSA identity operand first. Decide from the backend the build
    # routes to rather than from builder capabilities: the installed Python
    # package and the libtriton it loads are versioned separately, and capability
    # probes on this builder did not distinguish the two ABIs.
    #
    # Synthetic builders used by the portable frontend tests are not
    # ``ir.builder`` instances and must keep the shared model, including on an
    # Iluvatar build that also runs those tests.
    if not isinstance(builder, ir.builder):
        return False
    try:
        from triton._C import libtriton
        if hasattr(libtriton, "iluvatar"):
            return True
    except ImportError:  # pragma: no cover - libtriton is always present
        pass
    from triton._flagtree_backend import FLAGTREE_BACKEND
    return FLAGTREE_BACKEND == "iluvatar"


def validate_pipe_options(scope, readers, one_shot, fields) -> None:
    if scope != "cta":
        raise ValueError("Iluvatar TLE pipe supports only scope='cta'")
    if not fields:
        raise ValueError("Iluvatar TLE pipe requires at least one payload field")
    if one_shot:
        raise ValueError("Iluvatar TLE pipe does not support one_shot")


# Iluvatar lowers through its own ``iluvatar_tle`` dialect, whose pipe ops keep
# the field-only operand list. Keep that ABI in the backend-owned frontend; the
# shared pipe model carries a separate SSA identity.
_validate_pipe_reader_name = gpu_types._validate_pipe_reader_name
_validate_pipe_reader_fields = gpu_types._validate_pipe_reader_fields


class pipe_value_type(gpu_types.pipe_value_type):

    def __init__(self, capacity: int, scope: str, name: Optional[str], fields, readers=None, one_shot=False):
        self.capacity = capacity
        self.scope = scope
        self.name = name
        self.fields = list(fields)
        self.readers = None if readers is None else tuple(readers)
        self.one_shot = one_shot

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple["pipe_value", int]:
        values = {}
        for field_name, ty in self.fields:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values[field_name] = value
        return pipe_value(self.capacity, self.scope, self.name, values, self.readers, one_shot=self.one_shot), cursor

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        for _, ty in self.fields:
            ty._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        name = "anon" if self.name is None else self.name
        fields = "_".join(f"{field_name}_{ty.mangle()}" for field_name, ty in self.fields)
        readers = "spsc" if self.readers is None else "readers_" + "_".join(self.readers)
        mode = "oneshot" if self.one_shot else "cyclic"
        return f"pipe_{self.scope}_{self.capacity}_{name}_{mode}_{readers}_{fields}"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.capacity == other.capacity and self.scope == other.scope
                and self.name == other.name and self.fields == other.fields and self.readers == other.readers
                and self.one_shot == other.one_shot)

    def __str__(self) -> str:
        fields = ", ".join(f"{name}: {ty}" for name, ty in self.fields)
        readers = "default" if self.readers is None else ", ".join(self.readers)
        mode = "one_shot" if self.one_shot else "cyclic"
        return f"pipe<{self.scope}, {self.capacity}, {mode}, readers=[{readers}], {fields}>"


class pipe_value(gpu_types.pipe_value):

    def _flatten_ir(self, handles) -> None:
        for field in self.fields.values():
            field._flatten_ir(handles)

    @property
    def type(self):
        return pipe_value_type(self.capacity, self.scope, self.name,
                               [(name, value.type) for name, value in self.fields.items()], self.readers,
                               one_shot=self.one_shot)

    @tl.builtin
    def writer(self, _semantic: TLESemantic | None = None):
        return pipe_writer(self)

    @tl.builtin
    def reader(self, name=None, fields=None, _semantic: TLESemantic | None = None):
        reader_name = _validate_pipe_reader_name(self, name)
        field_names = _validate_pipe_reader_fields(self, fields)
        return pipe_reader(self, reader_name=reader_name, field_names=field_names)

    @tl.builtin
    def wait_drained(self, _semantic=None):
        raise NotImplementedError("Iluvatar pipe lowering does not support wait_drained")


class pipe_writer(gpu_types.pipe_writer):

    type_cls = gpu_types.pipe_writer_type

    def __init__(self, pipe: pipe_value, reader_name=None, field_names=None):
        if reader_name is not None or field_names is not None:
            raise ValueError("tle.pipe.writer does not accept reader endpoint metadata")
        super().__init__(pipe)

    @tl.builtin
    def acquire(self, iter, _semantic: TLESemantic | None = None):
        stage, phase = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_writer_acquire(self.pipe._field_handles(_semantic), stage.handle, phase.handle,
                                                     self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                     self.pipe._field_names())
        return self.pipe._make_slot(stage, _semantic=_semantic)

    @tl.builtin
    def commit(self, iter, _semantic: TLESemantic | None = None):
        stage, _ = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_writer_commit(self.pipe._field_handles(_semantic), stage.handle,
                                                    self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                    self.pipe._field_names())

    @tl.builtin
    def close(self, iter, _semantic: TLESemantic | None = None):
        if self.pipe.one_shot:
            raise ValueError("tle.pipe one_shot pipes do not support close")
        stage, phase = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_writer_close(self.pipe._field_handles(_semantic), stage.handle, phase.handle,
                                                   self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                   self.pipe._field_names())


class pipe_reader(gpu_types.pipe_reader):

    type_cls = gpu_types.pipe_reader_type

    def _reader_field_names(self):
        if self.field_names is None:
            return self.pipe._field_names()
        return list(self.field_names)

    @tl.builtin
    def wait(self, iter, _semantic: TLESemantic | None = None):
        stage, phase = self.pipe._stage_phase(iter, _semantic=_semantic)
        is_closed = _semantic.builder.create_pipe_reader_wait(self.pipe._field_handles(_semantic), stage.handle,
                                                              phase.handle, self.pipe.capacity, self.pipe.scope,
                                                              self.pipe._ir_name(), self.pipe._field_names(),
                                                              self.reader_name or "", self._reader_field_names())
        slot = self.pipe._make_slot(stage, _semantic=_semantic, field_names=self.field_names)
        return gpu_types.pipe_wait_result(slot, tl.tensor(is_closed, tl.int1))

    @tl.builtin
    def release(self, iter, _semantic: TLESemantic | None = None):
        stage, _ = self.pipe._stage_phase(iter, _semantic=_semantic)
        _semantic.builder.create_pipe_reader_release(self.pipe._field_handles(_semantic), stage.handle,
                                                     self.pipe.capacity, self.pipe.scope, self.pipe._ir_name(),
                                                     self.pipe._field_names(), self.reader_name or "",
                                                     self._reader_field_names())


pipe_writer_type = gpu_types.pipe_writer_type
pipe_reader_type = gpu_types.pipe_reader_type
