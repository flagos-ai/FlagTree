# flagtree tle
import triton.language.core as tl

from .gpu import types as gpu_types


def _unwrap_pipe_constexpr(value):
    if isinstance(value, tl.constexpr):
        value = value.value
    if isinstance(value, tl.tuple):
        return tuple(_unwrap_pipe_constexpr(item) for item in value.values)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap_pipe_constexpr(item) for item in value)
    if value is None or isinstance(value, str):
        return value
    value = tl._unwrap_if_constexpr(value)
    if isinstance(value, (tuple, list)):
        return type(value)(_unwrap_pipe_constexpr(item) for item in value)
    return value


def _validate_public_name(kind, value):
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"tle.pipe {kind} name must be a Python identifier, got {value!r}")
    if value.startswith("_"):
        raise ValueError(f"tle.pipe {kind} name must not start with '_', got {value!r}")
    if value in {"fields", "readers"}:
        raise ValueError(f"tle.pipe {kind} name {value!r} is reserved")


def _validate_reader_names(readers):
    readers = _unwrap_pipe_constexpr(readers)
    if readers is None:
        return None
    if isinstance(readers, str) or not isinstance(readers, (tuple, list)):
        raise ValueError("tle.pipe readers must be a compile-time tuple/list of strings")
    if not readers:
        raise ValueError("tle.pipe readers must not be empty")
    names = []
    seen = set()
    for reader in readers:
        reader = _unwrap_pipe_constexpr(reader)
        _validate_public_name("reader", reader)
        if reader in seen:
            raise ValueError(f"tle.pipe readers must be unique, got duplicate {reader!r}")
        seen.add(reader)
        names.append(reader)
    return tuple(names)


@tl.builtin
def pipe(
    *,
    capacity,
    scope="cta",
    name=None,
    readers=None,
    one_shot=False,
    _semantic=None,
    **fields,
) -> gpu_types.pipe_value:
    """
    Create a typed pipe descriptor.

    Pipe is a hardware-independent TLE dataflow edge. Without readers= it is an
    SPSC pipe with one default reader. With readers=("name", ...) it is an
    explicit SPMC pipe with named reader endpoints. The current MVP accepts GPU
    shared-memory buffered tensors as fields and lowers CTA-scoped pipes to the
    GPU NVWS backend.

    one_shot=True models a single ready/full edge. The writer still commits and
    readers still wait, but acquire/release/close are not part of the contract.
    """
    capacity = _unwrap_pipe_constexpr(capacity)
    scope = _unwrap_pipe_constexpr(scope)
    name = _unwrap_pipe_constexpr(name)
    one_shot = _unwrap_pipe_constexpr(one_shot)
    reader_names = _validate_reader_names(readers)

    if not isinstance(capacity, int):
        raise ValueError(f"tle.pipe capacity must be a compile-time int, got {type(capacity).__name__}")
    if capacity <= 0:
        raise ValueError(f"tle.pipe capacity must be positive, got {capacity}")
    if scope != "cta":
        raise ValueError(f"tle.pipe MVP supports only scope='cta', got {scope!r}")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"tle.pipe name must be a string or None, got {type(name).__name__}")
    if not isinstance(one_shot, bool):
        raise ValueError(f"tle.pipe one_shot must be a compile-time bool, got {type(one_shot).__name__}")
    if not fields:
        raise ValueError("tle.pipe requires at least one payload field")

    for field_name, field in fields.items():
        _validate_public_name("field", field_name)
        if not isinstance(field, gpu_types.buffered_tensor):
            raise ValueError(
                f"tle.pipe field {field_name!r} must be tle.gpu.buffered_tensor, got {type(field).__name__}")
        if field.type.storage is not gpu_types.smem:
            raise ValueError(f"tle.pipe field {field_name!r} must use tle.gpu.smem storage, got {field.type.storage}")
        if len(field.shape) < 2:
            raise ValueError(f"tle.pipe field {field_name!r} must have rank >= 2, got {len(field.shape)}")
        if field.shape[0] != capacity:
            raise ValueError(
                f"tle.pipe field {field_name!r} leading dimension must equal capacity {capacity}, got {field.shape[0]}")

    _semantic.builder.create_pipe_create([field.handle for field in fields.values()], capacity, scope, name or "",
                                         list(fields.keys()), list(reader_names or ()), one_shot)
    return gpu_types.pipe_value(capacity, scope, name, fields, reader_names, one_shot=one_shot)


pipe_slot = gpu_types.pipe_slot
pipe_value = gpu_types.pipe_value
pipe_reader = gpu_types.pipe_reader
pipe_writer = gpu_types.pipe_writer
pipe_wait_result = gpu_types.pipe_wait_result
