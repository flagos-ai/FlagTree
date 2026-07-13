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
