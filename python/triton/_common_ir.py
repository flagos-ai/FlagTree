from triton._C import libtriton

# MetaX exposes the query on ir; other backends may expose it on tle or omit it.
_query = getattr(getattr(libtriton, "tle", None), "is_common_ir_enabled", None)
if _query is None:
    _query = getattr(libtriton.ir, "is_common_ir_enabled", None)
ENABLED = _query is not None and _query()
