from triton._C import libtriton


def _capability_query():
    # Shared NVIDIA/TLE builds expose the query on libtriton.tle. Vendor-only
    # backends such as iluvatar expose the same query on their own plugin.
    for name in ("tle", "iluvatar"):
        query = getattr(getattr(libtriton, name, None), "is_common_ir_enabled", None)
        if query is not None:
            return query
    return None


_query = _capability_query()
ENABLED = _query is not None and _query()
