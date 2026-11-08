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
# Compile-time capability: whether the TLE / CommonIR plugin is built into this
# libtriton. Vendor builds without the plugin report False.
AVAILABLE = _query is not None and _query()


def _runtime_gate():
    # Runtime switch via the knobs system (env var FLAGTREE_COMMON_IR, default on).
    # Imported lazily to avoid an import cycle during triton initialization.
    from triton import knobs
    return bool(knobs.compilation.common_ir)


# Effective CommonIR switch, used by both the TLE frontend (tile.* emission) and
# the backend conversion pass (tile.* -> ttg.local_*): the plugin must be built
# in AND not disabled at runtime. Evaluated at import; set FLAGTREE_COMMON_IR in
# the environment before launching the process to toggle it.
ENABLED = AVAILABLE and _runtime_gate()
