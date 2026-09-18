from triton._C import libtriton

# Vendor builds may omit the shared TLE plugin or its CommonIR capability query.
_query = getattr(getattr(libtriton, "tle", None), "is_common_ir_enabled", None)
ENABLED = _query is not None and _query()
