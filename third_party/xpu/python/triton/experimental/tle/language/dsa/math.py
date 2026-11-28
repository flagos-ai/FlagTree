"""`tle.dsa.exp` and friends: SDNN math intrinsics with an explicit LUT domain.

Why: an SDNN EW LUT only covers the window its table was built over, and every
argument outside it becomes an endpoint constant (`kOverflow = kUnderflow = 0`).
`tl.exp`'s window is fixed at `[-15.984375, 0]`, so a positive argument comes
back as `1.0`.  A kernel that knows its own argument range can name it here and
the backend builds the table over that range instead.

How:
  - build the same `math.*` op the `tl.*` spelling builds;
  - attach the requested domain to it as `klx.lut_domain = "<min>:<max>"`;
  - `Legalize` copies it onto the `sdnn.ew` it emits, and `EWActTable` builds
    the table over it instead of over the default window.

`log`, `log2` and `rsqrt` reach the table *only* through these builtins: the
plain `tl.*` spellings keep the scalar `sdnn.math` loop, whose domain is
unbounded.  `exp` / `exp2` use a table either way; here the domain just changes.

Pass `little_core=True` instead of a domain to compute on the scalar
`sdnn.math` loop: no table, no window, and the exact result.  That is the escape
hatch when the piecewise-linear table is not accurate enough -- roughly
`h^2 / (8 x^2 ln2)` for `log2`, worst at the low end of the window.
`tle.dsa.exp2` has no scalar spelling (`MathFunc` has no `exp2`), so it rejects the
flag instead of silently doing something else.
"""

import triton.language.core as core

__all__ = ["exp", "exp2", "log", "log2", "rsqrt"]

#: Attribute carrying the caller's LUT domain, as `"<min>:<max>"`.
LUT_DOMAIN_ATTR = "klx.lut_domain"

#: Attribute asking `Legalize` for the scalar `sdnn.math` op instead of a table.
LITTLE_CORE_ATTR = "klx.little_core"


def _as_compile_time_number(value, what):
    """`value` as a float, or a readable error.

    Why: the table is generated at compile time and its bytes go into the
    cache key, so a runtime scalar cannot be a window.  Without this check the
    failure surfaces as `float() argument must be ...`, or worse, as Triton's
    "did you forget @triton.jit" wrapper message.
    """
    value = core._unwrap_if_constexpr(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"tle.dsa.*: {what} must be a compile-time number (int or float), got "
                        f"{type(value).__name__}; the LUT is built at compile time, so a "
                        "runtime value cannot be used -- pass a constant or use "
                        "little_core=True")
    return float(value)


def _stamp_domain(handle, lut_min, lut_max, little_core, _semantic):
    """Attach the caller's choice to the op that produced `handle`.

    Exactly one of the two forms is accepted: a `(lut_min, lut_max)` window, or
    `little_core=True`.  Mixing them is a mistake -- the window would be ignored
    -- so it raises rather than silently picking one.

    Rejecting `max <= min` here keeps a typo from reaching `EWActTable`, where
    it would only surface as a table built over a degenerate window.
    """
    little_core = core._unwrap_if_constexpr(little_core)
    has_domain = lut_min is not None and lut_max is not None
    if little_core:
        if lut_min is not None or lut_max is not None:
            raise ValueError("tle.dsa.*: pass either a (lut_min, lut_max) window or "
                             "little_core=True, not both")
        handle.set_attr(LITTLE_CORE_ATTR, _semantic.builder.get_unit_attr())
        return
    if not has_domain:
        raise ValueError("tle.dsa.*: pass a (lut_min, lut_max) window for the LUT, "
                         "or little_core=True for the scalar core")
    lut_min = _as_compile_time_number(lut_min, "lut_min")
    lut_max = _as_compile_time_number(lut_max, "lut_max")
    if not lut_max > lut_min:
        raise ValueError("tle.dsa.*: the LUT domain must satisfy max > min, got "
                         f"[{lut_min}, {lut_max}]")
    handle.set_attr(LUT_DOMAIN_ATTR, _semantic.builder.get_string_attr(f"{float(lut_min)}:{float(lut_max)}"))


@core.builtin
def exp(x, lut_min=None, lut_max=None, little_core=False, _semantic=None):
    x = _semantic.to_tensor(x)
    handle = _semantic.builder.create_exp(x.handle)
    _stamp_domain(handle, lut_min, lut_max, little_core, _semantic)
    return core.tensor(handle, x.type)


@core.builtin
def exp2(x, lut_min=None, lut_max=None, little_core=False, _semantic=None):
    if core._unwrap_if_constexpr(little_core):
        raise NotImplementedError("tle.dsa.exp2: sdnn.math has no exp2 scalar function (MathFunc stops at "
                                  "exp / log / log2 / rsqrt); pass a (lut_min, lut_max) window instead")
    x = _semantic.to_tensor(x)
    handle = _semantic.builder.create_exp2(x.handle)
    _stamp_domain(handle, lut_min, lut_max, little_core, _semantic)
    return core.tensor(handle, x.type)


@core.builtin
def log(x, lut_min=None, lut_max=None, little_core=False, _semantic=None):
    x = _semantic.to_tensor(x)
    handle = _semantic.builder.create_log(x.handle)
    _stamp_domain(handle, lut_min, lut_max, little_core, _semantic)
    return core.tensor(handle, x.type)


@core.builtin
def log2(x, lut_min=None, lut_max=None, little_core=False, _semantic=None):
    x = _semantic.to_tensor(x)
    handle = _semantic.builder.create_log2(x.handle)
    _stamp_domain(handle, lut_min, lut_max, little_core, _semantic)
    return core.tensor(handle, x.type)


@core.builtin
def rsqrt(x, lut_min=None, lut_max=None, little_core=False, _semantic=None):
    x = _semantic.to_tensor(x)
    handle = _semantic.builder.create_rsqrt(x.handle)
    _stamp_domain(handle, lut_min, lut_max, little_core, _semantic)
    return core.tensor(handle, x.type)
