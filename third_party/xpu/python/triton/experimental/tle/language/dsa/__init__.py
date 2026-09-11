# tle.dsa / tle.pipe are not available in the FlagTree XPU build.
#
# Both surfaces sit on the SDNN pipeline: `tle.dsa.alloc` emits a `memref.alloc`
# whose layout comes from TritonSDNN's `getAlignedMemRefType`, and every other
# entry point (subview / fill / copy / to_buffer, and all of tle.pipe) builds a
# TritonSDNN op. Their Python bindings therefore have to be compiled against
# TritonSDNN's IR headers, which this tree does not carry: FlagTree consumes the
# SDNN dialect as prebuilt objects (third_party/xpu/lib/Dialect/TritonSDNN/**,
# see FLAGTREE_XPU_SYNC_GUIDE.md) and ships no TritonSDNNOps.td / Dialect.h to
# generate the C++ op classes from.
#
# `tle.raw` on the cluster path and all of `tle.gpu` need no SDNN header -- their
# ops (`triton_xpu.raw`, `triton_xpu.tle_copy_g2l` and friends) are defined in
# the TritonXPU dialect, whose sources are in this tree -- so those work.

_MESSAGE = ("tle.{api} is not available in the FlagTree XPU build: it lowers to the TritonSDNN "
            "dialect, which FlagTree consumes as prebuilt objects without the IR headers needed "
            "to build its Python bindings. Available here: tle.raw (cluster path, "
            '@tle.raw.dialect("xpu3") + tle.raw.call, launched without is_sdnn) and tle.gpu '
            "(alloc / copy / local_ptr).")

__all__ = ["alloc", "subview", "fill", "copy", "to_buffer", "L1D", "L1W", "TM", "uni_sram"]


def _unsupported(api):

    def stub(*args, **kwargs):
        raise NotImplementedError(_MESSAGE.format(api=f"dsa.{api}"))

    stub.__name__ = api
    stub.__triton_builtin__ = True
    return stub


alloc = _unsupported("alloc")
subview = _unsupported("subview")
fill = _unsupported("fill")
copy = _unsupported("copy")
to_buffer = _unsupported("to_buffer")


class _UnsupportedSpace:
    """Address-space handle that only reports why it cannot be used."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"tle.dsa.{self._name} (unavailable)"

    def __getattr__(self, attr):
        raise NotImplementedError(_MESSAGE.format(api=f"dsa.{self._name}"))


L1D = _UnsupportedSpace("L1D")
L1W = _UnsupportedSpace("L1W")
TM = _UnsupportedSpace("TM")
uni_sram = _UnsupportedSpace("uni_sram")
