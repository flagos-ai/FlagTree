from triton._C.libtriton import tle as tle_ir

ENABLED = tle_ir.is_common_ir_enabled()
