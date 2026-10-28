# TritonSDNN IR — declaration surface, no TableGen targets.
#
# This directory used to generate the Op/Attr declaration headers that
# gluon_ir.cc's SDNN binding segment compiled against.  That segment now lives
# in the prebuilt gluon_ir_sdnn.cc.o, so nothing in the build reads the .td
# files here any more; the corresponding tablegen targets and the deferred
# dependencies on them were removed from third_party/xpu/CMakeLists.txt.
#
# The .td files this directory used to hold are gone with the declaration
# package: they were staged in by the now-retired `xpu-sdnn-ir-decls` cache
# entry, and nothing in the build reads them any more.
