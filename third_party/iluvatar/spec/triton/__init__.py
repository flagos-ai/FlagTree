from ._filecheck import filecheck_anonymize_ir, filecheck_default_kwargs, filecheck_make_ir, spec_get_stub_target
from .testing import get_max_tensorcore_tflops, nvsmi


def gluon_extend_language():
    # NOTE: Must use absolute path import.
    from triton.experimental.gluon import language, iluvatar
    language.BlockedLayout = iluvatar.IluvatarBlockedLayout
    language.DotOperandLayout = iluvatar.IluvatarDotOperandLayout
    language.SwizzledSharedLayout = iluvatar.IluvatarSwizzledSharedLayout
    language.iluvatar = iluvatar


__all__ = [
    "filecheck_make_ir",
    "get_max_tensorcore_tflops",
    "nvsmi",
    "spec_get_stub_target",
    "filecheck_anonymize_ir",
    "filecheck_default_kwargs",
    "gluon_extend_language",
]
