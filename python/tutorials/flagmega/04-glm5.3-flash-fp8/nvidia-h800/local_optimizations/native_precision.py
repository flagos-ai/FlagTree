"""Retain native RMS precision when selecting a different execution mesh."""
from dataclasses import replace

from triton.flagmega import ir as fm

from .native_tp4 import rewrite_qkv_norm_schedules


def apply_native_norm_precision(module):
    """Apply the native typed RMS policy after physical kernel selection.

    Execution meshes and FP8 implementation choices do not select a new RMS
    accumulation order. The native policy covers every complete FP32 RMS and
    BF16 Q/KV statistic by its input type and reduction ownership.
    """
    definitions = rewrite_qkv_norm_schedules(module)
    return fm.verify_module(replace(module, kernel_definitions=definitions))


def apply_native_decode_schedules(module):
    """Retain the post-selection policies used by the normal complete model.

    The physical execution mesh does not select different projection or RMS
    schedules. Keep these policies together when resuming physical selection.
    """
    from .native_tp4 import rewrite_mla_schedules
    from .head_schedule import rewrite as rewrite_head_schedules

    for rewrite in (rewrite_mla_schedules, rewrite_qkv_norm_schedules,
                    rewrite_head_schedules):
        module = fm.verify_module(replace(module, kernel_definitions=rewrite(module)))
    return module

