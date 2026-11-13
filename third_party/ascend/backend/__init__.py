# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
__version__ = '3.6.0'

import logging
from triton._C.libtriton.ascend import ir as ascend_ir

from .testing import do_bench_npu


def _apply_ascend_patch():
    from triton.compiler.code_generator import CodeGenerator

    if not getattr(CodeGenerator, "_ascend_patch_applied", False):
        _original_cg_init = CodeGenerator.__init__

        def _patched_cg_init(self, *args, **kwargs):
            """
            Monkey Patch for Ascend:
            Injects 'hacc.target' attribute into the module after initilization of module.
            """
            _original_cg_init(self, *args, **kwargs)
            options = self.builder.options
            context = self.context
            if hasattr(options, "arch") and options.arch:
                try:
                    builder = ascend_ir.ascendnpu_ir_builder(context, options.arch)

                    target_attr_str = f'#hacc.target<"{options.arch}">'
                    self.module.set_attr("hacc.target", builder.parse_attr(target_attr_str))
                except Exception as e:
                    logging.warning(f"[Ascend Patch] Failed to set hacc.target: {e}")

        CodeGenerator.__init__ = _patched_cg_init
        CodeGenerator._ascend_patch_applied = True

    # Patch compiler.parse to support Ascend-specific IR extensions
    # for ir_override and TRITON_KERNEL_OVERRIDE features.
    from triton.compiler import compiler as _compiler_module

    if not getattr(_compiler_module, "_ascend_parse_patch_applied", False):
        from pathlib import Path as _Path
        from triton._C.libtriton import ir as _ir

        # Keep the original community logic intact, adding Ascend extensions
        # to the existing text/binary branches.
        def _patched_parse(full_name, ext, context):
            if ext == "ttir" or ext == "ttgir":
                module = _ir.parse_mlir_module(full_name, context)
                module.context = context
                return module
            if ext in ("llir", "ptx", "amdgcn", "ttadapter", "bcmlir"):
                return _Path(full_name).read_text()
            if ext in ("cubin", "hsaco", "mlirbc", "npubin"):
                return _Path(full_name).read_bytes()

        _compiler_module.parse = _patched_parse
        _compiler_module._ascend_parse_patch_applied = True

    # Patch TritonSemantic.dot for Ascend-specific HF32 guard and
    # max_num_imprecise_acc warning.
    from triton.language.semantic import TritonSemantic

    if not getattr(TritonSemantic, "_ascend_dot_patch_applied", False):
        _original_dot = TritonSemantic.dot

        def _patched_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype):
            """
            Monkey Patch for Ascend:
            - HF32 precision only works for fp32 x fp32.
              When either input is not fp32, silently fall back
              to default precision (ieee).
            - Warn when max_num_imprecise_acc is explicitly set, since
              Ascend NPU does not support imprecise accumulation.
            """
            # Ascend does not support tf32, tf32 is converted to hf32. Avoid tf32 verification interception
            if input_precision == "tf32":
                input_precision = "hf32"
            # Ascend NPU does not support imprecise accumulation.
            # Force max_num_imprecise_acc to None so the upstream None
            # branch handles it (via max_num_imprecise_acc_default = 0),
            # avoiding the fp8 ValueError path which is NVIDIA-only.
            if max_num_imprecise_acc is not None:
                print("max_num_imprecise_acc in tl.dot is not supported on Ascend yet. "
                      "Thus it is ignored.")
                max_num_imprecise_acc = None

            return _original_dot(self, lhs, rhs, acc, input_precision, max_num_imprecise_acc, out_dtype)

        TritonSemantic.dot = _patched_dot
        TritonSemantic._ascend_dot_patch_applied = True

    # The core logical operators bitcast operands to i1, which fails for
    # wider integers. Use the existing truth-value cast for Ascend only:
    # TritonSemantic is shared by all backends in the current process.
    if not getattr(TritonSemantic, "_ascend_logical_patch_applied", False):
        from triton.language import core as _logical_core

        _original_logical_and = TritonSemantic.logical_and
        _original_logical_or = TritonSemantic.logical_or

        def _uses_ascend_options(semantic):
            # Resolve lazily: compiler.py invokes this patch while creating
            # NPUOptions, and imports must not introduce an eager cycle.
            from triton.backends.ascend.compiler import NPUOptions

            return isinstance(getattr(semantic.builder, "options", None), NPUOptions)

        def _patched_logical_and(self, input, other):
            if not _uses_ascend_options(self):
                return _original_logical_and(self, input, other)
            return self.and_(
                self.cast(input, _logical_core.int1),
                self.cast(other, _logical_core.int1),
            )

        def _patched_logical_or(self, input, other):
            if not _uses_ascend_options(self):
                return _original_logical_or(self, input, other)
            return self.or_(
                self.cast(input, _logical_core.int1),
                self.cast(other, _logical_core.int1),
            )

        TritonSemantic.logical_and = _patched_logical_and
        TritonSemantic.logical_or = _patched_logical_or
        TritonSemantic._ascend_logical_patch_applied = True

    # Install the unified, non-intrusive dtype interception for operators
    # whose dtypes are unsupported on Ascend. The rules live in
    # triton.backends.ascend.op_dtype_config, next to this backend.
    if not getattr(CodeGenerator, "_ascend_dtype_guard_applied", False):
        try:
            from triton.backends.ascend.dtype_guard import install_dtype_guard

            install_dtype_guard()
            CodeGenerator._ascend_dtype_guard_applied = True
        except Exception as e:
            logging.warning(f"[Ascend Patch] Failed to install dtype guard: {e}")

    # exp(fp16) is verified on A5, but the core whitelist rejects it.
    # Keep every other dtype and target on the original implementation.
    if not getattr(CodeGenerator, "_ascend_exp_fp16_patch_applied", False):
        from functools import wraps
        from triton import language as _language
        from triton.language import core as _core
        from triton.language import math as _math

        _original_exp = _math.exp
        _exp_check = getattr(_original_exp, "__wrapped__", None)
        _exp_body = getattr(_exp_check, "__wrapped__", None)
        if not callable(_exp_body) or getattr(_exp_check, "__name__", None) != "exp":
            raise RuntimeError("Unexpected decorator structure for triton.language.math.exp")

        @_core.builtin
        @wraps(_original_exp)
        def _patched_exp(x, _semantic=None):
            # Resolve lazily to avoid an import cycle with NPUOptions.
            from triton.backends.ascend.compiler import NPUOptions
            from triton.backends.ascend.utils import is_compile_on_910_95
            options = getattr(_semantic.builder, "options", None)
            if (isinstance(options, NPUOptions)
                    and is_compile_on_910_95(options.arch)
                    and isinstance(x, _core.tensor)
                    and x.dtype == _core.float16):
                return _exp_body(x, _semantic=_semantic)
            return _original_exp(x, _semantic=_semantic)

        _math.exp = _patched_exp
        if _language.exp is _original_exp:
            _language.exp = _patched_exp
        CodeGenerator._ascend_exp_fp16_patch_applied = True


__all__ = ["do_bench_npu"]
