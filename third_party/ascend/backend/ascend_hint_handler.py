# Copyright 2026- FlagOS Contributors

"""Ascend hint handler.

The 3.5/3.6 Ascend frontend currently has no backend-specific hint hooks; TLE
compile hints are handled by ``triton.language.extra.cann.extension``. Provide
the expected handler class so the generic hint manager does not emit a
missing-module warning on every kernel import.
"""

from triton.compiler.hint_manager import BaseHintHandler


class AscendHintHandler(BaseHintHandler):
    pass
