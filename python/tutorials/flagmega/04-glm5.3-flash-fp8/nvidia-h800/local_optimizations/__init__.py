# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Install the tutorial's reviewed kernel layouts before compiling artifacts."""
from importlib import import_module
from contextlib import ExitStack
import atexit

_trial_contexts = None


def activate_base():
    """Install once per process; distribution and resource gates stay active."""
    import_module(f"{__name__}.norm")
    import_module(f"{__name__}.mla")
    import_module(f"{__name__}.shared_work")
    import_module(f"{__name__}.replica_work")
    import_module(f"{__name__}.remote_work")
    import_module(f"{__name__}.kda_work")
    import_module(f"{__name__}.mhc_post_work")
    import_module(f"{__name__}.dsa_work")
    global _trial_contexts
    if _trial_contexts is None:
        _trial_contexts = ExitStack()
        parallel = import_module(f"{__name__}.input_parallel")
        _trial_contexts.enter_context(parallel.activate())
        from .replica_roles import activate as replica_activate
        roles = {"decode_mla_attention_sparse_dsa_indexer_key.vectorized.compute":((1,0),),
                 "decode_mla_attention_sparse_dsa_indexer_gate.vectorized.compute":((1,1),),
                 "decode_mla_attention_sparse_dsa_indexer_weights.vectorized.compute":((1,2),(2,0))}
        _trial_contexts.enter_context(replica_activate(roles))
        atexit.register(_trial_contexts.close)


def activate():
    """Install the complete measured workload recipe before normal code generation."""
    from .kernel_recipe import activate as install
    install()
