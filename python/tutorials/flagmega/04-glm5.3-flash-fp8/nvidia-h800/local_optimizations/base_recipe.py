from . import activate_base as tutorial_activate
from triton.flagmega import ir as fm

def activate(*, entry_fence_override=None):
    from .concurrent_call_lifetimes import activate as concurrent_lifetimes_activate
    concurrent_lifetimes_activate()
    tutorial_activate()
    from .completion import activate as completion_activate
    completion_activate()
    from .entry_fences import activate as fences_activate
    fences_activate()
    # The complete fence renderer extends this layer before Native wrappers
    # install their numerical and load-scheduling behavior above it.
    if entry_fence_override is not None:
        entry_fence_override()
    from .native_chunk_precision import activate as chunk_precision_activate
    chunk_precision_activate()
    from .native_dsa_pool_scores import activate as pool_scores_activate
    pool_scores_activate()
    from .native_latent_attention_schedule import activate as latent_schedule_activate
    latent_schedule_activate()
    from .native_moe_down_load_schedule import activate as moe_down_load_activate
    moe_down_load_activate()
