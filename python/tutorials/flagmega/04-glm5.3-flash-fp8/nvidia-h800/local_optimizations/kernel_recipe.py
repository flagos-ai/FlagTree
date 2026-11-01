"""Frozen complete-model owner-matched O-proj implementation override."""
from . import base_recipe as base
from . import fences
fm=base.fm
def activate():
    base.activate(entry_fence_override=fences.activate)
    from .join_override import activate as join_activate
    join_activate()
    from .qk64 import activate as qk_activate
    qk_activate()
    from .norm_layout import activate as norm_activate
    norm_activate()
    from .oproj import activate as oproj_activate
    oproj_activate()
    from .fusion import activate as fusion_activate
    fusion_activate()


    from .owner import activate as owner_activate
    global _owner_context
    _owner_context = owner_activate()
    _owner_context.__enter__()

    from .warp_owner import activate as warp_activate
    warp_activate()

    from .qa_norm import activate as qa_norm_activate
    qa_norm_activate()

    from .stage5 import activate as stage5_activate
    stage5_activate()

    from .qkv_concurrent import activate as qkv_activate
    qkv_activate()

    from .logits_load import activate as logits_load_activate
    logits_load_activate()

    from .recurrent_gb import activate as recurrent_gb_activate
    recurrent_gb_activate()
    from .conv_fb import activate as conv_fb_activate
    conv_fb_activate()

    from .head_load import activate as head_load_activate
    head_load_activate()
    from .dsa_inner import activate as dsa_inner_activate
    dsa_inner_activate()

    from .dsa_concurrent import activate as dsa_concurrent_activate
    dsa_concurrent_activate()

    from .router_quant import activate as router_quant_activate
    router_quant_activate()

    from .qb_n32 import activate as qb_activate
    qb_activate()

    from .head_k128 import activate as head_k128_activate
    head_k128_activate()
    from .mix_overlap import activate as mix_overlap_activate
    mix_overlap_activate()


    from .scalar_offsets import activate as scalar_offsets_activate
    scalar_offsets_activate()

    from .norm_distinct_owner import activate as distinct_owner_activate
    distinct_owner_activate()

    from .dsa_single_publication import activate as dsa_single_activate
    dsa_single_activate()

    from .mla_mix_overlap import activate as mla_mix_activate
    mla_mix_activate()

    from .dsa_column_owners import activate as dsa_column_activate
    dsa_column_activate()

    from .dsa_absorbed_overlap import activate as dsa_absorbed_activate
    dsa_absorbed_activate()

    from .kv_before_qb import activate as kv_before_qb_activate
    kv_before_qb_activate()

    from .v_expand32 import activate as v_expand32_activate
    v_expand32_activate()

    from .shared_merge_n4 import activate as shared_merge_activate
    shared_merge_activate()

    from .joint_dsa_absorbed_pool_scores import activate as joint_dsa_activate
    joint_dsa_activate()

    from .native_latent_probability_template import activate as probability_activate
    probability_activate()
