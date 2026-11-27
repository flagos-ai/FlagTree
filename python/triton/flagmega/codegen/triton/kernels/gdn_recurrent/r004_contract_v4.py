# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native positional names pinned by r004 v4 §§2.9–2.10."""
NATIVE_ARGUMENTS = {
    "clear": ("StateParent", "Layer", "Scale", "Bias"),
    "gather": ("StateParent", "Layer", "Selector", "GatheredState"),
    "pb": ("X", "BW", "B"), "pa": ("X", "AW", "A"),
    "beta": ("B", "Beta"), "bias": ("A", "DT", "Biased"),
    "gate": ("Biased", "ALog", "G"), "q": ("QKV", "QN"), "k": ("QKV", "KN"),
    "core": ("QN", "KN", "QKV", "Beta", "G", "GatheredState", "StateParent", "Layer", "Core"),
    "rms": ("Core", "Norm", "Weighted"), "y": ("Weighted", "Z", "Y"),
}


def native_arguments(plan):
    if plan['module'] in ('r017_T_full_p5_split', 'r017_T_full_p5_split4', 'r017_T_full_p5_split8', 'r017_T_full_p5_split4lut'):
        args=native_arguments({'module':'r004_T_full_p5_v6'})
        return {**args,'partials':('X','BW','AW','Partials'),
                'prep':(*args['prep'],'Partials')}
    if plan['module'] not in ('r004_T_full_p5_v4', 'r004_T_full_p5_v6', 'r017_T_full_p5_w1', 'r017_T_full_p5_w2', 'r017_T_full_p5_w4', 'r017_T_full_p5_w8', 'r017_T_full_p5_w16', 'r017_T_full_p5_w32', 'r017_T_full_p5_w64', 'r017_T_full_p5_split', 'r017_T_full_p5_split4', 'r017_T_full_p5_split8', 'r017_T_full_p5_split4lut'):
        return NATIVE_ARGUMENTS
    return {'prep': ('QKV','X','BW','AW','ALog','DT','QN','KN','Beta','G'),
            'core': ('QN','KN','QKV','Beta','G','StateParent','Layer','Core'),
            'y': ('Core','Z','Norm','Y')}
