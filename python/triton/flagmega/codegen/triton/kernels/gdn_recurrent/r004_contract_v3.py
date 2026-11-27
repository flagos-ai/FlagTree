# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Native positional names pinned by r004 v3 §2.9 / handshake revision 3."""
NATIVE_ARGUMENTS = {
    "pb": ("X", "BW", "B"), "pa": ("X", "AW", "A"),
    "beta": ("B", "Beta"), "bias": ("A", "DT", "Biased"),
    "gate": ("Biased", "ALog", "G"), "q": ("QKV", "QN"), "k": ("QKV", "KN"),
    "core": ("QN", "KN", "QKV", "Beta", "G", "StateParent", "Layer", "Core"),
    "rms": ("Core", "Norm", "Weighted"), "y": ("Weighted", "Z", "Y"),
}
