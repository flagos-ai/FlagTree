# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Share type/ownership proofs within one immutable distribution search."""


class ReshardRealizationAnalysis:
    def __init__(self, policy):
        self.policy = policy
        self.results = {}

    def classify(self, context):
        if context not in self.results:
            self.results[context] = self.policy.classify(context)
        return self.results[context]

    def uses_sharded_views_for_constants(self):
        return self.policy.uses_sharded_views_for_constants()
