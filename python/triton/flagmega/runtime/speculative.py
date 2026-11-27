# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Greedy prefix acceptance for transactional batched verification backends.

The target must return one greedy result per input row and retain a commit
point for every row, including all recurrent and convolution state. A token
scan backend cannot satisfy this protocol by advertising KV truncation alone.
"""
from dataclasses import dataclass
from triton.flagmega.errors import RuntimeContractError


@dataclass(frozen=True)
class SpeculativeStep:
    tokens: tuple[int, ...]
    accepted: int
    drafted: int
    verified_rows: int


def longest_greedy_prefix(drafted, verified):
    """verified[i] predicts the next token after input row i."""
    if len(verified) != len(drafted) + 1:
        raise RuntimeContractError('Verification requires k+1 greedy output rows.')
    for i, token in enumerate(drafted):
        if token != verified[i]:
            return i
    return len(drafted)


def speculative_step(target, draft, pending_token, *, k=1):
    """Consume one pending token, return an accepted prefix plus correction.

Proposal and verification are transactions. target.verify_many must perform
one multi-row forward; commit_rows(n) keeps exactly its first n input rows.
The returned correction/bonus is pending and has not yet entered target state.
Draft synchronization uses accepted main hidden states, not rejected features.
"""
    if type(k) is not int or not 1 <= k <= 4:
        raise RuntimeContractError('MTP draft k must be an integer from one to four.')
    if type(pending_token) is not int or pending_token < 0:
        raise RuntimeContractError('Pending token must be a nonnegative integer.')
    if not getattr(target, 'supports_batched_prefix_commit', False):
        raise RuntimeContractError('Speculative decode requires one batched verify and prefix commit '
                                   'of KV, recurrent state, and convolution history; decode-1 is insufficient.')
    proposal = draft.propose(pending_token, k)
    verification = None
    try:
        candidates = tuple(proposal.tokens)
        if len(candidates) != k or any(type(t) is not int or t < 0 for t in candidates):
            raise RuntimeContractError('Draft transaction must return exactly k nonnegative token IDs.')
        verification = target.verify_many((pending_token, *candidates))
        greedy = tuple(verification.greedy_tokens)
        if any(type(t) is not int or t < 0 for t in greedy):
            raise RuntimeContractError('Verifier must return nonnegative greedy token IDs.')
        accepted = longest_greedy_prefix(candidates, greedy)
        output = (*candidates[:accepted], greedy[accepted])
        # Prepare draft synchronization before publishing either state.
        proposal.prepare_commit(verification, accepted + 1, output[-1])
        verification.commit_rows(accepted + 1)
        proposal.commit()
        return SpeculativeStep(output, accepted, k, k + 1)
    finally:
        if verification is not None:
            verification.close()
        proposal.close()
