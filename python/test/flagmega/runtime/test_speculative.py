"""Transaction tests with history-dependent state and rejected suffix writes."""
import pytest
from triton.flagmega.runtime.speculative import longest_greedy_prefix, speculative_step
from triton.flagmega.errors import RuntimeContractError


class Target:
    supports_batched_prefix_commit=True
    def __init__(self,seed): self.state=seed;self.calls=0
    def verify_many(self,tokens):
        self.calls+=1
        outer=self
        states=[];greedy=[];s=self.state
        for token in tokens:
            s=(s*13+token*7+3)%1009;states.append(s);greedy.append(s%31)
        class Transaction:
            greedy_tokens=tuple(greedy)
            def commit_rows(self,n): outer.state=states[n-1]
            def close(self): pass
        return Transaction()


class Draft:
    def __init__(self,target,match): self.target=target;self.match=match;self.committed=False
    def propose(self,pending,k):
        s=self.target.state;token=pending;candidates=[]
        for i in range(k):
            s=(s*13+token*7+3)%1009
            token=(s%31 if i<self.match else (s%31+1)%31)
            candidates.append(token)
        outer=self
        class Proposal:
            tokens=tuple(candidates)
            def prepare_commit(self,verification,n,next_token): outer.pending=next_token
            def commit(self): outer.committed=True
            def close(self): pass
        return Proposal()


@pytest.mark.parametrize('seed',[20261120,1,37])
@pytest.mark.parametrize('k',[1,2,3,4])
@pytest.mark.parametrize('match',[0,1,2,3,4])
def test_greedy_prefix_and_state_commit(seed,k,match):
    target=Target(seed);draft=Draft(target,match);plain=Target(seed)
    result=speculative_step(target,draft,11,k=k)
    expected=[];pending=11
    for _ in result.tokens:
        transaction=plain.verify_many((pending,));pending=transaction.greedy_tokens[0]
        expected.append(pending);transaction.commit_rows(1)
    assert tuple(expected)==result.tokens
    assert target.state==plain.state and target.calls==1
    assert result.accepted==min(match,k)
    assert draft.committed and draft.pending==result.tokens[-1]


def test_decode_one_backend_cannot_silently_emulate_verification():
    with pytest.raises(RuntimeContractError,match='batched verify'):
        speculative_step(object(),object(),11,k=2)


@pytest.mark.parametrize('k',[0,5,True,1.5])
def test_bad_k_rejected(k):
    with pytest.raises(RuntimeContractError,match='draft k'): speculative_step(object(),object(),1,k=k)


def test_missing_bonus_row_rejected():
    with pytest.raises(RuntimeContractError,match='k\\+1'): longest_greedy_prefix((1,2),(1,2))
