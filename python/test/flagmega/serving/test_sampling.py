import pytest

from triton.flagmega.serving.sampling import SamplingConfig, sample_logits


@pytest.mark.parametrize("kwargs", [{"temperature": -1}, {"temperature": float("nan")},
                                    {"top_p": 0}, {"top_p": 2}, {"top_k": -1}, {"top_k": 1.5}])
def test_rejects_invalid_sampling_contract(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


def test_greedy_tie_break_top_k_top_p_and_seed():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2., 3., 3., -10.]])
    assert sample_logits(logits, SamplingConfig()).item() == 1
    logits = torch.tensor([[0., 10., 1.]])
    assert sample_logits(logits, SamplingConfig(temperature=1, top_k=1)).item() == 1
    assert sample_logits(logits, SamplingConfig(temperature=1, top_p=.1)).item() == 1
    def draw(seed):
        gen = torch.Generator().manual_seed(seed)
        return [sample_logits(torch.zeros((1, 8)), SamplingConfig(temperature=1), generator=gen).item()
                for _ in range(20)]
    assert draw(42) == draw(42)
