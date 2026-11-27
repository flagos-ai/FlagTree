"""Device tests of the serving graph/state ABI, independent of a large model."""

import pytest

from triton.flagmega.serving.backend import ArtifactBackend
from triton.flagmega.serving.engine import TextGenerationEngine


@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "graph"])
def test_device_graph_advances_state_and_reuses_prefix(torch_model, cuda_graph):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
    device = "cuda:0"
    torch_model.device = device
    torch_model.create_state = lambda: create_paged_attention_state(torch_model.state_config, device=device)
    builtin = torch_model.result_kind == "logits_token_state"
    def create_outputs():
        logits = torch.empty((1, 16), device=device)
        return (logits, torch.empty((1,), dtype=torch.int32, device=device)) if builtin else logits
    torch_model.create_outputs = create_outputs
    def run_into(logits, *args):
        token, inputs, state = args if builtin else (None, *args)
        # Tensor-only state transitions are captured without a host .item().
        state.kv_caches.flatten().scatter_(0, state.seq_lens.long(), inputs.to(torch.bfloat16))
        state.seq_lens.add_(1)
        logits.zero_().scatter_(1, ((inputs + state.seq_lens) % 16).long().reshape(1, 1), 1)
        if token is not None:
            token.copy_(logits.argmax(-1))
    torch_model.run_into = run_into
    backend = ArtifactBackend(torch_model, cuda_graph=cuda_graph)
    try:
        engine = TextGenerationEngine(backend)
        first = engine.generate([1, 2], max_new_tokens=3)
        assert first.token_ids == (4, 7, 11)
        assert backend.state.sequence_length == 4
        reused = engine.generate([1, 2, 4, 7, 3], max_new_tokens=2)
        assert reused.metrics.cached_tokens == 4
        engine.reset()
        fresh = engine.generate([1, 2, 4, 7, 3], max_new_tokens=2)
        assert reused.token_ids == fresh.token_ids
    finally:
        backend.close()
