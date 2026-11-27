import pytest

from triton.flagmega.serving.backend import ArtifactBackend


def test_eager_abi_warmup_reset_append_sample_rewind_and_close(torch_model):
    backend = ArtifactBackend(torch_model, cuda_graph=False)
    assert backend.length == backend.state.sequence_length == 0
    assert not backend.state.kv_caches.count_nonzero()
    assert torch_model.prepares == 1
    with pytest.raises(ValueError, match="logits"):
        backend.sample()
    backend.append(3)
    assert backend.sample() == 4
    backend.append(5)
    assert backend.sample() == 7
    assert backend.length == backend.state.sequence_length == 2
    backend.rewind(1)
    with pytest.raises(ValueError, match="logits"):
        backend.sample()
    backend.append(2)
    assert backend.sample() == 4
    backend.reset()
    assert not backend.state.kv_caches.count_nonzero()
    assert backend.info()["prefill"] == "compiled_token_scan"
    backend.close()
    backend.close()
    assert torch_model.closed
    with pytest.raises(ValueError, match="closed"):
        backend.append(1)


def test_capacity_and_token_bounds_are_checked_before_execution(torch_model):
    backend = ArtifactBackend(torch_model, cuda_graph=False, warmups=0)
    for token in (-1, 16, True):
        with pytest.raises(ValueError, match="Token ID"):
            backend.append(token)
    for _ in range(backend.capacity):
        backend.append(1)
    with pytest.raises(ValueError, match="capacity"):
        backend.append(1)
    assert backend.state.sequence_length == backend.capacity


def test_layer_hidden_output_is_not_treated_as_vocabulary_logits(torch_model):
    torch_model.ir_module.metadata["vocab_size"] = 8
    with pytest.raises(ValueError, match="vocabulary logits"):
        ArtifactBackend(torch_model, cuda_graph=False)
