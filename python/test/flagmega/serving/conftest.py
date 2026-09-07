from types import SimpleNamespace

import pytest


class FakeBackend:
    capacity = 64
    vocab_size = 32

    def __init__(self):
        self.tokens = []
        self.appends = []
        self.reset_count = 0

    @property
    def length(self):
        return len(self.tokens)

    def append(self, token):
        self.tokens.append(token)
        self.appends.append(token)

    def sample(self):
        return sum(self.tokens) % self.vocab_size

    def rewind(self, length):
        del self.tokens[length:]

    def reset(self):
        self.tokens.clear()
        self.reset_count += 1

    def synchronize(self):
        pass

    def close(self):
        pass


@pytest.fixture
def backend():
    return FakeBackend()


class TorchModel:
    def __init__(self, *, builtin, artifact):
        import torch
        from triton.flagmega.ir.ops.nn._paged_attention_state import PagedAttentionStateConfig
        self.device = "cpu"
        self.result_kind = "logits_token_state" if builtin else "tensor_state"
        self.state_config = PagedAttentionStateConfig(1, 1, 8, block_size=4, num_blocks=2)
        self.ir_module = SimpleNamespace(metadata={"vocab_size": 16})
        self.artifact = artifact
        self.manifest = {"semantic_hash": "test"}
        self.resource_report = None
        self.prepares = 0
        self.closed = False
        self.torch = torch

    def create_state(self):
        from triton.flagmega.ir.ops.nn._paged_attention_state import create_paged_attention_state
        return create_paged_attention_state(self.state_config)

    def create_outputs(self):
        logits = self.torch.empty((1, 16))
        return (logits, self.torch.empty((1,), dtype=self.torch.int32)) if self.result_kind == "logits_token_state" else logits

    def prepare(self, *args, **kwargs):
        self.prepares += 1

    def run_into(self, logits, *args):
        token, inputs, state = args if len(args) == 3 else (None, *args)
        state.kv_caches.flatten()[state.sequence_length] = inputs[0]
        state.seq_lens.add_(1)
        logits.zero_().scatter_(1, ((inputs + state.seq_lens) % 16).long().reshape(1, 1), 1)
        if token is not None:
            token.copy_(logits.argmax(-1))

    def close(self):
        self.closed = True


@pytest.fixture(params=[False, True], ids=["logits", "builtin-sampler"])
def torch_model(request, tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "generated_kernels.py").write_text("# fixture\n")
    return TorchModel(builtin=request.param, artifact=tmp_path)
