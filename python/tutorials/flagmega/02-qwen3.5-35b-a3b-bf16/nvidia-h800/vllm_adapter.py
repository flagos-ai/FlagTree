"""Out-of-tree vLLM model plugin executing a single FlagMega weight pool.

vLLM owns request scheduling and sampling. Decode can continue a verified
native prefill snapshot, or use the complete compiled phase bundle.
"""

import os

import torch
from torch import nn

from phase_backend import PhaseBundleBackend
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention


class FlagMegaModel(nn.Module):
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError("The BF16 request benchmark supports text inputs only")
        return torch.arange(len(input_tokens), dtype=torch.int64).expand(3, -1), 0

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        if (vllm_config.scheduler_config.max_num_seqs != 1
                or vllm_config.parallel_config.tensor_parallel_size != 1
                or vllm_config.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("This tutorial supports batch/concurrency/TP/PP one")
        self.native_prefill = os.environ.get("FLAGMEGA_NATIVE_PREFILL_REFERENCE")
        if self.native_prefill:
            from native_prefill_backend import NativePrefillBackend
            self.backend = NativePrefillBackend(os.environ["FLAGMEGA_REQUEST_ARTIFACT"], self.native_prefill,
                                               os.environ["FLAGMEGA_NATIVE_PREFILL_CHECKPOINT"])
        else:
            self.backend = PhaseBundleBackend(os.environ["FLAGMEGA_REQUEST_ARTIFACT"])
        config = vllm_config.model_config.hf_config.get_text_config()
        # Expose the attention geometry to the scheduler's page allocator.
        # Compiled state is private; these layers never execute a model forward.
        self.attention = nn.ModuleList([
            Attention(config.num_attention_heads, config.head_dim, config.head_dim ** -.5,
                      num_kv_heads=config.num_key_value_heads, cache_config=vllm_config.cache_config,
                      prefix=f"{prefix}model.layers.{index}.self_attn.attn")
            for index, kind in enumerate(config.layer_types) if kind == "full_attention"
        ])

    def load_weights(self, weights):
        # Artifact loading verifies and loads the packed checkpoint weight pool.
        return set()

    def embed_input_ids(self, input_ids):
        raise ValueError("The compiled artifact accepts token IDs, not embeddings")

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None or intermediate_tensors is not None:
            raise ValueError("Only raw-token, single-stage requests are supported")
        context = get_forward_context()
        if context.attn_metadata is None:
            # vLLM's memory profile is not a request and carries dummy positions.
            self.backend.reset()
            self.backend.append(0)
        else:
            tokens = input_ids.tolist()
            position_ids = (positions[0] if positions.ndim == 2 else positions).tolist()
            if self.native_prefill:
                if len(tokens) != 1 or position_ids != [self.backend.length - self.backend.native_prefix_length]:
                    raise ValueError("Continuation scheduler positions differ from the native prefix history")
                self.backend.append(int(tokens[0]))
                return self.backend.logits
            if position_ids[0] == 0:
                self.backend.reset()
            if position_ids != list(range(self.backend.length, self.backend.length + len(tokens))):
                raise ValueError("Scheduler positions differ from the artifact's complete state history")
            if self.backend.length == 0 and len(tokens) > 1:
                self.backend.prefill(tuple(int(t) for t in tokens))
            elif len(tokens) == 1:
                self.backend.append(int(tokens[0]))
            else:
                raise ValueError("The aligned benchmark requires one complete prefill and single-token decode")
        # vLLM selects the last scheduled token before invoking compute_logits.
        # All actual model arithmetic, including LM head, ran in the artifact.
        return self.backend.logits.expand(input_ids.numel(), -1)

    def compute_logits(self, hidden_states):
        return hidden_states
