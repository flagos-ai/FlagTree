# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Synchronous generation with exact token-prefix cache reuse."""

from dataclasses import dataclass
import time

from .metrics import GenerationMetrics


@dataclass(frozen=True)
class GenerationResult:
    prompt_tokens: tuple[int, ...]
    token_ids: tuple[int, ...]
    finish_reason: str
    metrics: GenerationMetrics

    def to_dict(self):
        return {"prompt_tokens": list(self.prompt_tokens), "token_ids": list(self.token_ids),
                "finish_reason": self.finish_reason, **self.metrics.to_dict()}


class TextGenerationEngine:
    """One mutable sequence; callbacks run synchronously and are timed.

    ``backend.append`` consumes a token, updates state, and produces next-token
    logits. The final emitted token is not yet in KV. Prefix reuse consequently
    tracks consumed IDs, never detokenized text or an assumed chat suffix.
    """

    def __init__(self, backend, *, eos_token_ids=(), context_size=None):
        self.backend = backend
        self.eos_token_ids = frozenset(eos_token_ids)
        self.capacity = backend.capacity if context_size is None else context_size
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int) or not 0 < self.capacity <= backend.capacity:
            raise ValueError("Context size must fit the artifact cache capacity")
        if any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < backend.vocab_size
               for t in self.eos_token_ids):
            raise ValueError("EOS token is outside the artifact vocabulary")
        self._consumed = []
        self._generating = False

    @property
    def cached_tokens(self):
        return tuple(self._consumed)

    def reset(self):
        if self._generating:
            raise RuntimeError("Cannot reset a running generation")
        self.backend.reset()
        self._consumed.clear()

    def generate(self, prompt_tokens, *, max_new_tokens=128, ignore_eos=False,
                 on_token=None, started_at=None, tokenization_ms=0.0):
        if self._generating:
            raise RuntimeError("Concurrent or reentrant generation is not supported")
        started = time.perf_counter() if started_at is None else started_at
        prompt = tuple(prompt_tokens)
        if not prompt or any(isinstance(t, bool) or not isinstance(t, int)
                             or not 0 <= t < self.backend.vocab_size for t in prompt):
            raise ValueError("Prompt must contain valid integer token IDs")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if len(prompt) + max_new_tokens > self.capacity:
            raise ValueError("Prompt plus requested output exceeds context capacity; reset or reduce the request")
        if self.backend.length != len(self._consumed):
            raise RuntimeError("Backend state and cached token history disagree")
        self._generating = True
        try:
            common = 0
            for actual, requested in zip(self._consumed, prompt):
                if actual != requested:
                    break
                common += 1
            # Even an entirely cached prompt needs its final token evaluated:
            # the logits buffer may currently describe a later cached position.
            common = min(common, len(prompt) - 1)
            self.backend.rewind(common)
            del self._consumed[common:]
            self.backend.synchronize()
            prefill_started = time.perf_counter()
            for token in prompt[common:]:
                self.backend.append(token)
                self._consumed.append(token)
            self.backend.synchronize()
            prefill_ms = (time.perf_counter() - prefill_started) * 1000
            generated, intervals = [], []
            reason = "length"
            for index in range(max_new_tokens):
                if index:
                    self.backend.append(generated[-1])
                    self._consumed.append(generated[-1])
                token = self.backend.sample()
                if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < self.backend.vocab_size:
                    raise RuntimeError("Sampler returned a token outside the vocabulary")
                generated.append(token)
                if on_token is not None:
                    on_token(token)
                now = time.perf_counter()
                if index == 0:
                    ttft_ms = (now - started) * 1000
                else:
                    intervals.append((now - previous) * 1000)
                previous = now
                if not ignore_eos and token in self.eos_token_ids:
                    reason = "eos"
                    break
            metrics = GenerationMetrics(len(prompt), common, len(prompt) - common, len(generated),
                                        prefill_ms, ttft_ms, tuple(intervals),
                                        (time.perf_counter() - started) * 1000, tokenization_ms)
            return GenerationResult(prompt, tuple(generated), reason, metrics)
        except BaseException:
            # An interrupted launch may already have advanced GPU state. Do not
            # reuse a prefix whose host/device commit point is no longer known.
            self.backend.synchronize()
            self.backend.reset()
            self._consumed.clear()
            raise
        finally:
            self._generating = False

    def close(self):
        self.backend.close()
