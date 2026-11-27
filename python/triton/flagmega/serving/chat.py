# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Tokenizer-owned chat formatting and incremental text display."""

import json
from pathlib import Path
import time


def load_tokenizer(checkpoint, backend):
    from transformers import AutoTokenizer
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    config = config.get("text_config", config)
    if config.get("vocab_size") != backend.vocab_size:
        raise ValueError("Tokenizer checkpoint and artifact vocabulary sizes differ")
    if config.get("num_hidden_layers") != backend.model.state_config.num_layers:
        raise ValueError("Text serving requires the complete checkpoint, not a layer-only artifact")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    generation = checkpoint / "generation_config.json"
    generation = json.loads(generation.read_text()) if generation.exists() else {}
    eos = generation.get("eos_token_id", config.get("eos_token_id", tokenizer.eos_token_id))
    eos = () if eos is None else ((eos,) if isinstance(eos, int) else tuple(eos))
    return tokenizer, eos


class TextStream:
    """Commit complete words/lines; never emit incomplete UTF-8 byte tokens."""

    def __init__(self, tokenizer, write):
        self.tokenizer = tokenizer
        self.write = write
        self.tokens = []
        self.emitted = 0
        self.text = ""

    def put(self, token):
        self.tokens.append(token)
        self.text = self.tokenizer.decode(self.tokens, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        boundary = max(self.text.rfind(" "), self.text.rfind("\n")) + 1
        # CJK text does not generally delimit words with spaces.
        if self.text and ("\u4e00" <= self.text[-1] <= "\u9fff"):
            boundary = len(self.text)
        if boundary > self.emitted:
            self.write(self.text[self.emitted:boundary])
            self.emitted = boundary

    def finish(self):
        if self.emitted < len(self.text):
            self.write(self.text[self.emitted:])
            self.emitted = len(self.text)


class ChatSession:
    def __init__(self, engine, tokenizer, *, system_prompt=None, template_kwargs=None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        if {"tokenize", "add_generation_prompt", "conversation"} & self.template_kwargs.keys():
            raise ValueError("Template kwargs cannot override tokenization or conversation ownership")
        self._initial = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
        self.messages = list(self._initial)

    def reset(self):
        self.engine.reset()
        self.messages = list(self._initial)

    def chat(self, text, *, max_new_tokens=128, write=lambda text: None):
        started = time.perf_counter()
        messages = [*self.messages, {"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    **self.template_kwargs)
        tokenization_ms = (time.perf_counter() - started) * 1000
        stream = TextStream(self.tokenizer, write)
        result = self.engine.generate(prompt, max_new_tokens=max_new_tokens, on_token=stream.put,
                                      started_at=started, tokenization_ms=tokenization_ms)
        stream.finish()
        # Commit only a completed turn. The next template is compared by exact
        # token prefix, including any changed assistant suffix or special tokens.
        self.messages = [*messages, {"role": "assistant", "content": stream.text}]
        return result, stream.text
