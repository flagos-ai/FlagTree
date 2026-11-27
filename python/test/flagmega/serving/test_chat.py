import pytest

from triton.flagmega.serving.chat import ChatSession, TextStream
from triton.flagmega.serving.engine import TextGenerationEngine


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [1, 2, len(messages)]

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


def test_chat_uses_template_and_commits_only_completed_turns(backend):
    tokenizer = Tokenizer()
    session = ChatSession(TextGenerationEngine(backend), tokenizer, system_prompt="system",
                          template_kwargs={"custom": False})
    result, text = session.chat("hello", max_new_tokens=2)
    assert tokenizer.kwargs == {"tokenize": True, "add_generation_prompt": True, "custom": False}
    assert session.messages[-1] == {"role": "assistant", "content": text}
    assert session.messages[0]["role"] == "system"
    before = list(session.messages)
    with pytest.raises(ValueError):
        session.chat("overflow", max_new_tokens=64)
    assert session.messages == before
    session.reset()
    assert session.messages == [{"role": "system", "content": "system"}]
    assert backend.length == 0


def test_stream_holds_partial_bytes_and_flushes_tail():
    class ByteTokenizer:
        def decode(self, tokens, **kwargs):
            return {1: "\ufffd", 2: "é", 3: "é ", 4: "é hi"}[len(tokens)]
    output = []
    stream = TextStream(ByteTokenizer(), output.append)
    stream.put(1)
    stream.put(2)
    assert output == []
    stream.put(3)
    stream.put(4)
    stream.finish()
    assert "".join(output) == "é hi"
