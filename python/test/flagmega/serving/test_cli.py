import os
from pathlib import Path
import subprocess
import sys

from triton.flagmega.serving.chat_cli import create_parser


def test_cli_options_and_help_do_not_import_vllm():
    args = create_parser().parse_args(["--artifact", "artifact", "--checkpoint", "checkpoint",
                                      "--no-cuda-graph", "--temp", ".5", "--prompt", "hello"])
    assert not args.cuda_graph and args.temp == .5
    code = """
import sys
class Reject:
    def find_spec(self, name, *args):
        if name == 'vllm' or name.startswith('vllm.'):
            raise AssertionError('standalone CLI imported vLLM')
sys.meta_path.insert(0, Reject())
from triton.flagmega.serving.chat_cli import main
main(['--help'])
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--metrics-file" in result.stdout


def test_interactive_multiline_stats_reset_and_metrics_file(backend, monkeypatch, tmp_path, capsys):
    from io import StringIO
    from triton.flagmega.serving import chat_cli
    from triton.flagmega.serving.backend import ArtifactBackend
    class Tokenizer:
        chat_template = "test"
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, len(messages)]
        def decode(self, tokens, **kwargs):
            return " ".join(map(str, tokens))
    backend.load_ms = backend.prepare_ms = 0
    backend.info = lambda: {"gpu": "test device"}
    monkeypatch.setattr(ArtifactBackend, "load", lambda *args, **kwargs: backend)
    monkeypatch.setattr(chat_cli, "load_tokenizer", lambda *args: (Tokenizer(), ()))
    monkeypatch.setattr(sys, "stdin", StringIO("hello\\\nworld\n/stats\n/reset\nbye\n/exit\n"))
    output = tmp_path / "session.jsonl"
    assert chat_cli.main(["--artifact", "unused", "--checkpoint", "unused", "-n", "2",
                          "--metrics-file", str(output)]) == 0
    import json
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["event"] for row in records] == ["load", "generation", "generation"]
    assert records[-1]["cached_tokens"] == 0
    assert backend.reset_count == 1
    assert "Conversation and KV cache reset" in capsys.readouterr().err
