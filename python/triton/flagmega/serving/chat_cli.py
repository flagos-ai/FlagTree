# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Standalone interactive text generation from a FlagMega executable artifact.

Prompt/system messages, chat templates and per-turn eval timings follow the
interaction style of llama.cpp's CLI. No vLLM or model-forward framework is
used: prefill is an explicit causal scan of the compiled single-token entry.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

from .chat import ChatSession, TextStream, load_tokenizer
from .engine import TextGenerationEngine
from .sampling import SamplingConfig


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local tokenizer/config checkpoint; weights come from artifact")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-c", "--ctx-size", type=int)
    parser.add_argument("-n", "--n-predict", type=int, default=128)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("-p", "--prompt")
    prompts.add_argument("-f", "--prompt-file", type=Path)
    prompts.add_argument("--prompt-token-file", type=Path, help="JSON list of raw token IDs, without a chat template")
    parser.add_argument("--raw", action="store_true", help="Completion without a chat template; requires a prompt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-i", "--interactive", action="store_true", help="Continue chatting after an initial prompt")
    mode.add_argument("--single-turn", action="store_true", help="Stop after the first interactive response")
    parser.add_argument("-sys", "--system-prompt")
    parser.add_argument("--chat-template-kwargs", default="{}", help="JSON object forwarded to the checkpoint's chat template")
    parser.add_argument("--temp", type=float, default=0.0, help="Zero selects greedy decoding")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true", help="Raw completion only, useful for fixed-length measurements")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--show-timings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-file", type=Path, help="New JSONL file containing load information and per-turn metrics/tokens")
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    sampling = SamplingConfig(args.temp, args.top_k, args.top_p, args.seed)
    if args.n_predict < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("n-predict/threads must be positive and warmups nonnegative")
    kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(kwargs, dict):
        parser.error("chat-template-kwargs must be a JSON object")
    prompt = args.prompt_file.read_text() if args.prompt_file is not None else args.prompt
    raw = args.raw or args.prompt_token_file is not None
    if raw and (prompt is None and args.prompt_token_file is None):
        parser.error("Raw completion requires a prompt")
    if raw and (args.interactive or args.system_prompt is not None):
        parser.error("Raw completion cannot use interactive mode or a system prompt")
    if args.ignore_eos and not raw:
        parser.error("ignore-eos is for raw completion, not interactive chat")
    if args.metrics_file is not None and args.metrics_file.exists():
        parser.error("Choose a new metrics file; existing measurements are not overwritten")

    import torch
    from .backend import ArtifactBackend
    torch.set_num_threads(args.threads)
    backend = ArtifactBackend.load(args.artifact, device=args.device, sampling=sampling,
                                   cuda_graph=args.cuda_graph, warmups=args.warmups)
    output = None
    try:
        tokenizer, eos = load_tokenizer(args.checkpoint, backend)
        engine = TextGenerationEngine(backend, eos_token_ids=eos, context_size=args.ctx_size)
        chat = ChatSession(engine, tokenizer, system_prompt=args.system_prompt, template_kwargs=kwargs)
        if not raw and not tokenizer.chat_template:
            parser.error("Checkpoint has no chat template; provide a template in its tokenizer config or use --raw")
        info = {**backend.info(), "sampling": asdict(sampling), "context_size": engine.capacity,
                "checkpoint": str(args.checkpoint.resolve()),
                "timing_boundary": "synchronous token generation, sampling, host delivery and text callback; no human input wait",
                "schema": "flagmega.chat/v1"}
        if args.metrics_file:
            args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            output = args.metrics_file.open("x")
            output.write(json.dumps({"event": "load", **info}) + "\n")
            output.flush()
        print(f"FlagMega | {info['gpu']} | context {engine.capacity} | prefill: compiled token scan\n"
              f"load {backend.load_ms:.2f} ms; prepare/JIT/graph {backend.prepare_ms:.2f} ms",
              file=sys.stderr, flush=True)
        last = None
        turn = 0

        def write(text):
            print(text, end="", flush=True)

        def record(result, text):
            nonlocal last, turn
            last = result.metrics
            turn += 1
            print()
            if args.show_timings:
                print(last.format(), file=sys.stderr, flush=True)
            if output is not None:
                output.write(json.dumps({"event": "generation", "turn": turn, "text": text,
                                         **result.to_dict()}, ensure_ascii=False) + "\n")
                output.flush()

        if raw:
            started = time.perf_counter()
            tokens = (json.loads(args.prompt_token_file.read_text()) if args.prompt_token_file is not None
                      else tokenizer.encode(prompt, add_special_tokens=False))
            tokenization_ms = (time.perf_counter() - started) * 1000
            stream = TextStream(tokenizer, write)
            result = engine.generate(tokens, max_new_tokens=args.n_predict, ignore_eos=args.ignore_eos,
                                     on_token=stream.put, started_at=started, tokenization_ms=tokenization_ms)
            stream.finish()
            record(result, stream.text)
            return 0
        if prompt is not None:
            result, text = chat.chat(prompt, max_new_tokens=args.n_predict, write=write)
            record(result, text)
            if not args.interactive:
                return 0
        print("Commands: /reset, /stats, /help, /exit. End a line with \\ for multiline input.",
              file=sys.stderr)
        while True:
            try:
                text = input("\n> ")
                while text.endswith("\\"):
                    text = text[:-1] + "\n" + input("... ")
                if text in {"/exit", "/quit"}:
                    break
                if text in {"/reset", "/clear"}:
                    chat.reset()
                    print("Conversation and KV cache reset.", file=sys.stderr)
                    continue
                if text == "/stats":
                    print(last.format() if last else "No completed turns.", file=sys.stderr)
                    continue
                if text == "/help":
                    print("/reset clears history; /stats shows the last turn; /exit quits. Ctrl-C cancels a turn.", file=sys.stderr)
                    continue
                if not text.strip():
                    continue
                result, answer = chat.chat(text, max_new_tokens=args.n_predict, write=write)
                record(result, answer)
                if args.single_turn:
                    break
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nInterrupted; unfinished turn was not committed.", file=sys.stderr)
            except ValueError as error:
                print(f"Request rejected: {error}", file=sys.stderr)
        return 0
    finally:
        if output is not None:
            output.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
