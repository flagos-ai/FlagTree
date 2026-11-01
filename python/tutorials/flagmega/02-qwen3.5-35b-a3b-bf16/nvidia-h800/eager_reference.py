"""Untimed greedy requests evaluated from imported IR with PyTorch operators."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from triton.flagmega import ir as fm
from triton.flagmega.evaluator import CheckpointWeightResolver, TorchEvaluator, create_gdn_state, create_paged_attention_state
from triton.flagmega.importer import DirectoryCheckpoint, import_model
from triton.flagmega.ir.ops.nn._gdn_state import GatedDeltaNetStateConfig
from triton.flagmega.ir.ops.nn._paged_attention_state import paged_attention_state_config_from_type


class TorchIRBackend:
    def __init__(self, checkpoint, *, device="cuda:0", numerical_profile="vllm-ae10e855a-inductor-level3"):
        self.checkpoint = DirectoryCheckpoint(checkpoint)
        self.device = device
        self.numerical_profile = numerical_profile
        self.decoder = self._import(None, 1)
        self.prefills = {}
        self.evaluator = TorchEvaluator(CheckpointWeightResolver(self.checkpoint, device=device))
        cfg = self.checkpoint.config["text_config"]
        if cfg["num_hidden_layers"] != 40:
            raise ValueError("The eager reference requires the complete 40-layer model")
        self.gdn = create_gdn_state(GatedDeltaNetStateConfig(
            num_layers=cfg["layer_types"].count("linear_attention"),
            num_key_heads=cfg["linear_num_key_heads"], num_value_heads=cfg["linear_num_value_heads"],
            key_head_dim=cfg["linear_key_head_dim"], value_head_dim=cfg["linear_value_head_dim"],
            conv_kernel_size=cfg["linear_conv_kernel_dim"], hidden_size=cfg["hidden_size"]), device=device)
        self.paged = create_paged_attention_state(
            paged_attention_state_config_from_type(self.decoder.node_map["paged_state"].type), device=device)
        self.length = 0

    def _import(self, mode, tokens):
        options = {} if mode is None else {"mode": mode, "num_tokens": tokens}
        return import_model(self.checkpoint, numerical_profile=self.numerical_profile, **options)

    def reset(self):
        self.gdn.convolution.zero_()
        self.gdn.recurrent.zero_()
        self.paged.kv_caches.zero_()
        self.paged.seq_lens.zero_()
        self.paged.slot_mapping.zero_()
        self.length = 0

    def forward(self, tokens, *, prefill=False):
        tokens = tuple(tokens)
        if prefill:
            if self.length:
                raise ValueError("The eager reference requires fresh prefill state")
            if len(tokens) not in self.prefills:
                self.prefills[len(tokens)] = self._import("prefill", len(tokens))
            module = self.prefills[len(tokens)]
        else:
            if len(tokens) != 1:
                raise ValueError("Decode consumes exactly one independently generated token")
            module = self.decoder
        self.paged.query_start_loc.copy_(torch.tensor([0, len(tokens)], dtype=torch.int32, device=self.device))
        self.paged.slot_mapping.fill_(self.length)
        with torch.inference_mode():
            output = self.evaluator.run(module, {
                "input_ids": torch.tensor(tokens, dtype=torch.int32, device=self.device),
                "gated_delta_net_state": self.gdn, "paged_attention_state": self.paged,
            })
        self.length += len(tokens)
        if int(self.paged.seq_lens.item()) != self.length:
            raise ValueError("Eager state must advance by the actual query length")
        if output[1].numel() != 1 or not bool(torch.isfinite(output[0]).all()):
            raise ValueError("Eager execution returned nonfinite logits or an invalid token shape")
        return int(output[1].item())

    def request(self, prompt, count=64):
        self.reset()
        token = self.forward(prompt, prefill=True)
        generated = [token]
        for _ in range(count - 1):
            token = self.forward((token,))
            generated.append(token)
        return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 1024, 2048])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--numerical-profile", choices=("nncase", "vllm-ae10e855a-inductor-level3"),
                        default="vllm-ae10e855a-inductor-level3")
    args = parser.parse_args()
    if args.output.exists() or args.tokens < 1:
        parser.error("Use a new output file and a positive token count")
    from transformers import AutoTokenizer
    from benchmark_requests import PROMPT
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    source = tokenizer.encode(PROMPT, add_special_tokens=False)
    backend = TorchIRBackend(args.checkpoint, numerical_profile=args.numerical_profile)
    scenarios = []
    for length in args.prompt_lengths:
        prompt = (source * ((length + len(source) - 1) // len(source)))[:length]
        tokens = backend.request(prompt, args.tokens)
        scenarios.append({"prompt_length": length, "decode_tokens": args.tokens,
                          "runs": [{"prompt_tokens": prompt, "token_ids": tokens}]})
        print(json.dumps({"prompt_length": length, "token_ids": tokens}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "flagmega.ir-eager-request-reference/v1", "performance_valid": False,
              "eager_reference_kind": "ir_torch", "torch": torch.__version__,
              "numerical_contract": args.numerical_profile,
              "checkpoint_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
              "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
              "reference_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "imported_semantic_hashes": {"decode": backend.decoder.semantic_hash,
                                          **{f"prefill-{n}": m.semantic_hash for n, m in backend.prefills.items()}},
              "scenarios": scenarios}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
