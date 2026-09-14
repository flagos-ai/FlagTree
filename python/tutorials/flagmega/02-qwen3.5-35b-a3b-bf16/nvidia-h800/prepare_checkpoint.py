"""Download the immutable BF16 checkpoint used by this tutorial."""

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3.5-35B-A3B"
REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
METADATA = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / ".local/checkpoint")
    parser.add_argument("--metadata-only", action="store_true", help="Inspect configuration/index without weights")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    from huggingface_hub import snapshot_download

    directory = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            local_dir=args.output,
            allow_patterns=[*METADATA, *([] if args.metadata_only else ["*.safetensors"])],
            max_workers=args.max_workers,
        ))
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "qwen3_5_moe" or config["text_config"].get("dtype") != "bfloat16":
        raise ValueError("Downloaded checkpoint does not match the BF16 MoE contract")
    print(
        json.dumps({
            "checkpoint": str(directory.resolve()), "model": MODEL, "revision": REVISION, "metadata_only":
            args.metadata_only
        }), flush=True)


if __name__ == "__main__":
    main()
