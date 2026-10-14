"""Download the pinned public pure-IQ4_XS checkpoint without converting weights."""

import argparse
import hashlib
import json
from pathlib import Path

REPO_ID = "jpetrina/Qwen3.8-27B-IQ4_XS-pure-GGUF"
REVISION = "5a285a6217024a96be7064d6caf0a06121630ee1"
FILENAME = "qwen3.8-27b-IQ4_XS-pure.gguf"
SHA256 = "ea5a3c45d407f9b9e5d2c0d647f0ea600f486f6b86b92b56d0823ba073dae675"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / ".local/checkpoint",
                        help="Checkpoint directory; default: .local/checkpoint")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO_ID, FILENAME, revision=REVISION, local_dir=args.output))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise SystemExit(f"Checkpoint checksum mismatch: {path}")
    print(
        json.dumps({"checkpoint": str(path.resolve()), "repo_id": REPO_ID, "revision": REVISION, "sha256": SHA256},
                   indent=2))


if __name__ == "__main__":
    main()
