"""Validate the Qwen3.8 IQ4_XS GGUF used by this tutorial.

The file is intentionally kept in GGUF form.  This command only reads metadata;
it never dequantizes or rewrites the model weights.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from download_checkpoint import FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / ".local/checkpoint" / FILENAME,
                        help="Qwen3.8-27B IQ4_XS GGUF path; default: .local/checkpoint")
    args = parser.parse_args()
    checkpoint = args.input.expanduser().resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".gguf":
        parser.error(f"GGUF checkpoint does not exist: {checkpoint}")

    try:
        from gguf import GGUFReader
    except ImportError as error:  # pragma: no cover - exercised in the tutorial environment
        raise SystemExit("Install the tutorial dependency first: python -m pip install -r requirements.txt") from error
    reader = GGUFReader(checkpoint)
    architecture = reader.get_field("general.architecture")
    if architecture is None or architecture.contents() != "qwen35":
        raise SystemExit("This tutorial requires a qwen35 architecture GGUF")
    types = Counter(str(tensor.tensor_type.name) for tensor in reader.tensors)
    layer_types = Counter(str(tensor.tensor_type.name) for tensor in reader.tensors if tensor.name.startswith("blk.0."))
    if not {"IQ4_XS", "Q4_K"}.intersection(types):
        raise SystemExit(f"No IQ4_XS/Q4_K tensors found in {checkpoint}")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "architecture": str(architecture.contents()),
                "tensor_count": len(reader.tensors),
                "storage_types": dict(sorted(types.items())),
                "layer_0_storage_types": dict(sorted(layer_types.items())),
                "native_packed_weights": True,
                "host_dequantization": False,
            }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
