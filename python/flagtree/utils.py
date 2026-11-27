# Copyright 2026 FlagOS Contributors
import os


def get_local_world_size() -> int:
    value = os.environ.get("LOCAL_WORLD_SIZE")
    if value is None:
        raise RuntimeError("LOCAL_WORLD_SIZE must be set for distributed autotuning, for example by torchrun")
    try:
        local_world_size = int(value)
    except ValueError as exc:
        raise ValueError("LOCAL_WORLD_SIZE must be a positive integer") from exc
    if local_world_size <= 0:
        raise ValueError("LOCAL_WORLD_SIZE must be a positive integer")
    return local_world_size
