"""Physical layout bridge, independently testable without importing vLLM."""

from contextlib import contextmanager


@contextmanager
def preserve_cache_slot(slot):
    """A diagnostic reference must not supply a missing write to the candidate."""
    original = slot.clone()
    try:
        yield
    finally:
        slot.copy_(original)


def semantic_layer_views(storage, config):
    """Zero-copy native [KV, page, token, head, dim] views of packed storage."""
    if tuple(storage.shape) != config.storage_shape or not storage.is_contiguous():
        raise ValueError("FlagMega KV storage must match its contiguous packed ABI")
    flat = storage.view(config.num_blocks, config.num_layers, 2,
                        config.block_size, config.num_kv_heads, config.head_dim)
    return [flat[:, layer].permute(1, 0, 2, 3, 4) for layer in range(config.num_layers)]
