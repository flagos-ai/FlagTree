"""Two-stage BF16/FP16 top-32 for contiguous row-major GPU scores."""
import torch
import triton
import triton.language as tl


@triton.jit
def _key(value, index, valid):
    bits = value.to(tl.uint16, bitcast=True).to(tl.uint32)
    rank = tl.where((bits & 0x8000) != 0, bits ^ 0xffff, bits ^ 0x8000)
    fp = value.to(tl.float32)
    rank = tl.where(fp == 0.0, 0x8000, rank)
    rank = tl.where(fp != fp, 0xffff, rank)
    return tl.where(valid, (rank << 16) | (0xffff - index.to(tl.uint32)), 0)


@triton.jit
def _local(X, PART, VALUES, INDICES, N: tl.constexpr, TILE: tl.constexpr,
           TILES: tl.constexpr, DIRECT: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    column = tile * TILE + tl.arange(0, TILE)
    v = tl.load(X + row * N + column, column < N, 0)
    ranked = tl.topk(_key(v, column, column < N), 32)
    out = tl.arange(0, 32)
    if DIRECT:
        index = 0xffff - (ranked & 0xffff)
        score = tl.load(X + row * N + index)
        tl.store(VALUES + row * 32 + out, score)
        tl.store(INDICES + row * 32 + out, index.to(tl.int64))
    else:
        tl.store(PART + (row * TILES + tile) * 32 + out, ranked)


@triton.jit
def _merge(X, PART, VALUES, INDICES, N: tl.constexpr, LENGTH: tl.constexpr,
           MERGE: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, MERGE)
    ranked = tl.topk(tl.load(PART + row * LENGTH + i, i < LENGTH, 0), 32)
    index = 0xffff - (ranked & 0xffff)
    score = tl.load(X + row * N + index)
    out = tl.arange(0, 32)
    tl.store(VALUES + row * 32 + out, score)
    tl.store(INDICES + row * 32 + out, index.to(tl.int64))


class TopK32Plan:
    """Descending values, smallest column first for numeric ties including signed zero.

    NaNs rank first and keep their input bit patterns. Each call overwrites the
    owned output and scratch. Use separate plans for concurrent execution.
    """
    def __init__(self, x, *, tile=None, waves=None):
        if (x.ndim != 2 or x.device.type != "cuda" or not x.is_contiguous()
                or x.dtype not in (torch.float16, torch.bfloat16)):
            raise ValueError("expected contiguous GPU [rows, width] FP16/BF16")
        self.rows, self.width = x.shape
        if self.rows < 1 or not 32 <= self.width <= 65535:
            raise ValueError("expected positive rows and width in [32,65535]")
        if tile is None:
            tile = 1024 if self.width >= 20480 else 256
        if waves is None:
            waves = 2 if tile == 1024 and self.rows >= 28 else 4
        if tile not in (128, 256, 512, 1024) or waves not in (2, 4, 8):
            raise ValueError("unsupported selection tile or wave count")
        self.x = x
        self.tile, self.waves = tile, waves
        self.tiles = triton.cdiv(self.width, self.tile)
        self.partial = torch.empty((self.rows, self.tiles, 32), device=x.device, dtype=torch.uint32)
        self.values = torch.empty((self.rows, 32), device=x.device, dtype=x.dtype)
        self.indices = torch.empty((self.rows, 32), device=x.device, dtype=torch.int64)
        self.last_local = self.last_merge = None

    def __call__(self):
        self.last_local = _local[(self.rows, self.tiles)](
            self.x, self.partial, self.values, self.indices, self.width,
            self.tile, self.tiles, self.tiles == 1, num_warps=self.waves, num_stages=1)
        if self.tiles > 1:
            size = self.tiles * 32
            self.last_merge = _merge[(self.rows,)](
                self.x, self.partial, self.values, self.indices, self.width, size,
                triton.next_power_of_2(size), num_warps=4, num_stages=1)
        return self.values, self.indices
