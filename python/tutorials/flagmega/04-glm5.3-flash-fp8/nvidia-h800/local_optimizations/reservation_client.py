"""Keep the required GPU memory reserved during CPU-only artifact loading."""
import json
import os
from pathlib import Path
import time

from triton.flagmega.runtime import load


def load_reserved(artifact, *, device):
    directory = Path(os.environ['FLAGMEGA_GPU_RESERVATION_DIR'])
    rank, pid = int(os.environ['LOCAL_RANK']), os.getpid()
    runtime = load(artifact)  # Public CPU-only construction/verification.
    ready = directory/f'ready-{rank}-{pid}.json'
    temporary = ready.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(rank=rank,pid=pid,artifact=str(Path(artifact).resolve()),
                                          uuid=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank]))+'\n')
    temporary.replace(ready)
    released = directory/f'released-{rank}-{pid}'
    start = time.monotonic()
    while not released.is_file():
        if time.monotonic()-start > 120:
            raise RuntimeError('GPU reservation handoff timed out')
        time.sleep(.05)
    # No IR load or compilation is hidden here: the ordinary runtime now
    # allocates its complete original symmetric window and readonly data.
    return runtime.load(device)
