"""Execute the compiled native IQ4_XS/Q4_K decoder on an AMD RDNA GPU (see ../nvidia-rtx5060ti/run.py)."""

import sys

from _shared import run_nvidia_script

if __name__ == '__main__':
    run_nvidia_script('run.py', sys.argv[1:])
