"""Run ../nvidia-rtx5060ti/llama_golden.py against a HIP llama.cpp build and record the HIP backend."""

import sys

from _shared import output_argument, relabel_llama_backend, run_nvidia_script, split_arch_argument

if __name__ == '__main__':
    arch, argv = split_arch_argument(sys.argv[1:])
    run_nvidia_script('llama_golden.py', argv)
    relabel_llama_backend(output_argument(argv), arch)
