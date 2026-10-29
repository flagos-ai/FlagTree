"""RDNA IQ4 kernel replacements for the Qwen3.8-27B decode tutorial.

``install()`` puts this package's templates in front of FlagMega's for the
current process, the same mechanism as the NVIDIA tutorial's
``agent_optimizations``. With ``q8_activations`` the native IQ4_XS GEMVs
quantize each call's activation once per CTA to q8_1 (round-to-nearest-even
int8, one fp16 scale per 32 values; the NVIDIA tutorial's ``q8_1_rne``
contract and llama.cpp's MMVQ activations) in LDS and dot it with
``v_dot4_i32_iu8``. Q4_K and every other projection keep the default path.
"""

import os
from pathlib import Path

from jinja2 import ChoiceLoader, FileSystemLoader
from triton.flagmega.codegen.triton import tir_package
from triton.flagmega.codegen.triton.templates import TritonTemplateRegistry

ROOT = Path(__file__).parent / 'templates'


class RdnaIQ4Registry(TritonTemplateRegistry):

    def __init__(self):
        super().__init__()
        self.environment.loader = ChoiceLoader([FileSystemLoader(str(ROOT)), self.environment.loader])

    def resolve(self, spec):
        for path in self.candidates(spec):
            if (ROOT / path).is_file():
                return path
        return super().resolve(spec)

    def render(self, template_name, context):
        prefetch = os.environ.get('FLAGMEGA_AMD_Q8_PREFETCH_KB')
        if prefetch is not None:
            context = {**context, 'q8_weight_prefetch_kb': int(prefetch)}
        return super().render(template_name, context)


def install(*, q8_activations=True):
    if q8_activations:
        tir_package.TritonTemplateRegistry = RdnaIQ4Registry
