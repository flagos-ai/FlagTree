import os

from .. import __path__ as _language_paths

# Preserve per-module fallback after selecting the specialized AMD package.
for _language_path in _language_paths:
    _amd_path = os.path.join(_language_path, "amd")
    if os.path.isdir(_amd_path) and _amd_path not in __path__:
        __path__.append(_amd_path)

from ._layouts import AMDMFMALayout, AMDWMMALayout
from . import cdna3, cdna4
from . import rdna3, rdna4
from . import gfx1250

__all__ = ["AMDMFMALayout", "AMDWMMALayout", "cdna3", "cdna4", "rdna3", "rdna4", "gfx1250"]
