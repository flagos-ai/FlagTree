# flagtree tle
import triton.language as _language


def __getattr__(name):
    language_extensions = getattr(_language, "ext", None)
    if language_extensions is None:
        raise RuntimeError(f"tle.gpu.tile.{name} requires a backend providing tl.ext.{name}")
    return getattr(language_extensions, name)
