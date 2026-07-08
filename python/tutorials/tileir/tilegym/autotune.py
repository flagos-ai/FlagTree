import os


def is_autotune_disabled():
    return os.environ.get("TILEGYM_DISABLE_AUTOTUNE", "0") == "1"
