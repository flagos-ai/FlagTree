"""Check the installed IFWA backend and its native module without launching a kernel."""
import argparse
import importlib
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--editable-root', type=Path)
    parser.add_argument('--torch-ifwa', action='store_true')
    args = parser.parse_args()

    if args.torch_ifwa:
        import torch
        importlib.import_module('torch_ifwa')
        assert hasattr(torch, 'ifwa')

    import triton
    from triton.backends import backends
    from triton._C.libtriton import intellifusion, ir

    root = args.editable_root or Path(sys.prefix)
    assert Path(triton.__file__).resolve().is_relative_to(root.resolve()), triton.__file__
    assert 'intellifusion' in backends, backends
    driver = importlib.import_module(backends['intellifusion'].driver.__module__)
    assert Path(driver.__file__).with_name('driver.c').is_file()
    if args.torch_ifwa:
        from triton.backends.intellifusion.compiler import resolve_default_libdevice_path
        expected = Path(os.environ['IFWA_HOME']) / 'ifvm/libdevice/libdevice-ifwa.bc'
        assert expected.is_file(), expected
        assert resolve_default_libdevice_path() == str(expected)
    intellifusion.load_dialects(ir.context())
    print('IFWA installation check passed')


if __name__ == '__main__':
    main()
