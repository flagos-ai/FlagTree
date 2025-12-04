def skip_package_dir(package):
    if package not in ['triton', 'triton/_C']:
        return True
    return False
