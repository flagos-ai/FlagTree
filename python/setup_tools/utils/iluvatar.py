import subprocess


def skip_package_dir(package):
    return True


def get_resources_url(resource_name):
    if resource_name == "llvm":
        return "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatar-llvm18-x86_64_v0.5.0.tar.gz"
    if resource_name == "plugin":
        return _resolve_plugin_url(_detect_gxx_abi_version())
    raise KeyError(resource_name)


def get_resources_hash(resource_name):
    if resource_name == "plugin":
        return _resolve_plugin_md5(_detect_gxx_abi_version())
    raise KeyError(resource_name)


_ILUVATAR_PLUGIN_URLS = {
    # CentOS7/Ubuntu20
    "1013":
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.17-glibcxx3.4.19-cxxabi1.3.13-linux-x86_64_v0.5.0.tar.gz",
    # Ubuntu22
    "1016":
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.16-ubuntu-x86_64_v0.5.0.tar.gz",
    # Ubuntu24
    "1018":
    "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/iluvatarTritonPlugin-cpython3.10-glibc2.39-glibcxx3.4.33-cxxabi1.3.18-ubuntu-x86_64_v0.5.0.tar.gz",
}

_ILUVATAR_PLUGIN_MD5 = {
    # CentOS7/Ubuntu20
    "1013": "31fa623b",
    # Ubuntu22
    "1016": "52142296",
    # Ubuntu24
    "1018": "fff7cb48",
}


def _detect_gxx_abi_version():
    try:
        macros = subprocess.check_output(["bash", "-lc", "echo | g++ -dM -E -x c++ -"], stderr=subprocess.DEVNULL,
                                         text=True)
    except Exception:
        return ""
    for line in macros.splitlines():
        if "__GXX_ABI_VERSION" in line:
            parts = line.strip().split()
            if parts:
                return parts[-1]
    return ""


def _resolve_plugin_url(abi):
    url = _ILUVATAR_PLUGIN_URLS.get(abi)
    if url:
        print(f"[INFO] Selected iluvatar plugin package for __GXX_ABI_VERSION={abi}")
        return url

    raise RuntimeError(f"Unsupported iluvatar plugin ABI: __GXX_ABI_VERSION={abi}. "
                       "Please update utils/iluvatar.py with a matching package.")


def _resolve_plugin_md5(abi):
    md5 = _ILUVATAR_PLUGIN_MD5.get(abi)
    if md5:
        return md5

    raise RuntimeError(f"Unsupported iluvatar plugin ABI: __GXX_ABI_VERSION={abi}. "
                       "Please update utils/iluvatar.py with a matching package.")
