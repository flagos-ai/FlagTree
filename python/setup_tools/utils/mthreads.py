import platform


def get_resources_url(resource_name):
    arch = platform.machine()
    if arch == 'aarch64':
        return {
            "llvm":
            "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-aarch64_v0.4.0.tar.gz",
            'plugin':
            "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-aarch64_v0.4.0.tar.gz"
        }[resource_name]
    elif arch == 'x86_64':
        return {
            "llvm":
            "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreads-llvm19-glibc2.35-glibcxx3.4.30-x64_v0.4.0.tar.gz",
            "plugin":
            "https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/mthreadsTritonPlugin-cpython3.10-glibc2.35-glibcxx3.4.30-cxxabi1.3.13-ubuntu-x86_64_v0.4.0.tar.gz"
        }[resource_name]


def get_resources_hash(resource_name):
    arch = platform.machine()
    if arch == 'aarch64':
        return {'plugin': "2f5f04a2"}[resource_name]
    elif arch == 'x86_64':
        return {"plugin": "2f5f04a2"}[resource_name]
