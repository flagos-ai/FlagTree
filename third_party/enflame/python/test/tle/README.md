
## 官方测试目录

- python/tutorials/tle
- python/tutorials/tle/deepseek_v32
- python/test/tle/integration
- python/test/tle/unit


## tutorials test

### 01-fft.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 third_party/enflame/python/test/tle/tutorials/01-fft.py

tle-fft-performance:
        N  Triton (ms)   TLE (ms)  Torch (ms)
0    64.0     1.975675   1.850480    0.465881
1   128.0     2.981638   2.691489    0.855568
2   256.0     5.165217   5.880163    1.401453
3   512.0     8.917063   9.067560    3.109712
4  1024.0    26.312688  20.888568   14.254661
```
暂时关闭走 `fft_kernel_tle_reg` 的场景，原因是 kurama 暂不支持 `tl.gather`。

### 02-moe_align_block_size.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 third_party/enflame/python/test/tle/tutorials/02-moe_align_block_size.py

num_tokens  num_experts  source  triton_ms  triton_atomic_ms  triton_atomic_fused_ms  tle_cluster_fused_ms  sglang_cuda_ms  yiakwy_cuda_ms
       256           64    zipf     0.1893            0.2140                      na                    na              na              na
       512           64    zipf     0.1869            0.2154                      na                    na              na              na
      1024           64    zipf     0.1869            0.2187                      na                    na              na              na
      2048           64    zipf     0.1747            0.2235                      na                    na              na              na
      4096           64    zipf     0.1287            0.2365                      na                    na              na              na
      8192           64    zipf     0.1024            0.2651                      na                    na              na              na
     16384           64    zipf     0.2112            0.3210                      na                    na              na              na
     32768           64    zipf     0.2441            0.4379                      na                    na              na              na
     65536           64    zipf     0.4152            0.6488                      na                    na              na              na
    163840           64    zipf     1.0686            1.4101                      na                    na              na              na

```

### 03-topk.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 third_party/enflame/python/test/tle/tutorials/03-topk.py
tle-topk-radix-vs-triton-vs-torch:
       M        N      K  Triton-RadixSelect (ms)  Triton-TopK (ms)  Torch-TopK (ms)
0   64.0    128.0    8.0                 0.206590          0.262524         0.029271
1   64.0   1024.0   32.0                 1.139390          1.039800         0.040016
2   64.0   8192.0  128.0                 8.247208          4.566421         0.052819
3  128.0  32768.0  256.0                49.890430         33.532890         0.150353
```

### 04-cluster-gemm.py

不支持

### deepseek_v32/01-topk_selector.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 third_party/enflame/python/test/tle/tutorials/deepseek_v32/01-topk_selector.py

topk-selector:
   batch   seq_len    topk  Triton (ms)  TRTLLM-Prefill (ms)  TRTLLM-Prefill-1024T (ms)  FlashInfer (ms)  TLE-TRT (ms)  TLE-TRT-1024T (ms)  TLE-Cluster (ms)
0    1.0  131072.0  2048.0    19.182341                  NaN                        NaN              NaN     51.306240           50.588268               NaN
1    1.0  262144.0  2048.0    37.385952                  NaN                        NaN              NaN    101.877792          100.372490               NaN
2    1.0  524288.0  2048.0    73.046127                  NaN                        NaN              NaN    203.172867          199.681854               NaN
3   64.0    4096.0   128.0     2.713480                  NaN                        NaN              NaN      3.787944            4.945063               NaN
4   64.0    8192.0   256.0     4.129827                  NaN                        NaN              NaN      7.467942            9.655407               NaN
5   64.0   32768.0  1024.0    12.919387                  NaN                        NaN              NaN     29.682093           40.519657               NaN
6   64.0   32768.0  2048.0    13.177432                  NaN                        NaN              NaN     29.616543           39.429501               NaN
7   64.0  131072.0  2048.0    49.329521                  NaN                        NaN              NaN    112.933456          151.970947               NaN
8   64.0  524288.0  2048.0   186.628342                  NaN                        NaN              NaN    447.051849          600.195801               NaN

```

### deepseek_v32/02-sparse-mla.py

  修复内容

  问题根因：grid 布局为 (B, SQ, VG*NH)，其中 SQ（序列长度 512/1024/2048）被分配到 grid.y（program_id(1)），但 GCU400 硬件限制 grid.y <= 255，导致所有测试用例都因资源不足而失败。

  修复方案：将 SQ 和 B 在 grid 中的位置互换：
  • grid = (B, SQ, VG*NH) → grid = (SQ, B, VG*NH)
  • 对应地，两个 kernel 中 program_id(0) 和 program_id(1) 的含义也互换
  • SQ 放到 grid.x（限制 65535），B 放到 grid.y（限制 255），B 通常很小（1-8），不会超限

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 third_party/enflame/python/test/tle/tutorials/deepseek_v32/02-sparse-mla.py
tle-sparse-mla-fwd:
     B       S     SKV      H  HKV    DQK     DV    topk  Triton (ms)     TLE (ms)
0  1.0   512.0  1024.0  128.0  1.0  192.0  128.0   512.0   113.260834   243.841461
1  1.0  1024.0  2048.0  128.0  1.0  192.0  128.0  1024.0   439.732819   880.081970
2  1.0  2048.0  4096.0  128.0  1.0  192.0  128.0  2048.0  1720.542725  3415.426270
3  1.0  1024.0  2048.0  128.0  1.0  160.0  128.0  1024.0   400.470856   724.980835
```


## integration test

### test_tle_distributed.py

不支持

### test_tle_gemm.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/integration/test_tle_gemm.py
```

### test_tle_local_store.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/integration/test_tle_local_store.py
```

### test_tle_pipeline_e2e.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/integration/test_tle_pipeline_e2e.py
```

### test_tle_tma_copy.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/integration/test_tle_tma_copy.py
```

### test_tle_topk_smem_fallback.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/integration/test_tle_topk_smem_fallback.py
```


## unittest

### test_extract_tile_dynamic_index.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_extract_tile_dynamic_index.py
```

### test_extract_tile_static_index.py

原始测试脚本只有一个 CTA，会导致 DSM 分配失败，改为 grid = (2, 2)

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_extract_tile_static_index.py
```

### test_insert_tile_dynamic_index.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_insert_tile_dynamic_index.py
```

### test_insert_tile_static_index.py

原始测试脚本只有一个 CTA，会导致 DSM 分配失败，改为 grid = (2, 2)

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_insert_tile_static_index.py
```

### test_tle_cumsum.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_tle_cumsum.py
```

### test_tle_distributed.py

不支持

### test_tle_gpu_local_ptr.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_tle_gpu_local_ptr.py
```

### test_tle.py

```bash
TRITON_ALWAYS_COMPILE=1 python3.12 -m pytest -s -v third_party/enflame/python/test/tle/unit/test_tle.py
```
