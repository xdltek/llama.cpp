# ggml-rpp User Guide

`ggml-rpp` is the RPP backend implementation for llama.cpp. It dispatches supported GGML operators to RPP devices. The current version mainly uses kernel mode and covers common LLM inference operators, including matrix multiplication, attention, normalization, GLU, RoPE, elementwise operators, and MoE operators.

This document describes dependencies, build steps, runtime parameters, and common debugging switches. See [CHANGELOG.md](CHANGELOG.md) in the same directory for version changes.

## Feature Overview

Current `llama.cpp` capabilities:

- All supported operators run in kernel mode.
- `mul_mat` supports quantization formats such as `q4_k`, `q5_k`, `q6_k`, `q8_0`, `iq2_s`, `iq2_xs`, and `iq3_xxs`, as well as the BF16 path.
- Supports Qwen3, Qwen 30B MoE IQ2M, Phi-4 text, and related model scenarios.
- Supports RPP graph capture, operator fusion, weight cache, Perfetto tracing, and debug dumps.
- Recommends BF16 KV cache and `n_ubatch=512`.

## Dependencies

The default RPP installation directory is `/usr/local/rpp`. You can override it with the `RPP_HOME` environment variable or the CMake variable `RPP_INSTALL_DIR`.

The RPP installation directory must contain:

- `include/rpp_drv_api.h`
- `include/rpp_runtime.h`
- `lib/liburpp.so`, or the corresponding RPP driver library for the target platform

Optional dependencies:

- When `GGML_RPP_USE_RT=ON` is enabled, `include/Infer.h` and `lib/libRppRT.so` are required.
- When `GGML_RPP_PERF_TRACE=ON` is enabled, `include/rpp_perf.h` and `lib/librpp_perf.so` are required.

## Build

Basic build:

```bash
mkdir -p build
cd build
cmake .. -DGGML_RPP=ON
cmake --build . -j8
```

If RPP is not installed under `/usr/local/rpp`:

```bash
export RPP_HOME=/path/to/rpp
cmake .. -DGGML_RPP=ON
```

Or:

```bash
cmake .. -DGGML_RPP=ON -DRPP_INSTALL_DIR=/path/to/rpp
```

The build creates the `ggml-rpp` backend library. The `ggml-rpp-kernels` target copies prebuilt kernel object files to the `rpp_kernel/` directory next to the runtime binary. During installation, `rpp_kernel/` is also installed to `CMAKE_INSTALL_BINDIR`.

## Common CMake Options

- `GGML_RPP_USE_BF16=ON`: enable BF16 operators. Default: `ON`.
- `GGML_RPP_PERF_TRACE=OFF`: enable Perfetto tracing. Default: `OFF`.
- `GGML_RPP_USE_DFS=OFF`: enable dynamic frequency scaling. Default: `OFF`.
- `GGML_RPP_USE_DFS_FLEXIBLE=OFF`: enable flexible DFS. Default: `OFF`.
- `GGML_RPP_DUMP_OPS=OFF`: dump graph and operator information. Default: `OFF`.
- `GGML_RPP_USE_UBATCH=ON`: enable micro-batch related paths. Default: `ON`.
- `GGML_RPP_USE_ASYNC=ON`: enable asynchronous operations. Default: `ON`.
- `GGML_RPP_USE_RT=OFF`: enable the legacy OpenRT / RppRT path. Default: `OFF`.
- `GGML_RPP_USE_GRAPHS=1`: enable RPP graphs. Default: `1`.
- `GGML_RPP_NO_PEER_COPY=1`: disable peer copy / event pipeline. Default: `1`.
- `GGML_RPP_SAVE_ENGINE=<path>`: directory for saving engines in the OpenRT path.
- `GGML_RPP_LOAD_ENGINE=<path>`: directory for loading engines in the OpenRT path.
- `GGML_RPP_PEER_MAX_BATCH_SIZE=<n>`: maximum batch size configuration for peer-related paths.

Example:

```bash
cmake .. \
  -DGGML_RPP=ON \
  -DGGML_RPP_USE_BF16=ON \
  -DGGML_RPP_USE_UBATCH=ON \
  -DGGML_RPP_USE_ASYNC=ON \
  -DGGML_RPP_PERF_TRACE=OFF
```

## Runtime Parameter Recommendations

RPP currently recommends BF16 KV cache and `n_ubatch=512`. If you use the llama.cpp C API, configure it as follows:

```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ubatch = 512;
ctx_params.type_k = GGML_TYPE_BF16;
ctx_params.type_v = GGML_TYPE_BF16;
```

If you use `llama-server`, explicitly set the context size, KV cache type, and micro-batch size:

```bash
./llama-server \
  -m /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8002 \
  -c 8192 \
  -ctk bf16 \
  -ctv bf16 \
  --no-warmup \
  -ub 512 \
  --context-shift \
  --fit off \
  -np 1 \
  --keep 128
```

Recommended options:

- `-c 8192`: set the context size.
- `-ctk bf16` / `-ctv bf16`: set the K/V cache type to BF16.
- `-ub 512`: set the micro-batch size.
- `--context-shift`: enable context shift.
- `--fit off`: disable automatic parameter fitting to device memory, which is useful for fixed-configuration testing.
- `-np 1`: set the number of parallel slots to 1.
- `--keep 128`: keep the first 128 prompt tokens.

## Runtime Environment Variables

- `GGML_RPP_BATCH_SIZE=512`: override `n_ubatch` in the RPP backend context. Default: `512`.
- `GGML_RPP_MAX_CONTEXT=8192`: override the maximum context size. Default: `8192`.
- `GGML_RPP_STUB_KV_STEP=0`: number of prebuilt KV steps for flash attention. Default: `0`. Each step corresponds to `256` tokens, so the default prebuild range is `2048` tokens. Longer sequences are built at runtime.
- `GGML_RPP_DISABLE_FUSION=1`: disable operator fusion. Fusion is enabled by default.
- `GGML_RPP_DISABLE_GRAPH_CAPTURE=1`: disable RPP graph capture and use direct dispatch. Graph capture is enabled by default.
- `GGML_RPP_WEIGHTS_CACHE_FILE=/path/to/cache`: enable the converted weight cache. On a cache hit, converted weights are loaded from the cache file to reduce repeated conversion cost.
- `GGML_RPP_NO_PINNED=1`: disable pinned host allocation. Pinned host allocation is enabled by default.

Example:

```bash
export GGML_RPP_BATCH_SIZE=512
export GGML_RPP_MAX_CONTEXT=8192
export GGML_RPP_STUB_KV_STEP=4
export GGML_RPP_WEIGHTS_CACHE_FILE=/path/to/model.rpp.weights.cache
```

To debug graph capture or fusion issues:

```bash
export GGML_RPP_DISABLE_GRAPH_CAPTURE=1
export GGML_RPP_DISABLE_FUSION=1
```

## Weight Cache

After `GGML_RPP_WEIGHTS_CACHE_FILE` is set, RPP writes converted weight layouts to the cache file. When loading the same model later and the cache key matches, RPP reads the converted weights from the cache file and copies them to device memory.

Notes:

- The cache file stores RPP-backend-specific converted weights, not original GGUF tensors.
- After changing weight conversion logic, cache version, model files, or tensor metadata, delete the old cache and regenerate it.
- The directory that contains the cache file must be writable.
- The first load is a cold start and creates or fills the cache. Cache-hit benefits appear on later loads.

## Kernel Files

RPP kernel object files are copied to the `rpp_kernel/` directory next to the runtime binary during the build. If kernel loading fails at runtime, check:

- Whether the corresponding `.o` files exist under `build/bin/rpp_kernel/` or the actual runtime directory.
- Whether the `ggml-rpp-kernels` target has been built.
- Whether `CMAKE_INSTALL_BINDIR/rpp_kernel/` is complete after installation.

Common kernel files include:

- `matmul_q4.o`, `matmul_q4_vxm.o`
- `matmul_q4k.o`, `matmul_q5k.o`, `matmul_q6k.o`, `matmul_q80.o`
- `matmul_q2s.o`, `matmul_q2xs.o`, `matmul_q3xxs.o`
- `rmsnorm.o`, `norm.o`
- `gelu.o`, `silu.o`, `tanh.o`
- `flash_atten.o`, `flash_atten_vxm.o`
- `set_rows.o`, `get_rows.o`
- `elementwise.o`, `rope.o`, `scale.o`

## Performance Debugging Tips

- Compare graph capture with direct dispatch by setting `GGML_RPP_DISABLE_GRAPH_CAPTURE=1`.
- Compare fusion behavior by setting `GGML_RPP_DISABLE_FUSION=1`.
- Compare weight conversion cost by testing cold start and warm start with and without `GGML_RPP_WEIGHTS_CACHE_FILE`.
- Tune ubatch by changing `GGML_RPP_BATCH_SIZE`. A common starting point is `512`.
- Tune the flash attention prebuild range by changing `GGML_RPP_STUB_KV_STEP`.
- To use Perfetto tracing, build with `GGML_RPP_PERF_TRACE=ON` and ensure the RPP perf library is available.

## FAQ

### CMake Cannot Find RPP Headers or Libraries

Make sure `RPP_HOME` or `RPP_INSTALL_DIR` points to the correct RPP installation directory:

```bash
export RPP_HOME=/usr/local/rpp
```

Then check:

```bash
ls $RPP_HOME/include/rpp_drv_api.h
ls $RPP_HOME/include/rpp_runtime.h
ls $RPP_HOME/lib
```

### Kernel Object Files Are Missing at Runtime

Rebuild and make sure `rpp_kernel/` has been copied to the executable directory:

```bash
cmake --build build --target ggml-rpp-kernels -j8
```

### Incorrect Results or Unexpected Performance

Check the following in order:

1. Make sure the KV cache type is BF16 and `n_ubatch` matches `GGML_RPP_BATCH_SIZE`.
2. Set `GGML_RPP_DISABLE_FUSION=1` to check whether the issue is related to fusion.
3. Set `GGML_RPP_DISABLE_GRAPH_CAPTURE=1` to check whether the issue is related to graph capture or replay.
4. Delete the old `GGML_RPP_WEIGHTS_CACHE_FILE` and reload to rule out stale cache data.
5. Check whether `/usr/local/rpp/etc/rpp_memcfg.ini` and `rpp_syscfg.ini` satisfy the current model size.
