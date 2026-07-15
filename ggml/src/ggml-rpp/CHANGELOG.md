# ggml-rpp CHANGELOG

## test_framwork:ce74d1c7ff2afa28c7d2295ddec6494f28a806cb

### Related Project Commits

- `rpp_drv`: `e780c6a528f8c8ee135504506fe52300b521b95f`
- `rpp_drv_api`: `eed2f2d491662a816ec8a4beb3b75b5b32d46d58`
- `rpp_tool_chain`: `db009b6ad75133636ae0c16617f23402cebc5b1a`

### Test Host Configuration

- OS: Ubuntu 20.04, Linux `5.15.0-139-generic`
- CPU: AMD Ryzen 5 5600G with Radeon Graphics
- CPU topology: 1 socket, 6 cores, 12 threads
- Memory: 64 GiB
- Model: Qwen3-30B-A3B-IQ2_M.gguf
- Test program: llama-simple-chat

### Changes

- Optimized weight loading by reducing host-side allocation, free, and synchronization overhead during model loading. On the current test host, weight loading time was reduced from about `33s` to about `17s`.
- Optimized graph initialization. With `GGML_RPP_STUB_KV_STEP=0`, first-token latency was reduced from about `15.6s` to about `4.5s`, and second-token latency was reduced from about `20s` to about `4.3s`. With `GGML_RPP_STUB_KV_STEP=32`, first-token latency was reduced from about `136s` to about `35s`, and second-token latency was reduced from about `117s` to about `6.5s`.
- Improved context length and ubatch configuration. Extra environment variables are no longer required for these values; users can configure context size and micro-batch size directly with `llama-server` parameters such as `-c 8192 -ub 512`.
- Optimized host memory usage for the 30B MoE model. Host memory usage was reduced from about `13G` to about `400M`.

