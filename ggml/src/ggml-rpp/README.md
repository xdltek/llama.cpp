# Version History

## tag: llama.cpp_v0.0.1:

### Update contents
1.support qwen3  
2.flash_attn, set_rows operators on cpu, and others operaors on rpp  
3.support ubatch=128, ubatch is set by user  

### Dependency librarys
The branchs:
- rpprt:commit c69311548b03515d527c1b76081658475f739971 (HEAD -> llm_awq_8k_dev, origin/llm_awq_8k_dev, origin/HEAD)  
- rpp_drv:commit fd539a17f3874b83b0504c7b702e758614ac3e42 (HEAD -> master, origin/master, origin/HEAD)  
- rpp_drv_api:commit 55ae672c48c174dd6cabd336aec41bb3c919cf89 (HEAD -> master, origin/master, origin/HEAD)  
- rpp_tool_chain:commit d4576e8269c14d6d515544bad85fd7853a7b96ca (HEAD -> master, origin/master, origin/HEAD)  

### Modify the code
1.drv_api  
file: rpp_server_common.h  
modify "#define RPP_DYNAMIC_QUEUE_NUM_2 826"  to "#define RPP_DYNAMIC_QUEUE_NUM_2 1530"  

2.rpp_memcfg.ini   
file: /usr/local/rpp/etc/rpp_memcfg.ini  
refer to the memory size of your own rpp, you need fix region4 and region9:  
region4,  kfunc,  0x30000000, 0x10000000,  
region9, dmem,  0x40000000, 0x1c0000000,  

## tag: llama.cpp_v0.0.2:

### Update contents
1.reuse nodes  
2.allocate memory of weights and nodes on device  
3.support glu and rope for phi4  
4.flash atten and set rows operators on rpp  
5.support ut test  

### Dependency librarys
- rpprt:commit 85feadf18d761522ab09c6e06b266a37f51a9297 (HEAD -> llm_awq_8k_dev, origin/llm_awq_8k_dev, origin/HEAD)
- rpp_drv:commit fd539a17f3874b83b0504c7b702e758614ac3e42 (HEAD -> master, origin/master, origin/HEAD)
- rpp_drv_api:commit 55ae672c48c174dd6cabd336aec41bb3c919cf89 (HEAD -> master, origin/master, origin/HEAD)
- rpp_tool_chain:commit d4576e8269c14d6d515544bad85fd7853a7b96ca (HEAD -> master, origin/master, origin/HEAD)

## tag: llama.cpp_v0.0.3
1.support memory in device  
2.support q4_1 and bf16  
3.mul_mat,add,rms_norm in kernel mode, others in rpprt mode  

### Dependency librarys
- rpprt:commit 8e86105a5db375a98951dad336cdd3bb0b20824e (HEAD -> llm_awq_8k_dev, origin/llm_awq_8k_dev, origin/HEAD)
- rpp_drv:commit fd539a17f3874b83b0504c7b702e758614ac3e42 (HEAD -> master, origin/master, origin/HEAD)
- rpp_drv_api:commit 55ae672c48c174dd6cabd336aec41bb3c919cf89 (HEAD -> master, origin/master, origin/HEAD)
- rpp_tool_chain:commit d4576e8269c14d6d515544bad85fd7853a7b96ca (HEAD -> master, origin/master, origin/HEAD)

### Modify the code
notice!!! if run q4_1 model, you need modify the source code, and run bf16, no need to modify the code  
1.drv_api  
file: rpp_server_common.h  
modify "#define RPP_DYNAMIC_QUEUE_NUM_2 826"  to "#define RPP_DYNAMIC_QUEUE_NUM_2 2536"  
file: ll_event.h  
modify "#define  RPP_EVENT_COMMON_NUM 1900"  to "#define RPP_EVENT_COMMON_NUM 2900"  
2.rpp_memcfg.ini  
file: /usr/local/rpp/etc/rpp_memcfg.ini  
refer to the memory size of your own rpp, you need fix region3, region4 and region9:   
region4, KFUNC,            0x30000000,     0x20000000,  
region7, GRAPH_KPARA,      0x13000000,     0x6000000,  
region8, DYNAIC_KQUEUE,    0x19000000,     0x17000000,  
region9, DMEM,             0x50000000,     0x3c0000000,  

## tag: llama.cpp_v0.0.4
1.all operator in kernel mode 
2.support q4k/q5k/q6k/iq2s/iq2xs/iq3xxs for mul_mat 
3.support qwen30B_moe_IQ2M/phi4_text/qwen3 0.6/1.7/8B
4.support perf

### Dependency librarys
- rpprt:commit fd539a17f3874b83b0504c7b702e758614ac3e42 (HEAD -> llm_awq_8k_dev, origin/llm_awq_8k_dev, origin/HEAD)
- rpp_drv:commit fd539a17f3874b83b0504c7b702e758614ac3e42 (HEAD -> master, origin/master, origin/HEAD)
- rpp_drv_api:commit 71729ca195a5e5613f74645ac1d224100f45c336 (HEAD -> master, origin/master, origin/HEAD)
- rpp_tool_chain:commit d4576e8269c14d6d515544bad85fd7853a7b96ca (HEAD -> master, origin/master, origin/HEAD)

### Modify the code
1.drv_api  
file: rpp_server_common.h  
modify "#define RPP_DYNAMIC_QUEUE_NUM_2 826"  to "#define RPP_DYNAMIC_QUEUE_NUM_2 2536"  
2.rpp_memcfg.ini  
file: /usr/local/rpp/etc/rpp_memcfg.ini  
refer to the memory size of your own rpp, you need fix region4, region7, region8 and region9:   
region4=<4,     KFUNC,            0x3c000000,     0x14000000,
region7=<7,     GRAPH_KPARA,      0x13000000,     0x12000000,
region8=<8,     DYNAIC_KQUEUE,    0x25000000,     0x17000000,
region9=<9,     DMEM,             0x50000000,     0x3c0000000,
3.rpp_syscfg.ini 
file: /usr/local/rpp/etc/rpp_syscfg.ini  
glb_common_event=3700
if you want run phi4, can set glb_common_event=3500

# Add code on your sample code
1.Because of only support bf16, so need set ctx_param kv type to bf16, and set n_ubatch=128:
```
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ubatch = 128;
    ctx_params.type_k = GGML_TYPE_BF16;
    ctx_params.type_v = GGML_TYPE_BF16;
```

# Compile command
mkdir build && cd build    
cmake .. -DGGML_RPP=ON && make -j8 

## Compile macros
Enable use bf16 operations:         GGML_RPP_USE_BF16=ON                default:OFF  
Enable perfetto tracing:            GGML_RPP_PERF_TRACE=ON              default:OFF 
Enable to dump ops information:     GGML_RPP_DUMP_OPS=ON                default:OFF 
Enable use upatch operations:       GGML_RPP_USE_UBATCH=ON              default:ON 
Enable asynchronous operations:     GGML_RPP_USE_ASYNC=ON               default:ON 
Enable dynamic frequency:           GGML_RPP_USE_DFS=OFF                default:OFF
Enable dynamic frequency flexible:  GGML_RPP_USE_DFS_FLEXIBLE=OFF       default:OFF  

## Environment variable
Support batch size:                 GGML_RPP_BATCH_SIZE=128    default:128, 
Support max context size:           GGML_RPP_MAX_CONTEXT=4096  default:4096 
Disable operator fusion:            GGML_RPP_DISABLE_FUSION=0  default:0 
Pre-build flash_attn_ext numbers:   GGML_RPP_STUB_KV_STEP=8    default:8, 8*256=2048 tokens, if tokens number > 2048, flash_attn_ext will be built in real-time
Disable graph capture, launch graph with one or multiple:                  GGML_RPP_DISABLE_GRAPH_CAPTURE=0  default:0 
Enable cache file,weights will be load from the cache file first:          GGML_RPP_WEIGHTS_CACHE_FILE=your cache path 


