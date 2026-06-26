cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GGML_RPP_USE_BF16 "RPP:Enable use bf16 operations" ON)
if(GGML_RPP_USE_BF16)
    add_definitions(-DGGML_RPP_USE_BF16=1)
    message(STATUS "RPP: Use bf16 operations     : ENABLED   (-DGGML_RPP_USE_BF16=ON)")
else()
    add_definitions(-DGGML_RPP_USE_BF16=0)
    message(STATUS "RPP: Use bf16 operations     : DISABLED  (-DGGML_RPP_USE_BF16=OFF)")
endif()

option(GGML_RPP_PERF_TRACE "Enable Perfetto tracing" OFF)
if(GGML_RPP_PERF_TRACE)
    add_definitions(-DGGML_RPP_PERF_TRACE=1)
    message(STATUS "RPP: Perfetto tracing            : ENABLED   (-DGGML_RPP_PERF_TRACE=ON)")
else()
    add_definitions(-DGGML_RPP_PERF_TRACE=0)
    message(STATUS "RPP: Perfetto tracing            : DISABLED  (-DGGML_RPP_PERF_TRACE=OFF)")
endif()

option(GGML_RPP_USE_DFS "Enable DFS" OFF)
if(GGML_RPP_USE_DFS)
    add_definitions(-DGGML_RPP_USE_DFS=1)
    message(STATUS "RPP: Dynamic frequency           : ENABLED   (-DGGML_RPP_USE_DFS=ON)")
else()
    add_definitions(-DGGML_RPP_USE_DFS=0)
    message(STATUS "RPP: Dynamic frequency           : DISABLED  (-DGGML_RPP_USE_DFS=OFF)")
endif()

option(GGML_RPP_USE_DFS_FLEXIBLE "Enable DFS Flexible" OFF)
if(GGML_RPP_USE_DFS_FLEXIBLE)
    add_definitions(-DGGML_RPP_USE_DFS_FLEXIBLE=1)
    message(STATUS "RPP: Dynamic frequency flexible  : ENABLED   (-DGGML_RPP_USE_DFS_FLEXIBLE=ON)")
else()
    add_definitions(-DGGML_RPP_USE_DFS_FLEXIBLE=0)
    message(STATUS "RPP: Dynamic frequency flexible  : DISABLED  (-DGGML_RPP_USE_DFS_FLEXIBLE=OFF)")
endif()

option(GGML_RPP_DUMP_OPS "RPP:Enable to dump ops information" OFF)
if(GGML_RPP_DUMP_OPS)
    add_definitions(-DGGML_RPP_DUMP_OPS=1)
    message(STATUS "RPP: Dump ops information        : ENABLED   (-DGGML_RPP_DUMP_OPS=ON)")
else()
    add_definitions(-DGGML_RPP_DUMP_OPS=0)
    message(STATUS "RPP: Dump ops information        : DISABLED  (-DGGML_RPP_DUMP_OPS=OFF)")
endif()

option(GGML_RPP_USE_UBATCH "RPP:Enable use upatch operations" ON)
if(GGML_RPP_USE_UBATCH)
    add_definitions(-DGGML_RPP_USE_UBATCH=1)
    message(STATUS "RPP: Use upatch operations       : ENABLED   (-DGGML_RPP_USE_UBATCH=ON)")
else()
    add_definitions(-DGGML_RPP_USE_UBATCH=0)
    message(STATUS "RPP: Use upatch operations       : DISABLED  (-DGGML_RPP_USE_UBATCH=OFF)")
endif()

option(GGML_RPP_USE_ASYNC "RPP:Enable asynchronous operations" ON)
if(GGML_RPP_USE_ASYNC)
    add_definitions(-DGGML_RPP_USE_ASYNC=1)
    message(STATUS "RPP: Asynchronous operations     : ENABLED   (-DGGML_RPP_USE_ASYNC=ON)")
else()
    add_definitions(-DGGML_RPP_USE_ASYNC=0)
    message(STATUS "RPP: Asynchronous operations     : DISABLED  (-DGGML_RPP_USE_ASYNC=OFF)")
endif()

option(GGML_RPP_USE_RT "RPP:Enable OpenRT (Infer.h/RppRT)" OFF)
if(GGML_RPP_USE_RT)
    add_definitions(-DGGML_RPP_USE_RT=1)
    message(STATUS "RPP: OpenRT backend              : ENABLED   (-DGGML_RPP_USE_RT=ON)")
else()
    add_definitions(-DGGML_RPP_USE_RT=0)
    message(STATUS "RPP: OpenRT backend              : DISABLED  (-DGGML_RPP_USE_RT=OFF)")
endif()

# add_compile_options(-D_GNU_SOURCE)
add_compile_definitions(GGML_RPP_SAVE_ENGINE=${GGML_RPP_SAVE_ENGINE})
add_compile_definitions(GGML_RPP_LOAD_ENGINE=${GGML_RPP_LOAD_ENGINE})
add_compile_definitions(GGML_RPP_PEER_MAX_BATCH_SIZE=${GGML_RPP_PEER_MAX_BATCH_SIZE})

if("rpp${RPP_INSTALL_DIR}" STREQUAL "rpp" AND DEFINED ENV{RPP_HOME})
    set(RPP_INSTALL_DIR $ENV{RPP_HOME})
    message(STATUS "RPP: updated RPP_INSTALL_DIR from RPP_HOME=$ENV{RPP_HOME}")
endif()

if(NOT RPP_INSTALL_DIR)
    set(RPP_INSTALL_DIR "/usr/local/rpp")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    find_path(RPP_INCLUDE_DIR NAMES rpp_drv_api.h rpp_runtime.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
    find_library(RPP_DRV_LIBRARY NAMES urpp PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    find_library(RPP_DRV_UV_LIBRARY NAMES uv_a PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    find_library(RPP_DRV_OSAL_LIBRARY NAMES rpp_osal_a PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    set(RPP_LIBRARY  ${RPP_DRV_LIBRARY} ${RPP_DRV_UV_LIBRARY} ${RPP_DRV_OSAL_LIBRARY} Ws2_32.lib Iphlpapi.lib Userenv.lib)
    if(GGML_RPP_PERF_TRACE)
        find_path(RPP_PERF_INCLUDE_DIR NAMES rpp_perf.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
        find_library(RPP_PERF_LIBRARY NAMES rpp_perf PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
        list(APPEND RPP_LIBRARY ${RPP_PERF_LIBRARY})
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_path(RPP_INCLUDE_DIR NAMES rpp_drv_api.h rpp_runtime.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
    find_library(RPP_DRV_LIBRARY NAMES urpp  PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
    set(RPP_LIBRARY ${RPP_DRV_LIBRARY})
    if(GGML_RPP_USE_RT)
        find_path(RPP_RT_INCLUDE_DIR NAMES Infer.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
        find_library(RPP_RT_LIBRARY NAMES RppRT  PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
        list(APPEND RPP_LIBRARY ${RPP_RT_LIBRARY})
    endif()
    if(GGML_RPP_PERF_TRACE)
        find_path(RPP_PERF_INCLUDE_DIR NAMES rpp_perf.h PATHS ${RPP_INSTALL_DIR}/include REQUIRED)
        find_library(RPP_PERF_LIBRARY NAMES rpp_perf  PATHS ${RPP_INSTALL_DIR}/lib REQUIRED)
        list(APPEND RPP_LIBRARY ${RPP_PERF_LIBRARY})
    endif()
endif()

option(GGML_RPP_USE_GRAPHS "Enable RPP graphs" 1)
option(GGML_RPP_NO_PEER_COPY "Disable RPP events/peer copy (0 = enable events for pipeline parallelism)" 1)

set(GGML_RPP_INCLUDE 
    ${RPP_INCLUDE_DIR}
    ${RPP_RT_INCLUDE_DIR}
    ${RPP_PERF_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)

file(GLOB GGML_RPP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_add/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_flash_attn_ext/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_get_rows/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_glu/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_rms_norm/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_l2_norm/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_rope/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_soft_max/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_set_rows/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_cpy/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_cont/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_scale/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_permute/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_norm/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_pool_2d/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_div/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat_id/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_reduce_sum/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_routing/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_unary/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_forward/src/*.cpp"
)

ggml_add_backend_library(ggml-rpp ${GGML_RPP_SOURCES})
target_link_libraries(ggml-rpp PRIVATE ${RPP_LIBRARY})
target_include_directories(ggml-rpp PRIVATE ${GGML_RPP_INCLUDE})
target_compile_definitions(ggml-rpp PRIVATE 
        GGML_RPP_USE_GRAPHS=${GGML_RPP_USE_GRAPHS}
        GGML_RPP_NO_PEER_COPY=${GGML_RPP_NO_PEER_COPY}
    )

if (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND TARGET ggml-cpu)
    # rpp_cpy uses ggml_compute_forward_dup from ggml-cpu/ops.cpp
    # Link ggml-cpu explicitly and export symbols from ggml-cpu DLL.
    target_link_libraries(ggml-rpp PRIVATE ggml-cpu)
    if (NOT GGML_BACKEND_DL)
        set_target_properties(ggml-cpu PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

# copy .o file
add_custom_command(TARGET ggml-rpp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q4_1/matmul_q4.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q4.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q4_1_vxm/matmul_q4_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q4_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_rms_norm/kernel/rmsnorm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/rmsnorm.o"  
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul/kernel/elementwise.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/elementwise.o" 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_rope/kernel/rope.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/rope.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_scale/kernel/scale.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/scale.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_norm/kernel/norm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/norm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_glu/kernel_geglu_erf/gelu.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/gelu.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_glu/kernel_swiglu/silu.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/silu.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_flash_attn_ext/kernel/flash_atten.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/flash_atten.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_flash_attn_ext/kernel_vxm/flash_atten_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/flash_atten_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_set_rows/kernel/set_rows.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/set_rows.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_get_rows/kernel/get_rows.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/get_rows.o"  
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q8_0_vxm/matmul_q80_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q80_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q8_0/matmul_q80.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q80.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q6_k_vxm/matmul_q6k_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q6k_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q6_k/matmul_q6k.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q6k.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q4_k_vxm/matmul_q4k_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q4k_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q4_k/matmul_q4k.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q4k.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q5_k_vxm/matmul_q5k_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q5k_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q5_k/matmul_q5k.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q5k.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q3_xxs_vxm/matmul_q3xxs_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q3xxs_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q3_xxs/matmul_q3xxs.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q3xxs.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q3_xxs_vxm_nolut/matmul_q3xxs_vxm_nolut.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q3xxs_vxm_nolut.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q3_xxs_nolut/matmul_q3xxs_nolut.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q3xxs_nolut.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_s/matmul_q2s.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2s.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_s_vxm/matmul_q2s_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2s_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_s_nolut/matmul_q2s_nolut.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2s_nolut.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_s_vxm_nolut/matmul_q2s_vxm_nolut.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2s_vxm_nolut.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_xs/matmul_q2xs.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2xs.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_xs_vxm/matmul_q2xs_vxm.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2xs_vxm.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_xs_nolut/matmul_q2xs_nolut.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2xs_nolut.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_q2_xs_vxm_nolut/matmul_q2xs_vxm_nolut.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_q2xs_vxm_nolut.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_mul_mat/kernel_bf16/matmul_bf16.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_bf16.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_cont/kernel/cont.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/cont.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_reduce_sum/kernel/reduce_sum.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/reduce_sum.o"    
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_pool_2d/kernel/pool_2d.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/pool_2d.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_routing/kernel/expert_routing.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/expert_routing.o"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_cpy/kernel/copy.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/copy.o"              
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_forward/kernel/matmul_id_fusion.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/matmul_id_fusion.o" 
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/rpp_expert_forward/kernel_vxm/kernel_topk_combine/topk_combine.o"
          "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/topk_combine.o"   
)

# install to CMAKE_INSTALL_BINDIR
install(DIRECTORY 
    "$<TARGET_FILE_DIR:ggml-rpp>/rpp_kernel/"
    DESTINATION ${CMAKE_INSTALL_BINDIR}/rpp_kernel
    USE_SOURCE_PERMISSIONS
    COMPONENT Runtime
)