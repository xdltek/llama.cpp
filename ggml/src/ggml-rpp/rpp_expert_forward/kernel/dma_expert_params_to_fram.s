//ENTRY:
.param .u32 expert_count_ddr_low
.param .u32 expert_count_ddr_hi
.param .u32 expert_offset_ddr_low
.param .u32 expert_offset_ddr_hi
.param .u32 expert_count_fram0_low
.param .u32 expert_count_fram1_low
.param .u32 expert_offset_fram0_low
.param .u32 expert_offset_fram1_low
.param .u32 fram_hi


.var .u32 %src_low
.var .u32 %src_hi
.var .u32 %dst_low
.var .u32 %dst_hi
.var .u32 %len
.var .u32 %expert_idx
.var .u32 %dwReg0
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2

LDPARAM.U16 [tid_xyz], %sreg0
LDPARAM.U16 [tbDim.x], %sreg1
LDPARAM.U16 [tbDim.y], %sreg2
CFGXYZ %sreg1, %sreg2, %sreg0

//-------------------------------------------------------------
//cfg_fft_w4_6 --> expert_count_fram0_low
//cfg_fft_w4_7 --> expert_offset_fram0_low
//cfg_fft_w4_10 --> expert_id
//-------------------------------------------------------------

LDPARAM.U32 [cfg_fft_w4_10], %expert_idx
LDPARAM.U32 [expert_count_ddr_low], %src_low
LDPARAM.U32 [expert_count_ddr_hi], %src_hi
MOVE.U32 4, %len
MUL.U32.LOW  %expert_idx, %len, %dwReg0
ADD.U32 %src_low, %dwReg0, %src_low

LDPARAM.U32 [fram_hi], %dst_hi
LDPARAM.U32 [expert_count_fram0_low], %dst_low

//-------------------------------------------------------------
//DMA expert count to fram
//-------------------------------------------------------------
STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
LDPARAM.U32 [expert_count_fram1_low], %dst_low
STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
//-------------------------------------------------------------
//DMA expert offset to fram
//-------------------------------------------------------------

LDPARAM.U32 [expert_offset_ddr_low], %src_low
LDPARAM.U32 [expert_offset_ddr_hi], %src_hi
ADD.U32 %src_low, %dwReg0, %src_low
LDPARAM.U32 [expert_offset_fram0_low], %dst_low

STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
LDPARAM.U32 [expert_offset_fram1_low], %dst_low
STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##