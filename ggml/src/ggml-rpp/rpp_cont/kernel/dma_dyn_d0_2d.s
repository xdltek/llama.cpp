//ENTRY:
.param .u32 src_ddr_low
.param .u32 src_ddr_hi
.param .u32 dst_ddr_low
.param .u32 dst_ddr_hi
.param .u32 in_stride
.param .u32 out_stride
.param .u32 len


.var .u32 %src_low
.var .u32 %src_hi
.var .u32 %dst_low
.var .u32 %dst_hi
.var .u32 %len
.var .u32 %in_stride
.var .u32 %out_stride
.var .u16 %cont_d0
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2
.var .u16 %one

LDPARAM.U16 [tid_xyz], %sreg0 
LDPARAM.U16 [tbDim.x], %sreg1
LDPARAM.U16 [tbDim.y], %sreg2
CFGXYZ %sreg1, %sreg2, %sreg0


/////////////////////////////////////////////////////////////
//Below 1 parameters set by warmup interface
//cfg_fft_w4_5 --> cont_d0
/////////////////////////////////////////////////////////////
LDPARAM.U16 [cfg_fft_w4_5], %cont_d0
MOVE.U16 1, %one

LDPARAM.U32 [src_ddr_low], %src_low
LDPARAM.U32 [src_ddr_hi], %src_hi
LDPARAM.U32 [dst_ddr_low], %dst_low
LDPARAM.U32 [dst_ddr_hi], %dst_hi
LDPARAM.U32 [len], %len
LDPARAM.U32 [in_stride], %in_stride
LDPARAM.U32 [out_stride], %out_stride


$LOOP:
STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
SUB.U16 %cont_d0, %one, %cont_d0
ADD.U32 %src_low, %in_stride, %src_low
ADD.U32 %dst_low, %out_stride, %dst_low
JMPC %cont_d0, $LOOP
##