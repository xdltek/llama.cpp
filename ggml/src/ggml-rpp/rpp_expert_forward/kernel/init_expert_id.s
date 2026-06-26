//ENTRY:
.param .u32 expert_id

.var .u32 %expert_id
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2

LDPARAM.U16 [tid_xyz], %sreg0
LDPARAM.U16 [tbDim.x], %sreg1
LDPARAM.U16 [tbDim.y], %sreg2
CFGXYZ %sreg1, %sreg2, %sreg0

//-------------------------------------------------------------
////cfg_fft_w4_10 --> expert_id
//-------------------------------------------------------------
LDPARAM.U32 [expert_id], %expert_id
STGLB.U32 %expert_id, [cfg_fft_w4_10]
##
