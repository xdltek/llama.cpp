//ENTRY:
.param .u32 row_data_ddr_low
.param .u32 row_data_ddr_hi
.param .u32 dst_sram_low
.param .u32 dst_sram_hi
.param .u32 bytes_per_row


.var .u32 %src_low
.var .u32 %src_hi
.var .u32 %dst_low
.var .u32 %dst_hi
.var .u32 %len
.var .u32 %start_rowid
.var .u32 %start_row_offset
.var .u32 %dwReg0
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2

LDPARAM.U16 [tid_xyz], %sreg0 
LDPARAM.U16 [tbDim.x], %sreg1
LDPARAM.U16 [tbDim.y], %sreg2
CFGXYZ %sreg1, %sreg2, %sreg0

/////////////////////////////////////////////////////////////
////cfg_fft_w4_0 --> start_rowid
////cfg_fft_w4_1 --> cont_rowlen
/////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////
//row_data_ddr[0]: 0x1111,  0x1111, .....       0x1111
//row_data_ddr[1]: 0x2222,  0x2222, .....       0x2222
//row_data_ddr[2]: 0x3333,  0x3333, .....       0x3333
//row_data_ddr[3]: 0x4444,  0x4444, .....       0x4444
/////////////////////////////////////////////////////////////
//Below 2 parameters set by warmup interface
//start_rowid = 1
//cont_rowlen = 2
////cfg_fft_w4_0 --> start_rowid
////cfg_fft_w4_1 --> cont_rowlen
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
////DMA row data from Device to Sram
///dst_sram[0]: 0x2222,  0x2222, .....       0x2222
///dst_sram[1]: 0x3333,  0x3333, .....       0x3333
/////////////////////////////////////////////////////////////
LDPARAM.U32 [cfg_fft_w4_0], %start_rowid
LDPARAM.U32 [row_data_ddr_low], %src_low
LDPARAM.U32 [row_data_ddr_hi], %src_hi
LDPARAM.U32 [bytes_per_row], %dwReg0
MUL.U32.LOW %dwReg0, %start_rowid, DS13
ADD.U32 %src_low, DS13, %src_low

LDPARAM.U32 [dst_sram_low], %dst_low
LDPARAM.U32 [dst_sram_hi], %dst_hi

LDPARAM.U32 [cfg_fft_w4_1], %len
MUL.U32.LOW %dwReg0, %len, %len

STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
