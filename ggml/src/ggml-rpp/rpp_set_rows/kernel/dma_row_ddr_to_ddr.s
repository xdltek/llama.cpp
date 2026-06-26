//ENTRY:
.param .u32 row_data_ddr_low
.param .u32 row_data_ddr_hi
.param .u32 dst_ddr_low
.param .u32 dst_ddr_hi
.param .u32 rowid_ddr_low
.param .u32 rowid_ddr_hi
.param .u32 rowid_fram0_low
.param .u32 rowid_fram1_low
.param .u32 rowid_fram_hi
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
////cfg_fft_w4_4 --> start_rowid_value
/////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////
//row_data_ddr[0]: 0x1111,  0x1111, .....       0x1111
//row_data_ddr[1]: 0x2222,  0x2222, .....       0x2222
//row_data_ddr[2]: 0x3333,  0x3333, .....       0x3333
//row_data_ddr[3]: 0x4444,  0x4444, .....       0x4444
/////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////
//rowid_ddr[0]: 2
//rowid_ddr[1]: 3
/////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////
//Below 2 parameters set by warmup interface
//start_rowid = 1
//cont_rowlen = 2
////cfg_fft_w4_0 --> start_rowid
////cfg_fft_w4_1 --> cont_rowlen
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
////DMA start_rowid_value from DDR to FRAM
///start_rowid_value = rowid_ddr[start_rowid]
///start_rowid_value = 3
/////////////////////////////////////////////////////////////
LDPARAM.U32 [rowid_ddr_low], %src_low
LDPARAM.U32 [rowid_ddr_hi], %src_hi


LDPARAM.U32 [cfg_fft_w4_0], %start_rowid
/////////////////////////////////////////////////////////
//The rowid element is int64, it is why DS13 is 8
//////////////////////////////////////////////////////////
MOVE.U32 8, DS13
MUL.U32.LOW  %start_rowid, DS13, DS13
ADD.U32 %src_low, DS13, %src_low
MOVE.U32 4, %len
LDPARAM.U32 [rowid_fram0_low], %dst_low
LDPARAM.U32 [rowid_fram_hi], %dst_hi

STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
LDPARAM.U32 [rowid_fram1_low], %dst_low
STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
LDPARAM.U32 [cfg_fft_w4_4], %start_row_offset

/////////////////////////////////////////////////////////////
////DMA row data from Device to Device
///dst_ddr[start_row_offset] = row_data_ddr[start_rowid]
///start_rowid = 1
///start_row_offset = 3
///row_data_ddr[start_rowid]: 0x2222,  0x2222, .....       0x2222
///dst_ddr[0] xxxx, xxxx, xxxx, ....xxxx
///dst_ddr[1] xxxx, xxxx, xxxx, ....xxxx
///dst_ddr[2] xxxx, xxxx, xxxx, ....xxxx
///dst_ddr[3] 0x2222,  0x2222, ....0x2222
/////////////////////////////////////////////////////////////

LDPARAM.U32 [row_data_ddr_low], %src_low
LDPARAM.U32 [row_data_ddr_hi], %src_hi
LDPARAM.U32 [bytes_per_row], %dwReg0
MUL.U32.LOW %dwReg0, %start_rowid, DS13
ADD.U32 %src_low, DS13, %src_low

LDPARAM.U32 [dst_ddr_low], %dst_low
LDPARAM.U32 [dst_ddr_hi], %dst_hi
MUL.U32.LOW %dwReg0, %start_row_offset, DS13
ADD.U32 %dst_low, DS13, %dst_low

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
