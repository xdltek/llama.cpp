//ENTRY:
.param .u32 row_data_ddr_low
.param .u32 row_data_ddr_hi
.param .u32 src_ddr_low
.param .u32 src_ddr_hi
.param .u32 rowid_ddr_low
.param .u32 rowid_ddr_hi
.param .u32 rowid_fram0_low
.param .u32 rowid_fram1_low
.param .u32 rowid_fram_hi
.param .u32 bytes_per_row
.param .u32 bytes_per_rowid

.var .u32 %src_low
.var .u32 %src_hi
.var .u32 %dst_low
.var .u32 %dst_hi
.var .u32 %len
.var .u32 %start_rowid
.var .u32 %start_row_offset
.var .u32 %dwReg0
.var .u32 %bytes_per_rowid
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2

LDPARAM.U16 [tid_xyz], %sreg0 
LDPARAM.U16 [tbDim.x], %sreg1
LDPARAM.U16 [tbDim.y], %sreg2
CFGXYZ %sreg1, %sreg2, %sreg0

LDPARAM.U32 [bytes_per_rowid], %bytes_per_rowid

//------------------------------------------------------------------------
//CDMA the rowid from DDR into FRAM
//start_rowid indicates the start position in rowid_ddr[] to be transfered
//rowid_fram_base & cfg_fft_w4_4 indicates the FRAM offset to store rowid

//CDMA the rowdata from src_ddr DDR to row_data_ddr DDR
//LDPARAM.U32 [cfg_fft_w4_4], %start_row_offset --> this indicate the src_data row offset
//------------------------------------------------------------------------
//cfg_fft_w4_2 --> start_rowid
//cfg_fft_w4_3 --> cont_rowlen
//cfg_fft_w4_4 --> rowid_fram_base
//------------------------------------------------------------------------
//------------------------------------------------------------------------
//Below 2 parameters set by warmup interface
//start_rowid = 1
//cont_rowlen = 2
////cfg_fft_w4_2 --> start_rowid
////cfg_fft_w4_3 --> cont_rowlen
//------------------------------------------------------------------------

/////////////////////////////////////////////////////////////
////DMA start_rowid_value from DDR to FRAM
///start_rowid_value = rowid_ddr[start_rowid]
///start_rowid_value = 3
/////////////////////////////////////////////////////////////
LDPARAM.U32 [rowid_ddr_low], %src_low
LDPARAM.U32 [rowid_ddr_hi], %src_hi

LDPARAM.U32 [cfg_fft_w4_2], %start_rowid
/////////////////////////////////////////////////////////
//The rowid element is int64, it is why DS13 is 8
//////////////////////////////////////////////////////////
MUL.U32.LOW  %start_rowid, %bytes_per_rowid, DS13
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

LDPARAM.U32 [row_data_ddr_low], %dst_low
LDPARAM.U32 [row_data_ddr_hi], %dst_hi
LDPARAM.U32 [bytes_per_row], %dwReg0
MUL.U32.LOW %dwReg0, %start_rowid, DS13
ADD.U32 %dst_low, DS13, %dst_low

LDPARAM.U32 [src_ddr_low], %src_low
LDPARAM.U32 [src_ddr_hi], %src_hi
MUL.U32.LOW %dwReg0, %start_row_offset, DS13
ADD.U32 %src_low, DS13, %src_low

LDPARAM.U32 [cfg_fft_w4_3], %len
MUL.U32.LOW %dwReg0, %len, %len

STGLB.U32 %src_low, [dma_ch0_mod1_src_lo]
STGLB.U32 %src_hi, [dma_ch0_mod1_src_hi]
STGLB.U32 %dst_low, [dma_ch0_mod1_dst_lo]
STGLB.U32 %dst_hi, [dma_ch0_mod1_dst_hi]
STGLB.U32 %len, [dma_ch0_mod1_len]
##
POLL.CHN1
##
