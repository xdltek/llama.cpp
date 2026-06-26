//ENTRY:
.param .u32 input0
.param .u32 input1
.param .u32 out
.param .u32 outLow
.param .u32 outStrideZ
.param .u32 offset0
.param .u32 offset1
.param .u16 in0StrideZ
.param .u16 in1StrideY
.param .u16 outStrideY
.param .u16 channel


.var .u32 %input0
.var .u32 %input1
.var .u32 %out
.var .u32 %outLow
.var .u32 %offset
.var .u32 %blockOffset
.var .u32 %outStrideZ
.var .u16 %outStrideY
.var .u16 %in1StrideY
.var .u16 %channel
.var .u16 %one
.var .u16 %zero
.var .u16 %sreg0
.var .u16 %sreg1



LDPARAM.U16 [tid_xyz], %sreg0
LDPARAM.U16 [tbDim.x], %zero
LDPARAM.U16 [tbDim.y], %sreg1
CFGXYZ %zero, %sreg1, %sreg0

MOVE.U32 0, VAB
MMOV 0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##

//-------------------------------------------------------------------------
//in0 [expand][D]  ==> query
//in1 [D][N][kv_page]  ==> past key & current key
//out [expand][N][kv_page]
//--------------------------------------------------------------------------
// in0  [expand]  |  [D]
//      [z]       |
//--------------------------------------------------------------------------
// in1  [D]       |  [N]
//                |  [grid.x][y]   |  [x]
//--------------------------------------------------------------------------
//out   [expand]  |  [N]
//      [z]       |  [grid.x][y]   |  [x]
//--------------------------------------------------------------------------

LDPARAM.U32 [tid_base], %input0
LDPARAM.U32 [tid_depack], %input1
LOADCONT %input0, MB
DECTID.Z MB, %input1, VA
##
LDPARAM.U16 [in0StrideZ], %sreg0
MUL.U16.LOW VA, %sreg0, VA
##
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [out], %out
LDPARAM.U32 [outLow], %outLow
LDPARAM.U16 [channel], %channel
MOVE.U16 1, %one
MOVE.U16 0, %zero
LDPARAM.U32 [outStrideZ], %outStrideZ
LDPARAM.U16 [outStrideY], %outStrideY
LDPARAM.U16 [in1StrideY], %in1StrideY
CVT.U16.U32 %blockIdx, %one
MUL.U16.LOW %one, %sreg1, %one
MUL.U16.WIDE %one, %in1StrideY, %blockOffset
ADD.U32 %blockOffset, %blockOffset, %blockOffset
ADD.U32 %input1, %blockOffset, %input1
ADD.U32 %out, %blockOffset, %out
ADD.U32 %outLow, %blockOffset, %outLow
MOVE.U16 1, %one
LDPARAM.U16 [offset0], %sreg0
LDPARAM.U32 [offset1], %offset

MSHIFTADD VA, %input0, 1, DVA
MOVE.U32 0, VAB
##
$BLOCK_ENTRY_0:
LOADALN %input1, %in1StrideY, %zero, MB
LOAD [DVA], MA
MADD DVA, %sreg0, DVA
HFMA MA, MB, VAB, VAB
##
SUB.U16 %channel, %one, %channel
ADD.U32 %input1, %offset, %input1
JMPC %channel, $BLOCK_ENTRY_0
STOREALN.Z32 %outLow, %outStrideY, %outStrideZ, VA
MOVE.U16 VB, VA
##
STOREALN.Z32 %out, %outStrideY, %outStrideZ, VA
##
