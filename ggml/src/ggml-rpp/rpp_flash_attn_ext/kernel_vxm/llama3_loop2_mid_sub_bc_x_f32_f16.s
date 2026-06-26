.param .u32 inputHigh
.param .u32 inputLow
.param .u32 input1High
.param .u32 input1Low
.param .u32 output
.param .u32 nLoopStride
.param .u32 blockLoopStride
.param .u32 in1BlockLoopStride
.param .u16 rowStrideY
.param .u16 blockRepeatNum
.param .u16 kv_page_repeat

.var .u32 %inAddrHigh
.var .u32 %inAddrLow
.var .u32 %output
.var .u32 %nLoopStride
.var .u32 %blockLoopStride
.var .u32 %in1BlockLoopStride
.var .u32 %tmp
.var .u32 %tmp1
.var .u32 %tmp2
.var .u16 %nRepeatNum
.var .u16 %blockRepeatNum
.var .u16 %rowStrideY
.var .u16 %nLoopIdx
.var .u16 %blockLoopIdx
.var .u16 %one

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %blockRepeatNum
LDPARAM.U16 [tbDim.y], %nLoopIdx
LDPARAM.U16 [tid_xyz], %blockLoopIdx
CFGXYZ %blockRepeatNum, %nLoopIdx, %blockLoopIdx

$BLOCK_ENTRY:
MMOV 0, DVA
MOVE.U32 0, VAB
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##

LDPARAM.U32 [output], %output
LDPARAM.U32 [nLoopStride], %nLoopStride
LDPARAM.U32 [blockLoopStride], %blockLoopStride
LDPARAM.U32 [in1BlockLoopStride], %in1BlockLoopStride
LDPARAM.U16 [rowStrideY], %rowStrideY
LDPARAM.U16 [kv_page_repeat], %nRepeatNum
MOVE.U16 1, %one

LDPARAM.U32 [tid_base], %tmp
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp, MB
DECTID.Y MB, %tmp1, VA
##
COPY VA, IC
##

MOVE.U16 0, %nLoopIdx
$N_LOOP:

LDPARAM.U16 [blockRepeatNum], %blockRepeatNum
MOVE.U16 0, %blockLoopIdx
$BLOCK_LOOP:

CVT.U16.U32 %nLoopIdx, %tmp
MUL.U32 %tmp, %nLoopStride, %tmp
CVT.U16.U32 %blockLoopIdx, %tmp1
MUL.U32 %tmp1, %in1BlockLoopStride, %tmp2
MUL.U32 %tmp1, %blockLoopStride, %tmp1
ADD.U32 %tmp, %tmp1, %tmp

LDPARAM.U32 [inputHigh], %inAddrHigh
ADD.U32 %inAddrHigh, %tmp, %inAddrHigh
LOADALN %inAddrHigh, %rowStrideY, %rowStrideY, IB
##
LDPARAM.U32 [inputLow], %inAddrLow
ADD.U32 %inAddrLow, %tmp, %inAddrLow
LOADALN %inAddrLow, %rowStrideY, %rowStrideY, IA
##
LDPARAM.U32 [input1High], %inAddrHigh
ADD.U32 %inAddrHigh, %tmp2, %inAddrHigh
LOAD [%inAddrHigh + (IC << 1)], IF
##
LDPARAM.U32 [input1Low], %inAddrLow
ADD.U32 %inAddrLow, %tmp2, %inAddrLow
LOAD [%inAddrLow + (IC << 1)], IE
##
SUB.F32 IAB, IEF, VAB
##
CVT.F32.F16 VAB, VA
##
ADD.U32 %output, %tmp, %tmp
STOREALN %tmp, %rowStrideY, %rowStrideY, VA
##
ADD.U16 %blockLoopIdx, %one, %blockLoopIdx
SUB.U16 %blockRepeatNum, %one, %blockRepeatNum
JMPC %blockRepeatNum, $BLOCK_LOOP

ADD.U16 %nLoopIdx, %one, %nLoopIdx
SUB.U16 %nRepeatNum, %one, %nRepeatNum
JMPC %nRepeatNum, $N_LOOP
##
