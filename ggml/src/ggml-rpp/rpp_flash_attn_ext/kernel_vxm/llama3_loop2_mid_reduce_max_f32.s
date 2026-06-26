.param .u32 inputHigh
.param .u32 inputLow
.param .u32 outputHigh
.param .u32 outputLow
.param .u32 nLoopStride
.param .u32 blockLoopStride
.param .u16 inputRowStrideY
.param .u16 outputRowStrideY
.param .u16 blockRepeatNum
.param .u16 kv_page_repeat

.var .u32 %inputHigh
.var .u32 %inputLow
.var .u32 %outputHigh
.var .u32 %outputLow
.var .u32 %nLoopStride
.var .u32 %blockLoopStride
.var .u32 %tmp
.var .u32 %tmp1
.var .u16 %nRepeatNum
.var .u16 %blockRepeatNum
.var .u16 %inputRowStrideY
.var .u16 %outputRowStrideY
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

LDPARAM.U32 [inputHigh], %inputHigh
LDPARAM.U32 [inputLow], %inputLow
LDPARAM.U32 [outputHigh], %outputHigh
LDPARAM.U32 [outputLow], %outputLow
LDPARAM.U32 [nLoopStride], %nLoopStride
LDPARAM.U32 [blockLoopStride], %blockLoopStride
LDPARAM.U16 [inputRowStrideY], %inputRowStrideY
LDPARAM.U16 [outputRowStrideY], %outputRowStrideY
LDPARAM.U16 [blockRepeatNum], %blockRepeatNum
MOVE.U16 1, %one

MOVE.U16 0, %blockLoopIdx
$BLOCK_LOOP:

LDPARAM.U16 [kv_page_repeat], %nRepeatNum
MOVE.U16 0, %nLoopIdx
MOVE.U32 0xff7fffff, VAB
##
COPY VAB, IAB
##
$N_LOOP:

CVT.U16.U32 %nLoopIdx, %tmp
MUL.U32 %tmp, %nLoopStride, %tmp
CVT.U16.U32 %blockLoopIdx, %tmp1
MUL.U32 %tmp1, %blockLoopStride, %tmp1
ADD.U32 %tmp, %tmp1, %tmp

LDPARAM.U32 [inputHigh], %inputHigh
LDPARAM.U32 [inputLow], %inputLow
ADD.U32 %inputHigh, %tmp, %inputHigh
ADD.U32 %inputLow, %tmp, %inputLow

LOADALN %inputLow, %inputRowStrideY, %inputRowStrideY, IC
##
LOADALN %inputHigh, %inputRowStrideY, %inputRowStrideY, ID
##
MAX.F32 IAB, ICD, VAB
##
COPY VAB, IAB
##

ADD.U16 %nLoopIdx, %one, %nLoopIdx
SUB.U16 %nRepeatNum, %one, %nRepeatNum
JMPC %nRepeatNum, $N_LOOP

LDPARAM.U32 [outputHigh], %outputHigh
LDPARAM.U32 [outputLow], %outputLow
ADD.U32 %outputHigh, %tmp1, %outputHigh
ADD.U32 %outputLow, %tmp1, %outputLow

STOREALN %outputLow, %outputRowStrideY, %outputRowStrideY, IA
##
STOREALN %outputHigh, %outputRowStrideY, %outputRowStrideY, IB
##

ADD.U16 %blockLoopIdx, %one, %blockLoopIdx
SUB.U16 %blockRepeatNum, %one, %blockRepeatNum
JMPC %blockRepeatNum, $BLOCK_LOOP
##
