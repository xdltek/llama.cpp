.param .u32 inputHigh
.param .u32 inputLow
.param .u32 input1
.param .u32 outputHigh
.param .u32 outputLow
.param .u32 nLoopStride
.param .u32 blockLoopStride
.param .u16 rowStrideY
.param .u32 in1NLoopStride
.param .u16 blockRepeatNum
.param .u16 kv_page_repeatNum


.var .u32 %inputAddr
.var .u32 %input1
.var .u32 %outputAddr
.var .u32 %nLoopStride
.var .u32 %blockLoopStride
.var .u32 %in1NLoopStride
.var .u32 %tmp
.var .u32 %tmp1
.var .u32 %tmp2
.var .u16 %tmp3
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

LDPARAM.U32 [input1], %input1
LDPARAM.U32 [nLoopStride], %nLoopStride
LDPARAM.U32 [blockLoopStride], %blockLoopStride
LDPARAM.U16 [rowStrideY], %rowStrideY
LDPARAM.U32 [in1NLoopStride], %in1NLoopStride
LDPARAM.U16 [kv_page_repeatNum], %nRepeatNum
MOVE.U16 1, %one

MOVE.U16 0, %nLoopIdx
$N_LOOP:

LDPARAM.U16 [blockRepeatNum], %blockRepeatNum
MOVE.U16 0, %blockLoopIdx
$BLOCK_LOOP:

CVT.U16.U32 %nLoopIdx, %tmp
MUL.U32 %tmp, %in1NLoopStride, %tmp2
MUL.U32 %tmp, %nLoopStride, %tmp
CVT.U16.U32 %blockLoopIdx, %tmp1
MUL.U32 %tmp1, %blockLoopStride, %tmp1
ADD.U32 %tmp, %tmp1, %tmp

LDPARAM.U32 [inputHigh], %inputAddr
ADD.U32 %inputAddr, %tmp, %inputAddr
LOADALN %inputAddr, %rowStrideY, %rowStrideY, IB
##
LDPARAM.U32 [inputLow], %inputAddr
ADD.U32 %inputAddr, %tmp, %inputAddr
LOADALN %inputAddr, %rowStrideY, %rowStrideY, IA
##
ADD.U32 %input1, %tmp2, %tmp2
MOVE.U16 0, %tmp3
LOADALN %tmp2, %tmp3, %tmp3, MB
CVT.F16.F32 MB, VAB
##
ADD.F32 VAB, IAB, VAB
##
COPY VAB, IAB
##
LDPARAM.U32 [outputHigh], %outputAddr
ADD.U32 %outputAddr, %tmp, %outputAddr
STOREALN %outputAddr, %rowStrideY, %rowStrideY, IB
##
LDPARAM.U32 [outputLow], %outputAddr
ADD.U32 %outputAddr, %tmp, %outputAddr
STOREALN %outputAddr, %rowStrideY, %rowStrideY, IA
##

ADD.U16 %blockLoopIdx, %one, %blockLoopIdx
SUB.U16 %blockRepeatNum, %one, %blockRepeatNum
JMPC %blockRepeatNum, $BLOCK_LOOP

ADD.U16 %nLoopIdx, %one, %nLoopIdx
SUB.U16 %nRepeatNum, %one, %nRepeatNum
JMPC %nRepeatNum, $N_LOOP
##
