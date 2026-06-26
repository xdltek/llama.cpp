// Only support ADD, SUB and MUL
.param .u32 input0
.param .u32 input1
.param .u32 output
.param .u32 loopStride
.param .u16 repeatNum

.var .u32 %input0
.var .u32 %input1
.var .u32 %output
.var .u32 %loopStride
.var .u16 %repeatNum
.var .u16 %operation
.var .u16 %one
.var .u16 %sumOp
.var .u16 %subOp
.var .u16 %prodOp
.var .u16 %tmp

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %repeatNum
LDPARAM.U16 [tbDim.y], %operation
LDPARAM.U16 [tid_xyz], %one
CFGXYZ %repeatNum, %operation, %one

$BLOCK_ENTRY:
MMOV 0, DVA
MOVE.U32 0, VAB
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##

MOVE.U16 1, %one

LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [output], %output
LDPARAM.U32 [loopStride], %loopStride
LDPARAM.U16 [repeatNum], %repeatNum

$SUM_LOOP:
LOADCONT %input0, MB
CVT.F16.F32 MB, VAB
##
COPY VAB, IAB
LOADCONT %input1, MB
CVT.F16.F32 MB, VAB
##
ADD.F32 IAB, VAB, VAB
##
CVT.F32.F16 VAB, VA
##
STORECONT %output, VA
##
ADD.U32 %input0, %loopStride, %input0
ADD.U32 %input1, %loopStride, %input1
ADD.U32 %output, %loopStride, %output
SUB.U16 %repeatNum, %one, %repeatNum
JMPC %repeatNum, $SUM_LOOP
##
