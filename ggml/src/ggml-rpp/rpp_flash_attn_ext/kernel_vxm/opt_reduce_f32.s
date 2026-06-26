.param .u32 inputHigh
.param .u32 outputHigh
.param .u32 inputLow
.param .u32 outputLow
.param .u32 inBlockStride
.param .u32 outBlockStride
.param .u32 loopStride
.param .u32 avgScale
.param .u16 loopNum
.param .u16 operation

.var .u32 %inputHigh
.var .u32 %inputLow
.var .u32 %outputHigh
.var .u32 %outputLow
.var .u32 %blockStride
.var .u32 %loopStride
.var .u32 %avgScale
.var .u16 %one
.var .u16 %loopNum
.var .u16 %op
.var .u16 %sumOp
.var .u16 %prodOp
.var .u16 %maxOp
.var .u16 %minOp
.var .u16 %avgOp
.var .u16 %tmp

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %one
LDPARAM.U16 [tbDim.y], %loopNum
LDPARAM.U16 [tid_xyz], %op
CFGXYZ %one, %loopNum, %op

$BLOCK_ENTRY:
MMOV 0, DVA
MOVE.U16 0, VA
##
COPY VA, IA
COPY VA, IB
COPY VA, IC
COPY VA, ID
COPYEXT VA, IE
COPYEXT VA, IF
COPYEXT VA, IG
COPYEXT VA, IH
##

MOVE.U16 1, %one
MOVE.U16 0, %sumOp
MOVE.U16 1, %prodOp
MOVE.U16 2, %maxOp
MOVE.U16 3, %minOp
MOVE.U16 4, %avgOp

LDPARAM.U32 [inputHigh], %inputHigh
LDPARAM.U32 [inputLow], %inputLow
LDPARAM.U32 [outputHigh], %outputHigh
LDPARAM.U32 [outputLow], %outputLow
LDPARAM.U32 [loopStride], %loopStride
LDPARAM.U16 [loopNum], %loopNum
LDPARAM.U16 [operation], %op

CVT.U16.U32 %blockIdx, %avgScale
LDPARAM.U32 [inBlockStride], %blockStride
MUL.U32 %avgScale, %blockStride, %blockStride
ADD.U32 %inputHigh, %blockStride, %inputHigh
ADD.U32 %inputLow, %blockStride, %inputLow
LDPARAM.U32 [outBlockStride], %blockStride
MUL.U32 %avgScale, %blockStride, %blockStride
ADD.U32 %outputHigh, %blockStride, %outputHigh
ADD.U32 %outputLow, %blockStride, %outputLow

CMP.U16.EQ %op, %sumOp, %tmp
JMPC %tmp, $SUM
CMP.U16.EQ %op, %prodOp, %tmp
JMPC %tmp, $PROD
CMP.U16.EQ %op, %maxOp, %tmp
JMPC %tmp, $MAX
CMP.U16.EQ %op, %minOp, %tmp
JMPC %tmp, $MIN
CMP.U16.EQ %op, %avgOp, %tmp
JMPC %tmp, $AVG

$SUM:
MOVE.U32 0, VAB
##
COPY VAB, IAB
##
$SUM_LOOP:
LOADCONT %inputHigh, ID
##
LOADCONT %inputLow, IC
##
ADD.F32 ICD, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $SUM_LOOP
JMP $END

$PROD:
MOVE.U32 0x3f800000, VAB
##
COPY VAB, IAB
##
$PROD_LOOP:
LOADCONT %inputHigh, ID
##
LOADCONT %inputLow, IC
##
MUL.F32 ICD, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $PROD_LOOP
JMP $END

$MAX:
MOVE.U32 0xff800000, VAB
##
COPY VAB, IAB
##
$MAX_LOOP:
LOADCONT %inputHigh, ID
##
LOADCONT %inputLow, IC
##
MAX.F32 ICD, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MAX_LOOP
JMP $END

$MIN:
MOVE.U32 0x7f800000, VAB
##
COPY VAB, IAB
##
$MIN_LOOP:
LOADCONT %inputHigh, ID
##
LOADCONT %inputLow, IC
##
MIN.F32 ICD, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MIN_LOOP
JMP $END

$AVG:
MOVE.U32 0, VAB
##
COPY VAB, IAB
##
$AVG_LOOP:
LOADCONT %inputHigh, ID
##
LOADCONT %inputLow, IC
##
ADD.F32 ICD, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $AVG_LOOP
LDPARAM.U32 [avgScale], %avgScale
MUL.F32 IAB, %avgScale, VAB
##
COPY VAB, IAB
##
JMP $END

$END:
STORECONT %outputHigh, IB
##
STORECONT %outputLow, IA
##
