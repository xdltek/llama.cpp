.param .u32 inputHigh
.param .u32 outputHigh
.param .u32 inputLow
.param .u32 outputLow
.param .u32 inBlockStride
.param .u32 outBlockStride
.param .u32 loopStride
.param .u32 avgScale
.param .u32 inStrideZ
.param .u16 loopNum
.param .u16 operation

.var .u32 %inputHigh
.var .u32 %outputHigh
.var .u32 %inputLow
.var .u32 %outputLow
.var .u32 %blockStride
.var .u32 %loopStride
.var .u32 %avgScale
.var .u32 %inStrideZ
.var .u16 %loopNum
.var .u16 %op
.var .u16 %sumOp
.var .u16 %prodOp
.var .u16 %maxOp
.var .u16 %minOp
.var .u16 %avgOp
.var .u16 %zero
.var .u16 %one

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %op
LDPARAM.U16 [tbDim.y], %sumOp
LDPARAM.U16 [tid_xyz], %prodOp
CFGXYZ %op, %sumOp, %prodOp

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
LDPARAM.U32 [outputHigh], %outputHigh
LDPARAM.U32 [inputLow], %inputLow
LDPARAM.U32 [outputLow], %outputLow
LDPARAM.U32 [loopStride], %loopStride
LDPARAM.U16 [loopNum], %loopNum
LDPARAM.U32 [inStrideZ], %inStrideZ
LDPARAM.U16 [operation], %op

CVT.U16.U32 %blockIdx, %avgScale
LDPARAM.U32 [inBlockStride], %blockStride
MUL.U32 %blockStride, %avgScale, %blockStride
ADD.U32 %inputHigh, %blockStride, %inputHigh
ADD.U32 %inputLow, %blockStride, %inputLow
LDPARAM.U32 [outBlockStride], %blockStride
MUL.U32 %blockStride, %avgScale, %blockStride
ADD.U32 %outputHigh, %blockStride, %outputHigh
ADD.U32 %outputLow, %blockStride, %outputLow

LDPARAM.U32 [avgScale], %avgScale
MOVE.U16 0, %zero
MOVE.U16 0, %sumOp
MOVE.U16 1, %prodOp
MOVE.U16 2, %maxOp
MOVE.U16 3, %minOp
MOVE.U16 4, %avgOp

CMP.U16.EQ %op, %sumOp, %one
JMPC %one, $SUM
CMP.U16.EQ %op, %prodOp, %one
JMPC %one, $PROD
CMP.U16.EQ %op, %maxOp, %one
JMPC %one, $MAX
CMP.U16.EQ %op, %minOp, %one
JMPC %one, $MIN
CMP.U16.EQ %op, %avgOp, %one
JMPC %one, $AVG

$SUM:
MOVE.U16 1, %one
MOVE.U32 0, VAB
##
$SUM_LOOP:
COPY VAB, IAB
LOADALN.Z32 %inputHigh, %zero, %inStrideZ, ID
##
LOADALN.Z32 %inputLow, %zero, %inStrideZ, IC
##
ADD.F32 ICD, IAB, VAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $SUM_LOOP
JMP $END
TERM VAB
##

$PROD:
MOVE.U16 1, %one
MOVE.U32 0x3f800000, VAB
##
$PROD_LOOP:
COPY VAB, IAB
LOADALN.Z32 %inputHigh, %zero, %inStrideZ, ID
##
LOADALN.Z32 %inputLow, %zero, %inStrideZ, IC
##
MUL.F32 ICD, IAB, VAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $PROD_LOOP
JMP $END
TERM VAB
##

$MAX:
MOVE.U16 1, %one
MOVE.U32 0xff800000, VAB
##
$MAX_LOOP:
COPY VAB, IAB
LOADALN.Z32 %inputHigh, %zero, %inStrideZ, ID
##
LOADALN.Z32 %inputLow, %zero, %inStrideZ, IC
##
MAX.F32 IAB, ICD, VAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MAX_LOOP
JMP $END
TERM VAB
##

$MIN:
MOVE.U16 1, %one
MOVE.U32 0x7f800000, VAB
##
$MIN_LOOP:
COPY VAB, IAB
LOADALN.Z32 %inputHigh, %zero, %inStrideZ, ID
##
LOADALN.Z32 %inputLow, %zero, %inStrideZ, IC
##
MIN.F32 IAB, ICD, VAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MIN_LOOP
JMP $END
TERM VAB
##

$AVG:
MOVE.U16 1, %one
MOVE.U32 0, VAB
##
$AVG_LOOP:
COPY VAB, IAB
LOADALN.Z32 %inputHigh, %zero, %inStrideZ, ID
##
LOADALN.Z32 %inputLow, %zero, %inStrideZ, IC
##
ADD.F32 ICD, IAB, VAB
##
ADD.U32 %inputHigh, %loopStride, %inputHigh
ADD.U32 %inputLow, %loopStride, %inputLow
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $AVG_LOOP
MUL.F32 VAB, %avgScale, VAB
##

$END:
COPY VAB, IAB
##
STORECONT %outputHigh, IB
##
STORECONT %outputLow, IA
##