.param .u32 input
.param .u32 output
.param .u32 inBlockStride
.param .u32 outBlockStride
.param .u32 loopStride
.param .u32 avgScale
.param .u32 inStrideZ
.param .u16 loopNum
.param .u16 operation

.var .u32 %input
.var .u32 %output
.var .u32 %inBlockStride
.var .u32 %outBlockStride
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

LDPARAM.U32 [input], %input
LDPARAM.U32 [output], %output
LDPARAM.U32 [inBlockStride], %inBlockStride
LDPARAM.U32 [outBlockStride], %outBlockStride
LDPARAM.U32 [loopStride], %loopStride
LDPARAM.U16 [loopNum], %loopNum
LDPARAM.U32 [inStrideZ], %inStrideZ
LDPARAM.U16 [operation], %op

CVT.U16.U32 %blockIdx, %avgScale
MUL.U32 %inBlockStride, %avgScale, %inBlockStride
MUL.U32 %outBlockStride, %avgScale, %outBlockStride
ADD.U32 %input, %inBlockStride, %input
ADD.U32 %output, %outBlockStride, %output

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
LOADALN.Z32 %input, %zero, %inStrideZ, MB
CVT.F16.F32 MB, VAB
##
ADD.F32 VAB, IAB, VAB
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $SUM_LOOP
JMP $END_CAST
TERM VAB
##
$PROD:
MOVE.U16 1, %one
MOVE.U32 0x3f800000, VAB
##
$PROD_LOOP:
COPY VAB, IAB
LOADALN.Z32 %input, %zero, %inStrideZ, MB
CVT.F16.F32 MB, VAB
##
MUL.F32 VAB, IAB, VAB
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $PROD_LOOP
JMP $END_CAST
TERM VAB
##
$MAX:
MOVE.U16 1, %one
MOVE.U16 0xff80, VA
##
$MAX_LOOP:
LOADALN.Z32 %input, %zero, %inStrideZ, MB
MAX.F16 MB, VA, VA
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MAX_LOOP
JMP $END
TERM VA
##
$MIN:
MOVE.U16 1, %one
MOVE.U16 0x7f80, VA
##
$MIN_LOOP:
LOADALN.Z32 %input, %zero, %inStrideZ, MB
MIN.F16 MB, VA, VA
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MIN_LOOP
JMP $END
TERM VA
##
$AVG:
MOVE.U16 1, %one
MOVE.U32 0, VAB
##
$AVG_LOOP:
COPY VAB, IAB
LOADALN.Z32 %input, %zero, %inStrideZ,MB
CVT.F16.F32 MB, VAB
##
ADD.F32 VAB, IAB, VAB
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $AVG_LOOP
MUL.F32 VAB, %avgScale, VAB
##
$END_CAST:
CVT.F32.F16 VAB, VA
##
$END:
STORECONT %output, VA
##
