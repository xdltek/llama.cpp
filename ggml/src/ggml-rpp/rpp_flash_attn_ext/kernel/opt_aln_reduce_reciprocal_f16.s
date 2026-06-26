.param .u32 input
.param .u32 output
.param .u32 inBlockStride
.param .u32 outBlockStride
.param .u32 loopStride
.param .u32 avgScale
.param .u16 loopNum
.param .u16 inStrideY
.param .u32 operation
.param .u32 reciprocal_table

.var .u32 %input
.var .u32 %output
.var .u32 %inBlockStride
.var .u32 %outBlockStride
.var .u32 %loopStride
.var .u32 %avgScale
.var .u16 %loopNum
.var .u16 %inStrideY
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
LDPARAM.U16 [inStrideY], %inStrideY
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
$SUM_LOOP:
LOADALN %input, %inStrideY, %zero, MB
CVT.F16.F32 MB, VAB
##
ADD.F32 VAB, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $SUM_LOOP
CVT.F32.F16 IAB, VA
##
COPY VA, IA
##
JMP $END

$PROD:
MOVE.U16 1, %one
MOVE.U32 0x3f800000, VAB
##
COPY VAB, IAB
##
$PROD_LOOP:
LOADALN %input, %inStrideY, %zero, MB
CVT.F16.F32 MB, VAB
##
MUL.F32 VAB, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $PROD_LOOP
CVT.F32.F16 IAB, VA
##
COPY VA, IA
##
JMP $END

$MAX:
MOVE.U16 1, %one
MOVE.U16 0xff80, VA
##
COPY VA, IA
##
$MAX_LOOP:
LOADALN %input, %inStrideY, %zero, MB
ADD.U16 MB, %zero, VA
##
MAX.F16 VA, IA, VA
##
COPY VA, IA
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MAX_LOOP
JMP $END

$MIN:
MOVE.U16 1, %one
MOVE.U16 0x7f80, VA
##
COPY VA, IA
##
$MIN_LOOP:
LOADALN %input, %inStrideY, %zero, MB
ADD.U16 MB, %zero, VA
##
MIN.F16 VA, IA, VA
##
COPY VA, IA
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MIN_LOOP
JMP $END

$AVG:
MOVE.U16 1, %one
$AVG_LOOP:
LOADALN %input, %inStrideY, %zero, MB
CVT.F16.F32 MB, VAB
##
ADD.F32 VAB, IAB, VAB
##
COPY VAB, IAB
##
ADD.U32 %input, %loopStride, %input
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $AVG_LOOP
MUL.F32 IAB, %avgScale, VAB
##
CVT.F32.F16 VAB, VA
##
COPY VA, IA
##
JMP $END

$END:
LDPARAM.U32 [reciprocal_table], %avgScale
LOAD [%avgScale + (IA << 1)], MA
ADD.U16 MA, %zero, VA
##
STORECONT %output, VA
##
