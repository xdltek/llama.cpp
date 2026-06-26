.param .u32 input
.param .u32 output
.param .u32 inBlockStride
.param .u32 outBlockStride
.param .u32 inStrideY
.param .u32 inStrideZ
.param .u32 outStrideY
.param .u32 outStrideZ
.param .u32 loopStride
.param .u32 avgScale
.param .u16 loopNum
.param .u16 operation
.param .u32 reciprocal_table

.var .u32 %input
.var .u32 %output
.var .u32 %inBlockStride
.var .u32 %outBlockStride
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

LDPARAM.U32 [input], %input
LDPARAM.U32 [output], %output
LDPARAM.U16 [loopNum], %loopNum
LDPARAM.U16 [operation], %op

CVT.U16.U32 %blockIdx, %avgScale
LDPARAM.U32 [inBlockStride], %inBlockStride
MUL.U32 %avgScale, %inBlockStride, %inBlockStride
ADD.U32 %input, %inBlockStride, %input
LDPARAM.U32 [outBlockStride], %outBlockStride
MUL.U32 %avgScale, %outBlockStride, %avgScale
ADD.U32 %output, %avgScale, %output

LDPARAM.U32 [tid_base], %avgScale
LDPARAM.U32 [tid_depack], %loopStride
LOADCONT %avgScale, MB
DECTID.X MB, %loopStride, VA
##
SHL.U16 VA, %one, VA
##
CVT.U16.U32 VA, VAB
##
COPY VAB, ICD
LOADCONT %avgScale, MB
DECTID.Y MB, %loopStride, VA
##
CVT.U16.U32 VA, VAB
##
COPYEXT VAB, IEF
LOADCONT %avgScale, MB
DECTID.Z MB, %loopStride, VA
##
CVT.U16.U32 VA, VAB
##
COPYEXT VAB, IGH

MMOV ICD, DVA
LDPARAM.U32 [inStrideY], %avgScale
MUL.U32 IEF, %avgScale, VAB
##
MADD DVA, VAB, DVA
LDPARAM.U32 [inStrideZ], %avgScale
MUL.U32 IGH, %avgScale, VAB
##
MADD DVA, VAB, DVA
##
LDPARAM.U32 [loopStride], %loopStride
MADD.SCA32 DVA, %input, DVA
##

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
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
ADD.F32 VAB, IAB, VAB
##
COPY VAB, IAB
MADD.SCA32 DVA, %loopStride, DVA
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $SUM_LOOP
JMP $END

$PROD:
MOVE.U32 0x3f800000, VAB
##
COPY VAB, IAB
##
$PROD_LOOP:
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
MUL.F32 VAB, IAB, VAB
##
COPY VAB, IAB
MADD.SCA32 DVA, %loopStride, DVA
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $PROD_LOOP
JMP $END

$MAX:
MOVE.U32 0xff800000, VAB
##
COPY VAB, IAB
##
$MAX_LOOP:
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
MAX.F32 VAB, IAB, VAB
##
COPY VAB, IAB
MADD.SCA32 DVA, %loopStride, DVA
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MAX_LOOP
JMP $END

$MIN:
MOVE.U32 0x7f800000, VAB
##
COPY VAB, IAB
##
$MIN_LOOP:
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
MIN.F32 VAB, IAB, VAB
##
COPY VAB, IAB
MADD.SCA32 DVA, %loopStride, DVA
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MIN_LOOP
JMP $END

$AVG:
MOVE.U32 0, VAB
##
COPY VAB, IAB
##
$AVG_LOOP:
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
ADD.F32 VAB, IAB, VAB
##
COPY VAB, IAB
MADD.SCA32 DVA, %loopStride, DVA
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $AVG_LOOP
LDPARAM.U32 [avgScale], %avgScale
MUL.F32 IAB, %avgScale, VAB
##
COPY VAB, IAB
##
JMP $END

$END:
CVT.F32.F16 IAB, VA
##
MOVE.U16 0, %tmp
LDPARAM.U32 [reciprocal_table], %avgScale
LOAD [%avgScale + (VA << 1)], MA
ADD.U16 MA, %tmp, VA
##
STORECONT %output, VA
##
