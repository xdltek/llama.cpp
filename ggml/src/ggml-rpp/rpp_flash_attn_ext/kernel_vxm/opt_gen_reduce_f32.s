.param .u32 inputHigh
.param .u32 outputHigh
.param .u32 inputLow
.param .u32 outputLow
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

.var .u32 %inputHigh
.var .u32 %outputHigh
.var .u32 %inputLow
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
LDPARAM.U32 [outputHigh], %outputHigh
LDPARAM.U32 [inputLow], %inputLow
LDPARAM.U32 [outputLow], %outputLow
LDPARAM.U16 [loopNum], %loopNum
LDPARAM.U16 [operation], %op

CVT.U16.U32 %blockIdx, %avgScale

LDPARAM.U32 [inBlockStride], %blockStride
MUL.U32 %avgScale, %blockStride, %blockStride
ADD.U32 %inputHigh, %blockStride, %inputHigh
ADD.U32 %inputLow, %blockStride, %inputLow

LDPARAM.U32 [outBlockStride], %blockStride
MUL.U32 %avgScale, %blockStride, %avgScale
ADD.U32 %outputHigh, %avgScale, %outputHigh
ADD.U32 %outputLow, %avgScale, %outputLow

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

LDPARAM.U32 [inStrideY], %avgScale
MUL.U32 IEF, %avgScale, VAB
##
ADD.U32 ICD, VAB, VAB
##
COPY VAB, ICD
LDPARAM.U32 [inStrideZ], %avgScale
MUL.U32 IGH, %avgScale, VAB
##
ADD.U32 ICD, VAB, VAB
##
COPY VAB, ICD
##

LDPARAM.U32 [loopStride], %loopStride
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
$SUM_LOOP:
COPY VAB, IAB
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputHigh, DVA
##
LOAD [DVA], IF
##
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputLow, DVA
##
LOAD [DVA], IE
ADD.U32 ICD, %loopStride, VAB
##
COPY VAB, ICD
ADD.F32 IEF, IAB, VAB
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $SUM_LOOP
JMP $END
TERM VAB
##

$PROD:
MOVE.U32 0x3f800000, VAB
##
$PROD_LOOP:
COPY VAB, IAB
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputHigh, DVA
##
LOAD [DVA], IF
##
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputLow, DVA
##
LOAD [DVA], IE
ADD.U32 ICD, %loopStride, VAB
##
COPY VAB, ICD
MUL.F32 IEF, IAB, VAB
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $PROD_LOOP
JMP $END
TERM VAB
##

$MAX:
MOVE.U32 0xff800000, VAB
##
$MAX_LOOP:
COPY VAB, IAB
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputHigh, DVA
##
LOAD [DVA], IF
##
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputLow, DVA
##
LOAD [DVA], IE
ADD.U32 ICD, %loopStride, VAB
##
COPY VAB, ICD
MAX.F32 IEF, IAB, VAB
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MAX_LOOP
JMP $END
TERM VAB
##

$MIN:
MOVE.U32 0x7f800000, VAB
##
$MIN_LOOP:
COPY VAB, IAB
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputHigh, DVA
##
LOAD [DVA], IF
##
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputLow, DVA
##
LOAD [DVA], IE
ADD.U32 ICD, %loopStride, VAB
##
COPY VAB, ICD
MIN.F32 IEF, IAB, VAB
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $MIN_LOOP
JMP $END
TERM VAB
##

$AVG:
MOVE.U32 0, VAB
##
$AVG_LOOP:
COPY VAB, IAB
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputHigh, DVA
##
LOAD [DVA], IF
##
MMOV ICD, DVA
##
MADD.SCA32 DVA, %inputLow, DVA
##
LOAD [DVA], IE
ADD.U32 ICD, %loopStride, VAB
##
COPY VAB, ICD
ADD.F32 IEF, IAB, VAB
##
SUB.U16 %loopNum, %one, %loopNum
JMPC %loopNum, $AVG_LOOP
LDPARAM.U32 [avgScale], %avgScale
MUL.F32 VAB, %avgScale, VAB
##

$END:
COPY VAB, IAB
##
STORECONT %outputHigh, IB
##
STORECONT %outputLow, IA
##
