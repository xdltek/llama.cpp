.param .u32 input
.param .u32 output
.param .u32 inBlockStride
.param .u32 outBlockStride
.param .u32 inRowSize
.param .u16 payloadCLen
.param .u16 padValue

.var .u32 %input
.var .u32 %output
.var .u32 %inBlockStride
.var .u32 %outBlockStride
.var .u32 %inRowSize
.var .u32 %tmp0
.var .u32 %tmp1
.var .u16 %payloadCLen
.var .u16 %padValue
.var .u16 %tmp2
.var .u16 %one
.var .u16 %mask
.var .u16 %seven
.var .u16 %zero

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %payloadCLen
LDPARAM.U16 [tbDim.y], %padValue
LDPARAM.U16 [tid_xyz], %tmp2
CFGXYZ %payloadCLen, %padValue, %tmp2

LDPARAM.U32 [inBlockStride], %inBlockStride
LDPARAM.U32 [outBlockStride], %outBlockStride
LDPARAM.U32 [inRowSize], %inRowSize
LDPARAM.U16 [payloadCLen], %payloadCLen
LDPARAM.U16 [padValue], %padValue
MOVE.U16 2, %tmp2
MOVE.U16 1, %one
MOVE.U16 0, %zero

$BLOCK_ENTRY:
LDPARAM.U32 [input], %input
LDPARAM.U32 [output], %output
CVT.U16.U32 %blockIdx, %tmp0
MUL.U32 %inBlockStride, %tmp0, %tmp1
ADD.U32 %input, %tmp1, %input
MUL.U32 %outBlockStride, %tmp0, %tmp1
ADD.U32 %output, %tmp1, %output

MMOV 0, DVA
MOVE.U16 %padValue, VA
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

LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, MB
DECTID.X MB, %tmp1, VA
##
COPY VA, IA
LOADCONT %tmp0, MB
DECTID.Y MB, %tmp1, VA
##
MUL.U16.WIDE VA, %tmp2, VAB
##
COPY VAB, ICD
CVT.U16.U32 IA, VAB
##
MUL.U32 VAB, %inRowSize, VAB
##
ADD.U32 VAB, ICD, VAB
##
MADD.SCA32 VAB, %input, DVA
CMP.U16.LT IA, %payloadCLen, VA
##
COPY VA, IB

MOVE.U16 7, %seven
MOVE.U16 1, %mask
LOADMASK [DVA], %mask, MA
ADD.U16 MA, %zero, VA
##
$LOAD_LOOP:
SHL.U16 %mask, %one, %mask
LOADMASK [DVA], %mask, MA
MERGEMASK VA, MA, %mask, VA
##
SUB.U16 %seven, %one, %seven
JMPC %seven, $LOAD_LOOP

MERGE IH, VA, IB, VA
##
STORECONT %output, VA
##
