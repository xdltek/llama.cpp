.param .u32 input0
.param .u32 input1
.param .u32 output
.param .u32 loopStride
.param .u16 repeatNum

.var .u32 %input0
.var .u32 %input1
.var .u32 %output
.var .u32 %loopStride
.var .u32 %tmp1
.var .u32 %tmp2
.var .u16 %repeatNum
.var .u16 %one
.var .u16 %tmp0

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %repeatNum
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %tmp0
CFGXYZ %repeatNum, %one, %tmp0

$BLOCK_ENTRY:
MMOV 0, DVA
MOVE.U32 0, VAB
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##

LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [output], %output
LDPARAM.U32 [loopStride], %loopStride
LDPARAM.U16 [repeatNum], %repeatNum
MOVE.U16 1, %one

LDPARAM.U32 [tid_base], %tmp1
LDPARAM.U32 [tid_depack], %tmp2
LOADCONT %tmp1, MB
DECTID.Y MB, %tmp2, VA
##
LOAD [%input1 + (VA << 1)], MA
CVT.F16.F32 MA, VAB
##
COPY VAB, ICD
##

$LOOP:
LOADCONT %input0, MB
CVT.F16.F32 MB, VAB
##
MUL.F32 VAB, ICD, VAB
##
CVT.F32.F16 VAB, VA
##
STORECONT %output, VA
##
ADD.U32 %input0, %loopStride, %input0
ADD.U32 %output, %loopStride, %output
SUB.U16 %repeatNum, %one, %repeatNum
JMPC %repeatNum, $LOOP
##
