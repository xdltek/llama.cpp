.param .u32 input0
.param .u32 input1
.param .u32 output
.param .u32 in0BlockStride
.param .u32 in1BlockStride

.var .u32 %input0
.var .u32 %input1
.var .u32 %output
.var .u32 %in0BlockStride
.var .u32 %in1BlockStride
.var .u32 %tmp
.var .u16 %tmp0
.var .u16 %tmp1
.var .u16 %tmp2

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %tmp0
LDPARAM.U16 [tbDim.y], %tmp1
LDPARAM.U16 [tid_xyz], %tmp2
CFGXYZ %tmp0, %tmp1, %tmp2

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
LDPARAM.U32 [in0BlockStride], %in0BlockStride
LDPARAM.U32 [in1BlockStride], %in1BlockStride

CVT.U16.U32 %blockIdx, %tmp
MUL.U32 %in0BlockStride, %tmp, %in0BlockStride
MUL.U32 %in1BlockStride, %tmp, %in1BlockStride
ADD.U32 %input0, %in0BlockStride, %input0
ADD.U32 %input1, %in1BlockStride, %input1
ADD.U32 %output, %in0BlockStride, %output

##
LOADCONT %input0, MB
CVT.F16.F32 MB, VAB
MMOV %input1, DVA
##
COPY VAB, IAB
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
SUB.F32 IAB, VAB, VAB
##
MUL.F32 VAB, VAB, VAB
##
CVT.F32.F16 VAB, VA
##
STORECONT %output, VA
##
