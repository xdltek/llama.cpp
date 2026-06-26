.param .u32 input0
.param .u32 input1
.param .u32 input2
.param .u32 output
.param .u32 in0BlockStride
.param .u32 in1BlockStride
.param .u16 type
.param .u16 cntX

.var .u32 %input0
.var .u32 %input1
.var .u32 %input2
.var .u32 %output
.var .u32 %in0BlockStride
.var .u32 %in1BlockStride
.var .u32 %tmp
.var .u16 %type
.var .u16 %loop
.var .u16 %one

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %type
LDPARAM.U16 [tbDim.y], %loop
LDPARAM.U16 [tid_xyz], %one
CFGXYZ %type, %loop, %one

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
LDPARAM.U32 [input2], %input2
LDPARAM.U32 [output], %output
LDPARAM.U32 [in0BlockStride], %in0BlockStride
LDPARAM.U32 [in1BlockStride], %in1BlockStride
LDPARAM.U16 [cntX], %loop
LDPARAM.U16 [type], %type
MOVE.U16 1, %one

JMPC %type, $TYPE32
$TYPE16:
LOADCONT %input2, MB
CVT.F16.F32 MB, VAB
##
COPY VAB, ICD
##
JMP $LOOP0

$TYPE32:
LDPARAM.U32 [tid_base], %tmp
LDPARAM.U32 [tid_depack], DS13
LOADCONT %tmp, MB
DECTID.X MB, DS13, VA
##
MSHIFTADD VA, %input2, 2, DVA
##
LOAD [DVA], IC
##
LOAD [DVA + 2], ID
##

$LOOP0:
LOADCONT %input0, MB
CVT.F16.F32 MB, VAB
MMOV %input1, DVA
##
COPY VAB, IAB
LOAD [DVA], MA
CVT.F16.F32 MA, VAB
##
MUL.F32 VAB, IAB, VAB
##
MUL.F32 VAB, ICD, VAB
##
CVT.F32.F16 VAB, VA
##
STORECONT %output, VA
##
ADD.U32 %input0, %in0BlockStride, %input0
ADD.U32 %input1, %in1BlockStride, %input1
ADD.U32 %output, %in0BlockStride, %output
SUB.U16 %loop, %one, %loop
JMPC %loop, $LOOP0
##