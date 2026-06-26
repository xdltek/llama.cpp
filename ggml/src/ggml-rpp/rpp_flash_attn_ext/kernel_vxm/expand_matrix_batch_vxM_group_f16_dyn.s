//ENTRY:
.param .u32 input0
.param .u32 input1
.param .u32 out
.param .u32 in0Rollback
.param .u32 in1Rollback
.param .u32 outBatchStride
.param .u32 in1StrideZ
.param .u16 kv_step
.param .u16 batch
.param .u32 deQuant
.param .u32 rowSize
.param .U32 shrNum
.param .U16 kv_page_repeat

.var .u32 %input0
.var .u32 %input1
.var .u32 %out
.var .u32 %deQuant
.var .u32 %offset0
.var .u32 %offset1
.var .u32 %offset2
.var .u32 %offset3
.var .u32 %in1StrideZ
.var .u16 %channel
.var .u16 %sreg1
.var .u16 %sreg2
.var .u16 %loop
.var .u16 %N

//In0 is G * N * 256
//In1 is N * 8 * 256 * 128
//block x = 128, y = 4, z = 8

LDPARAM.U16 [tid_xyz], %channel
LDPARAM.U16 [tbDim.x], %sreg2
LDPARAM.U16 [tbDim.y], %sreg1
CFGXYZ %sreg2, %sreg1, %channel


MOVE.U32 1, %offset0
MOVE.U16 2, S26
MUL.U16.WIDE %sreg2, S26, %offset0

LDPARAM.U16 [kv_page_repeat], %N

MOVE.U32 2, %offset1
MOVE.U32 0, %offset2
MOVE.U32 0, VAB
MMOV %input0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
LDPARAM.U32 [tid_base], %input0
LDPARAM.U32 [tid_depack], %input1
LOADCONT %input0, MB
COPYEXT MB, IH
DECTID.Z MB, %input1, VA
##
MUL.U16 VA, %sreg1, VA
##
COPYEXT VA, IF
DECTID.Y IH, %input1, VA
##
ADD.U16 VA, IF, VA
##
LDPARAM.U16 [kv_step], %channel
LDPARAM.U32 [rowSize], %offset3
MUL.U16.WIDE VA, %offset3, VAB
##
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [out], %out
LDPARAM.U32 [in1StrideZ], %in1StrideZ

SUB.S32 %input1, %offset0, %input1

MOVE.U16 1, %sreg1

MADD.SCA32 VAB, %input0, DVA
MOVE.U32 0, VAB
##
MOVE.U16 %channel, %loop

$BLOCK_ENTRY:
//offset0 is tb.X
ADD.U32 %input1, %offset0, %input1
COPY VAB, ICD
LOAD [DVA], IA
LOADALN.Z32 %input1, %offset2, %in1StrideZ, MB
MADD DVA, 2, DVA
MOVE.U16 MB, VA
##
HFMA IA, VA, ICD, VAB
##
SUB.U16 %loop, %sreg1, %loop
JMPC %loop, $BLOCK_ENTRY
//LDPARAM.U32 [in1Rollback], %offset3
//ADD.U32 %input1, %offset3, %input1
//LDPARAM.U32 [in0Rollback], %offset3
//MADD DVA, %offset3, DVA
MOVE.U32 VAB, VAB
##
MOVE.U16 %channel, %loop
SUB.U16 %N, %sreg1, %N
JMPC %N, $BLOCK_ENTRY

CVT.F32.F16 VAB, VA
##
STORECONT %out, VA
##
