//ENTRY:
.param .u32 input0
.param .u32 input1
.param .u32 out
.param .u32 in0BatchStride
.param .u32 in1BatchStride
.param .u32 outBatchStride
.param .u16 channel
.param .u16 batch
.param .u32 deQuant
.param .u32 rowSize
.param .u16 expandStride
.param .u16 rollback
.param .u32 inStrideZ
.param .u32 outStrideZ
.param .u32 outLow


.var .u32 %input0
.var .u32 %input1
.var .u32 %input1Tmp
.var .u32 %out
.var .u32 %outLow

.var .u32 %inBatchStride
.var .u32 %outBatchStride
.var .u32 %offset0
.var .u32 %Tmp0
.var .u16 %batch
.var .u16 %channel
.var .u16 %one
.var .u16 %expandStride
.var .u16 %rollback
.var .u16 %sreg0
.var .u16 %sreg1


LDPARAM.U16 [tbDim.z], %sreg0
LDPARAM.U16 [tbDim.x], %sreg1
MUL.U16.LOW  %sreg0, %sreg1, %expandStride
LDPARAM.U16 [tbDim.y], %rollback
CFGXYZ %sreg1, %rollback, %expandStride

MOVE.U32 0, VAB
MMOV 0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##

LDPARAM.U32 [rowSize], %offset0
//rowIdx = counter & 0x3F
//rowSize & rowIdx
LDPARAM.U16 [expandStride], %expandStride
LDPARAM.U16 [rollback], %rollback

LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [out], %out
LDPARAM.U32 [outLow], %outLow

LDPARAM.U32 [outBatchStride], %outBatchStride

LDPARAM.U16 [batch], %batch
LDPARAM.U16 [channel], %channel

MOVE.U16 1, %one

SUB.S32 %out, %outBatchStride, %out
SUB.S32 %outLow, %outBatchStride, %outLow
SUB.S32 %input1, %offset0, %input1Tmp


LDPARAM.U16 [outStrideZ], %sreg1

MMOV %input0, DVA
MOVE.U32 0, VAB
##
///////////////////first 2 expand /////////////////////
LDPARAM.U32 [inStrideZ], %Tmp0
$BLOCK_ENTRY_0:
COPY VAB, ICD
ADD.U32 %input1Tmp, %offset0, %input1Tmp
LOADALN.Z32 %input1Tmp, %sreg0, %Tmp0, IF
LOAD [DVA], MA
MADD DVA, %expandStride, DVA
CVT.F16.F32 MA, VAB
##
MUL.F32 VAB, IEF, VAB
##
ADD.F32 VAB, IAB, VAB
##

COPY VAB, IAB
LOAD [DVA], MA
MSUB DVA, %rollback, DVA
CVT.F16.F32 MA, VAB
##
MUL.F32 VAB, IEF, VAB
##
ADD.F32 VAB, ICD, VAB
##

SUB.U16 %channel, %one, %channel
JMPC %channel, $BLOCK_ENTRY_0
LDPARAM.U16 [channel], %channel
COPY VAB, ICD
ADD.U32 %out, %outBatchStride, %out
STOREALN %out, %sreg1, %sreg1, IB
##
MOVE.U32 0, VAB
ADD.U32 %outLow, %outBatchStride, %outLow
STOREALN %outLow, %sreg1, %sreg1, IA
##
COPY VAB, IAB
SHL.U16 %expandStride, %one, %sreg0
CVT.U16.U32 %sreg0, DS13
ADD.U32 %input0, DS13, DS13
MMOV DS13, DVA
SUB.S32 %input1, %offset0, %input1Tmp
ADD.U32 %out, %outBatchStride, %out
STOREALN %out, %sreg1, %sreg1, ID
##
MOVE.U32 0, VAB
ADD.U32 %outLow, %outBatchStride, %outLow
STOREALN %outLow, %sreg1, %sreg1, IC
##
///////////////////next 2 expand /////////////////////
$BLOCK_ENTRY_1:
COPY VAB, IAB
ADD.U32 %input1Tmp, %offset0, %input1Tmp
LOADALN.Z32 %input1Tmp, %sreg0, %Tmp0, IF
LOAD [DVA], MA
MADD DVA, 2, DVA
CVT.F16.F32 MA, VAB
##
MUL.F32 IEF, VAB, VAB
##
ADD.F32 VAB, IAB, VAB
##
SUB.U16 %channel, %one, %channel
JMPC %channel, $BLOCK_ENTRY_1
LDPARAM.U16 [channel], %channel
COPY VAB, IAB
##
ADD.U32 %out, %outBatchStride, %out
STOREALN %out, %sreg1, %sreg1, IB
##
MOVE.U32 0, VAB
ADD.U32 %outLow, %outBatchStride, %outLow
STOREALN %outLow, %sreg1, %sreg1, IA
##
LDPARAM.U32 [in0BatchStride], %inBatchStride
ADD.U32 %input0, %inBatchStride, %input0
LDPARAM.U32 [in1BatchStride], %inBatchStride
ADD.U32 %input1, %inBatchStride, %input1
SUB.U32 %input1, %offset0, %input1Tmp
MMOV %input0, DVA
COPY VAB, IAB
##
MOVE.U32 0, VAB
##
SUB.U16 %batch, %one, %batch
JMPC %batch, $BLOCK_ENTRY_0
TERM VAB
##
