.param .u32 input
.param .u32 luTable
.param .u32 output
.param .u32 nLoopStride
.param .u32 blockLoopStride
.param .u16 rowStrideY
.param .u16 blockRepeatNum
.param .u16 kv_page_repeat

.var .u32 %input
.var .u32 %luTable
.var .u32 %output
.var .u32 %nLoopStride
.var .u32 %blockLoopStride
.var .u32 %tmp
.var .u32 %tmp1
.var .u16 %nRepeatNum
.var .u16 %blockRepeatNum
.var .u16 %rowStrideY
.var .u16 %nLoopIdx
.var .u16 %blockLoopIdx
.var .u16 %one

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %blockRepeatNum
LDPARAM.U16 [tbDim.y], %nLoopIdx
LDPARAM.U16 [tid_xyz], %blockLoopIdx
CFGXYZ %blockRepeatNum, %nLoopIdx, %blockLoopIdx

$BLOCK_ENTRY:
LDPARAM.U32 [input], %input
LDPARAM.U32 [luTable], %luTable
LDPARAM.U32 [output], %output
LDPARAM.U32 [nLoopStride], %nLoopStride
LDPARAM.U32 [blockLoopStride], %blockLoopStride
LDPARAM.U16 [rowStrideY], %rowStrideY
LDPARAM.U16 [kv_page_repeat], %nRepeatNum
MOVE.U16 1, %one

MOVE.U16 0, %nLoopIdx
$N_LOOP:

LDPARAM.U16 [blockRepeatNum], %blockRepeatNum
MOVE.U16 0, %blockLoopIdx
$BLOCK_LOOP:

CVT.U16.U32 %nLoopIdx, %tmp
MUL.U32 %tmp, %nLoopStride, %tmp
CVT.U16.U32 %blockLoopIdx, %tmp1
MUL.U32 %tmp1, %blockLoopStride, %tmp1
ADD.U32 %tmp, %tmp1, %tmp
ADD.U32 %input, %tmp, %tmp1
ADD.U32 %output, %tmp, %tmp

LOADALN %tmp1, %rowStrideY, %rowStrideY, IA
##
LOAD [%luTable + (IA << 1)], IB
##
STOREALN %tmp, %rowStrideY, %rowStrideY, IB
##

ADD.U16 %blockLoopIdx, %one, %blockLoopIdx
SUB.U16 %blockRepeatNum, %one, %blockRepeatNum
JMPC %blockRepeatNum, $BLOCK_LOOP

ADD.U16 %nLoopIdx, %one, %nLoopIdx
SUB.U16 %nRepeatNum, %one, %nRepeatNum
JMPC %nRepeatNum, $N_LOOP
##
