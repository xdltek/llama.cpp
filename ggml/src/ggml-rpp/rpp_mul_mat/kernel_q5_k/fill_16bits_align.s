//ENTRY: fill the memory continuously
.param .u32 output       //output
.param .u32 blockSize
.param .u16 value

.var .u32 %output
.var .u32 %offset0
.var .u32 %block_size
.var .u16 %one
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2

MOVE.U16 1, %one
JMPC %blockIdx, $BLOCK_ENTRY_1
LDPARAM.U16 [tid_xyz], %S0        // 一个thread block 的元素个数
LDPARAM.U16 [tbDim.x], %S2
LDPARAM.U16 [tbDim.y], %S1
CFGXYZ %S2, %S1, %S0
JMP $BLOCK_ENTRY
$BLOCK_ENTRY_1:
//only grid in x axis
MOVE.U16 %blockIdx.x, %S0
LDPARAM.U16 [grid_dim_x], %S1
//bypass only one block
ADD.U16 %S0, %one, %S2
SUB.U16 %S1, %S2, %S2
JMPC %S2, $BLOCK_ENTRY
//config the block dimension for the last block
LDPARAM.U16 [xyzcfg_shadow], %S0
LDPARAM.U16 [dimx_shadow], %S2
LDPARAM.U16 [dimy_shadow], %S1
CFGXYZ %S2, %S1, %S0

$BLOCK_ENTRY:
LDPARAM.U32 [output], %output
LDPARAM.U32 [blockSize], %block_size

MUL.U16.WIDE %blockIdx, %one, %offset0
MUL.U32.LOW %offset0, %block_size, %offset0
ADD.S32 %output, %offset0, %output

LDPARAM.U16 [value], %S1
MOVE.U16 %S1, VA
##
STORECONT %output, VA
##


