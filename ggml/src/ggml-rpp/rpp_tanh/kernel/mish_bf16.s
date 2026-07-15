//ENTRY:               // y = x * tanh(softplus(x) = x * tanh(ln(1+e(x))
.param .u32 input      //input
.param .u32 out        //out
.param .u32 lut_addr   //lookup table address
.param .u32 blockSize  //block size


.var .u32 %input         //声明标量寄存器变量，  ‘%’开始的变量位于标量寄存器（总共32个）
.var .u32 %offset0
.var .u32 %out
.var .u32 %tmp0
.var .u32 %tmp1
.var .u16 %one
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2
.var .u16 %tbDim.x
.var .u16 %unCnt

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tid_xyz], %sreg0        // 一个thread block 的元素个数
LDPARAM.U16 [tbDim.x], %tbDim.x
LDPARAM.U16 [tbDim.y], %sreg1
CFGXYZ %tbDim.x, %sreg1, %sreg0

LDPARAM.U32 [blockSize],  %tmp0
LDPARAM.U32 [lut_addr], %tmp1
LDPARAM.U16 [grid_dim_x], %sreg2
MOVE.U16 1, %one
$BLOCK_ENTRY:
//only grid in x axis
ADD.U16 %blockIdx, %one, %unCnt
SUB.U16 %sreg2, %unCnt, %unCnt
JMPC %unCnt, $BLOCK_INIT
//config the block dimension for the last block
LDPARAM.U16 [xyzcfg_shadow], %sreg0
LDPARAM.U16 [dimx_shadow], %tbDim.x
LDPARAM.U16 [dimy_shadow], %sreg1
CFGXYZ %tbDim.x, %sreg1, %sreg0
$BLOCK_INIT:


LDPARAM.U32 [input], %input
MUL.U16.WIDE %blockIdx, %one, %offset0
MUL.U32.LOW %offset0, %tmp0, %offset0
ADD.S32 %input, %offset0, %input
LDPARAM.U32 [out], %out
ADD.S32 %out, %offset0, %out
LOADCONT %input, IC	    	// MA MB 载入数据的临时寄存器
##
LOAD [%tmp1 + (IC << 1)], IA
##
STORECONT %out, IA
##