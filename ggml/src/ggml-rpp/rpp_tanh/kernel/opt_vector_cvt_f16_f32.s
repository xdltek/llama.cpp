//ENTRY:                 // 与C++中 tRPP_PARA 按比特对应，有次序
.param .u32 input        //input
.param .u32 out          //out
.param .u16 typeIn       //input data type
.param .u16 typeOut      //output data type
.param .u32 Un           //thread block 数目
.param .u32 unStrideIn   //input 一个thread block 占的内存大小, in bytes
.param .u32 unStrideOut  //output 一个thread block 占的内存大小, in bytes


.var .u32 %input         //声明标量寄存器变量，  ‘%’开始的变量位于标量寄存器（总共32个）
.var .u32 %out
.var .u32 %tmp0
.var .u32 %tmp1
.var .u16 %one
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3


JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tid_xyz], %S0        // 一个thread block 的元素个数
LDPARAM.U16 [tbDim.x], %S2
LDPARAM.U16 [tbDim.y], %S1
CFGXYZ %S2, %S1, %S0
MOVE.U16 1, %one
JMP $KERNEL_START
$BLOCK_ENTRY:
//only grid in x axis
MOVE.U16 %blockIdx, %S0
LDPARAM.U16 [grid_dim_x], %S1
ADD.U16 %S0, %one, %S2
SUB.U16 %S1, %S2, %S2
JMPC %S2, $KERNEL_START
//config the block dimension for the last block
LDPARAM.U16 [xyzcfg_shadow], %S0
LDPARAM.U16 [dimx_shadow], %S2
LDPARAM.U16 [dimy_shadow], %S1
CFGXYZ %S2, %S1, %S0

$KERNEL_START:

LDPARAM.U32 [input], %input
LDPARAM.U32 [out],   %out
MOVE.U16 1, %one

//input
MUL.U16.WIDE %blockIdx, %one, %tmp0
LDPARAM.U32 [unStrideIn],   %tmp1
MUL.U32.LOW %tmp0, %tmp1, %tmp0
ADD.S32 %input, %tmp0, %input
//output
MUL.U16.WIDE %blockIdx, %one, %tmp0
LDPARAM.U32 [unStrideOut],   %tmp1
MUL.U32.LOW %tmp0, %tmp1, %tmp0
ADD.S32 %out, %tmp0, %out

LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, MB
DECTID.X MB, %tmp1, VA  //IA is threadIdx.x
##
COPY VA, IA
LOADCONT %input, IB    // load the input data
##
$STORE:
STORE IB, [%out + (IA << 2)]
##
