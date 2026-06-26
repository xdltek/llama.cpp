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
.var .u16 %Un

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
MMOV 0, DVA
MOVE.U16 0, VA
##
COPY VA, IA
COPY VA, IB
COPY VA, IC
COPY VA, ID
COPYEXT VA, IE
COPYEXT VA, IF
COPYEXT VA, IG
COPYEXT VA, IH
FORWARD DVA
##

LDPARAM.U32 [input], %input
LDPARAM.U16 [Un],    %Un
LDPARAM.U32 [out],   %out
LDPARAM.U16 [typeIn], %S0
LDPARAM.U16 [typeOut], %S1

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
LOAD [%input + (VA << 2)], IA    // load the input data
MOVE.U16 VA, VA
##
MOVE.U32 2, DS13
ADD.U32 %input, DS13, %input
LOAD [%input + (VA << 2)], IB    // load the input data
##
MOVE.U16 0, %S2
CMP.S16.EQ %S0, %S2 ,%S3
JMPNC %S3, $label_S32
MOVE.U16 4, %S2
CMP.S16.EQ %S1, %S2 ,%S3
JMPNC %S3, $F32_TO_U16
CVT.F32.S16 IAB, IA
##
JMP $STORE
##
$F32_TO_U16:
MOVE.U16 12, %S2
CMP.S16.EQ %S1, %S2 ,%S3
JMPNC %S3, $F32_TO_F16
CVT.F32.U16 IAB, IA
##
JMP $STORE
##
$F32_TO_F16:
AND.U16 IB, 1, VA
##
ADD.U16 VA, 0x7fff, VA
##
CVT.U16.U32 VA, VAB
##
ADD.U32 VAB, IAB, VAB
##
CVT.F32.F16 VAB, VA
##
COPY VA, IA
##
JMP $STORE
##

$label_S32:
MOVE.U16 5, %S2
CMP.S16.EQ %S0, %S2 ,%S3
JMPNC %S3, $label_U32
MOVE.U16 4, %S2
CMP.S16.EQ %S1, %S2 ,%S3
JMPNC %S3, $S32_TO_U16
CVT.S32.S16 IAB, IA
##
JMP $STORE
##
$S32_TO_U16:
MOVE.U16 12, %S2
CMP.S16.EQ %S1, %S2 ,%S3
JMPNC %S3, $S32_TO_F16
CVT.S32.U16 IAB, IA
##
JMP $STORE
##
$S32_TO_F16:
CVT.S32.F16 IAB, IA
##
JMP $STORE
##
$label_U32:
MOVE.U16 13, %S2
CMP.S16.EQ %S0, %S2 ,%S3
JMPNC %S3, $end
MOVE.U16 4, %S2
CMP.S16.EQ %S1, %S2 ,%S3
JMPNC %S3, $U32_TO_U16
CVT.U32.S16 IAB, IA
##
JMP $STORE
##
$U32_TO_U16:
MOVE.U16 12, %S2
CMP.S16.EQ %S1, %S2 ,%S3
JMPNC %S3, $U32_TO_F16
CVT.U32.U16 IAB, IA
##
JMP $STORE
##
$U32_TO_F16:
CVT.U32.F16 IAB, IA
##
$STORE:
STORECONT %out, IA
##
$end:
LDPARAM.U16 [typeIn], %S0
##