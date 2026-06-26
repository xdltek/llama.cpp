.param .u32 input     //input
.param .u32 lut_addr
.param .u32 out       //out
.param .u32 bias       //out

.var .u32 %input         //声明标量寄存器变量，  ‘%’开始的变量位于标量寄存器（总共32个）
.var .u32 %offset0
.var .u32 %out
.var .u32 %bias

.var .u16 %one
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %tbDim.x

.var .u32 %tmp0
.var .u32 %tmp1
.var .u32 %epsilon
.var .u32 %Magic

LDPARAM.U16 [tid_xyz], %sreg0        // 一个thread block 的元素个数
LDPARAM.U16 [tbDim.x], %tbDim.x
LDPARAM.U16 [tbDim.y], %sreg1
CFGXYZ %tbDim.x, %sreg1, %sreg0
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
##
LDPARAM.U32 [lut_addr], %tmp1
LDPARAM.U32 [input], %input
LDPARAM.U32 [out], %out
LDPARAM.U32 [bias], %bias
LOADCONT %input, MB
CVT.F16.F32 MB, VAB
##
ADD.F32 VAB, %bias, VAB
##
CVT.F32.F16 VAB, VA
##
LOAD [%tmp1 + (VA << 1)], IA
##
STORECONT %out, IA
##