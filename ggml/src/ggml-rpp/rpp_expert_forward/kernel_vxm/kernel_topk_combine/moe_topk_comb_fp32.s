//ENTRY:
.param .u32 input0
.param .u32 weights
.param .u32 input_stride
.param .u32 out
.param .u16 topk


.var .u32 %input0
.var .u32 %weight_lo
.var .u32 %weight_hi
.var .u32 %input_stride
.var .u32 %out
.var .u32 %temp
.var .u16 %loop
.var .u16 %topk
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %one

LDPARAM.U16 [tid_xyz], %S0 
LDPARAM.U16 [tbDim.x], %S1
LDPARAM.U16 [tbDim.y], %S2
CFGXYZ %S1, %S2, %S0
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
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [weights], %weight_lo
LDPARAM.U32 [input_stride], %input_stride
LDPARAM.U32 [out], %out
LDPARAM.U16 [topk], %topk
MOVE.U16 1, %one
MOVE.U32 2, %temp
ADD.U32  %weight_lo, %temp, %weight_hi
MOVE.U32 4, %temp
MOVE.U32 0, VAB
##
//----------------------------------------------------------------------------------------------------   
// Input    [topk]   |   [N]
//                   |   [x]
// weights  [topk]
// output            |   [N]
//----------------------------------------------------------------------------------------------------
$BLOCK_ENTRY:
COPYEXT VAB, IEF
LOADCONT %input0, MB 
CVT.F16.F32 MB, VAB
##
COPY VAB, ICD
LOAD [%weight_lo + (IB << 1)], IG
##
LOAD [%weight_hi + (IB << 1)], IH
##
MUL.F32 ICD, IGH, VAB
##
ADD.F32 VAB, IEF, VAB
##
ADD.U32 %weight_lo, %temp, %weight_lo
ADD.U32 %weight_hi, %temp, %weight_hi
ADD.U32 %input0, %input_stride, %input0
SUB.U16 %topk, %one, %topk
JMPC %topk, $BLOCK_ENTRY
COPYEXT VAB, IEF
##
STORECONT %out, IF
##
