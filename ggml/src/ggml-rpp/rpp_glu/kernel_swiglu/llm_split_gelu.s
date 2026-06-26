.param .u32 input0
.param .u32 input1Stride
.param .u32 lut
.param .u32 out
.param .u32 in0StrideY
.param .u32 outStrideY
.param .u32 in0BlockXStride
.param .u32 outBlockXStride
.param .u32 in0BlockYStride
.param .u32 outBlockYStride


.var .u32 %input0
.var .u32 %lut
.var .u32 %out
.var .u32 %tmp0
.var .u32 %tmp1
.var .u32 %tmp2
.var .u32 %tmp3
.var .u16 %in0StrideY
.var .u16 %outStrideY
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3

LDPARAM.U16 [tid_xyz], %S0
LDPARAM.U16 [tbDim.x], %S2
LDPARAM.U16 [tbDim.y], %S1
CFGXYZ %S2, %S1, %S0

LDPARAM.U32 [input1Stride], %tmp3

LDPARAM.U32 [input0], %input0
LDPARAM.U32 [lut], %lut
LDPARAM.U32 [out], %out

LDPARAM.U16 [in0StrideY], %in0StrideY
LDPARAM.U16 [outStrideY], %outStrideY

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

CVT.U16.U32 S28, %tmp1
LDPARAM.U32 [in0BlockXStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %input0, %tmp2, %input0

LDPARAM.U32 [outBlockXStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %out, %tmp2, %out

CVT.U16.U32 S29, %tmp1
LDPARAM.U32 [in0BlockYStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %input0, %tmp2, %input0

LDPARAM.U32 [outBlockYStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %out, %tmp2, %out


LOADALN %input0, %in0StrideY,  %in0StrideY, IA
##
ADD.U32 %input0, %tmp3, %input0
LOADALN %input0, %in0StrideY,  %in0StrideY, IB
##
LOAD [%lut + (IA << 1)], MA
MUL.F16 MA, IB, VA
##
STOREALN %out, %outStrideY,  %outStrideY, VA
##