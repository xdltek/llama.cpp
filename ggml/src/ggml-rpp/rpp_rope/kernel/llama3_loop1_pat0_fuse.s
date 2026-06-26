.param .u32 input0
.param .u32 input1
.param .u32 input2
.param .u32 out
.param .u32 in0StrideY
.param .u32 in1StrideY
.param .u32 outStrideY
.param .u32 in0BlockYStride
.param .u32 in0BlockZStride
.param .u32 in1BlockYStride
.param .u32 outBlockYStride
.param .u32 outBlockZStride


.var .u32 %input0
.var .u32 %input1
.var .u32 %input2
.var .u32 %out
.var .u32 %tmp0
.var .u32 %tmp1
.var .u32 %tmp2

.var .u16 %in0StrideY
.var .u16 %in1StrideY
.var .u16 %outStrideY

.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3

LDPARAM.U16 [tid_xyz], %S0
LDPARAM.U16 [tbDim.x], %S2
LDPARAM.U16 [tbDim.y], %S1
CFGXYZ %S2, %S1, %S0

MOVE.U16 2, %S1
MUL.U16.WIDE %S1, %S2, %tmp0

LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [input2], %input2
LDPARAM.U32 [out], %out

LDPARAM.U16 [in0StrideY], %in0StrideY
LDPARAM.U16 [in1StrideY], %in1StrideY
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

//in0 [128][8][128]
//in1 [128][128]
//in2 [128][128]
//out [8][128][128]
//x = 64, y = 64, z = 1, bx = 2, by = 2, bz = 8
//in0 [by * 64][bz][bx * 64]
//in1 [by * 64][bx * 64]
//in2 [by * 64][bx * 64]
//out [bz][by * 64][bx * 64]
//in0StrideY = bz * bx * 64 = 8 * 2 * 64
//in1StrideY = bx * 64 = 128
//in2StrideY = bx * 64 = 128
//outStrideY = bx * 64 = 128

//in0BlockYStride = 64 * bz * bx * 64 = 64 * 8 * 2 * 128
//in0BlockZStride = bx * 64

CVT.U16.U32 S29, %tmp1
LDPARAM.U32 [in0BlockYStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %input0, %tmp2, %input0

LDPARAM.U32 [in1BlockYStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %input1, %tmp2, %input1
ADD.U32 %input2, %tmp2, %input2

LDPARAM.U32 [outBlockYStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %out, %tmp2, %out

CVT.U16.U32 S30, %tmp1
LDPARAM.U32 [in0BlockZStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %input0, %tmp2, %input0

LDPARAM.U32 [outBlockZStride], %tmp2
MUL.U32 %tmp1, %tmp2, %tmp2
ADD.U32 %out, %tmp2, %out


LOADALN %input0, %in0StrideY,  %in0StrideY, IA
##
ADD.U32 %input0, %tmp0, %input0
LOADALN %input0, %in0StrideY,  %in0StrideY, MB
COPY MB, IB
NEG.F16 MB, VA
##
COPYEXT VA, IE
LOADALN %input1, %in1StrideY,  %in1StrideY, IC
##
LOADALN %input2, %in1StrideY,  %in1StrideY, ID
##
//x0 = IA
//x1 = IB
//-x1 = IE
//cos0 = IC
//sin0 = ID
//y0 = x0 * cos0 - x1 * sin1;
//y1 = x0 * sin0 + x1 * cos1;
MUL.F16 IA, IC, VA
##
FMA IE, ID, VA, VA
##
STOREALN %out, %outStrideY,  %outStrideY, VA
MUL.F16 IA, ID, VA
##
FMA IB, IC, VA, VA
##
ADD.U32 %out, %tmp0, %out
STOREALN %out, %outStrideY,  %outStrideY, VA
##