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
LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, MB
COPYEXT MB, IH
DECTID.X MB, %tmp1, VA
##
COPY VA, IC
DECTID.Y IH, %tmp1, VA
##
COPY VA, ID
MUL.U16.WIDE VA, %in0StrideY, VAB
##
COPYEXT VAB, IGH
CVT.U16.U32 IC, VAB
##
ADD.U32 VAB, IGH, VAB
##
SHL.U32 VAB, 1, VAB
##
COPYEXT VAB, IGH
MUL.U16.LOW ID, %in1StrideY, VA
##
ADD.U16 VA, IC, VA
##
MOVE.U16 2, %S1
MUL.U16.WIDE %S2, %S1, %tmp0
COPYEXT VA, IF
##
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

//-----------------------------------------------------
//d0 = x0*cos - x1*sin
//d1 = x0*sin + x1*cos
//------------------------------------------------------
//x0 = IA
//x1 = IB
//-x1 = IE
//cos0 = IC
//sin0 = ID
//y0 = x0 * cos0 - x1 * sin1;
//y1 = x0 * sin0 + x1 * cos1;
//LOAD [%input0 + (IF << 1)], IA
MADD.SCA32 IGH, %input0, DVA
##
LOAD [DVA], IA
MADD.SCA32 DVA, %tmp0, DVA
##
LOAD [DVA], MA
COPY MA, IB
NEG.F16 MA, VA
##
COPYEXT VA, IE
LOAD [%input1 + (IF << 1)], IC
##
LOAD [%input2 + (IF << 1)], ID
MUL.F16 IA, IC, VA
##
FMA IE, ID, VA, VA
MADD.SCA32 IGH, %out, DVA
##
STORE VA, [DVA]
MADD.SCA32 DVA, %tmp0, DVA
MUL.F16 IA, ID, VA
##
FMA IB, IC, VA, VA
##
STORE VA, [DVA]
##