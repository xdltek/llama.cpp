//ENTRY:
.param .u32 input
.param .u32 out
.param .u32 inStrideX
.param .u32 inStrideY
.param .u32 outStrideY
.param .u32 outStrideZ
.param .u32 inBXStride
.param .u32 inBYStride
.param .u32 inBZStride
.param .u32 outBXStride
.param .u32 outBYStride
.param .u32 outBZStride


.var .u32 %input
.var .u32 %out
.var .u32 %bidX32
.var .u32 %bidY32
.var .u32 %bidZ32
.var .u32 %offset0
.var .u32 %BXStride
.var .u32 %BYStride
.var .u32 %BZStride
.var .u32 %inStrideX
.var .u32 %inStrideY
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %sreg2


LDPARAM.U16 [tid_xyz], %sreg0
LDPARAM.U16 [tbDim.x], %sreg1
LDPARAM.U16 [tbDim.y], %sreg2
CFGXYZ %sreg1, %sreg2, %sreg0
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
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//offset0 =  (blockIdx.x * blockDim.x + threadIdx.x) * in_stridex
//offset0 += (blockIdx.y * blockDim.y + threadIdx.y) * in_stridey
//offset0 +=  blockIdx.z * in_stridez
//offset0 = scalar0 + vector0
//scalar0 = blockIdx.x * blockDim.x * in_stridex + blockIdx.y * blockDim.y * in_stridey + blockIdx.z * in_stridez
//        = blockIdx.x * inBXStride + blockIdx.y * inBYStride + blockIdx.z * inBZStride
//vector0 = threadIdx.x * in_stridex + threadIdx.y * in_stridey
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
CVT.U16.U32 S28, %bidX32
CVT.U16.U32 S29, %bidY32
CVT.U16.U32 S30, %bidZ32

LDPARAM.U32 [inStrideX], %inStrideX
LDPARAM.U32 [inStrideY], %inStrideY
LDPARAM.U32 [input], %input
LDPARAM.U32 [inBXStride], %BXStride
LDPARAM.U32 [inBYStride], %BYStride
LDPARAM.U32 [inBZStride], %BZStride
MUL.U32.LOW %bidX32, %BXStride, %BXStride
MUL.U32.LOW %bidY32, %BYStride, %BYStride
MUL.U32.LOW %bidZ32, %BZStride, %BZStride
ADD.U32 %input, %BXStride, %input
ADD.U32 %input, %BYStride, %input
ADD.U32 %input, %BZStride, %input

LDPARAM.U32 [tid_base], %BXStride
LDPARAM.U32 [tid_depack], %BYStride
LOADCONT %BXStride, MB
COPYEXT MB, IH
DECTID.X MB, %BYStride, VA
##
CVT.U16.U32 VA, VAB
##
MUL.U32.LOW VAB, %inStrideX, VAB
##
MADD.SCA32 VAB, %input, DVA
DECTID.Y IH, %BYStride, VA
##
CVT.U16.U32 VA, VAB
##
MUL.U32.LOW VAB, %inStrideY, VAB
##
MADD DVA, VAB, DVA
##
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//offset1 =  (blockIdx.x * blockDim.x + threadIdx.x) * out_stridex
//offset1 += (blockIdx.y * blockDim.y + threadIdx.y) * out_stridey
//offset1 +=  blockIdx.z * out_stridez
//offset1 = scalar1 + vector1
//scalar1 = blockIdx.x * blockDim.x * out_stridex + blockIdx.y * blockDim.y * out_stridey + blockIdx.z * out_stridez
//        = blockIdx.x * outBXStride + blockIdx.y * outBYStride + blockIdx.z * outBZStride
//vector1 = threadIdx.y * out_stridey + threadIdx.x
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

LDPARAM.U32 [out], %out
LDPARAM.U32 [outBXStride], %BXStride
LDPARAM.U32 [outBYStride], %BYStride
LDPARAM.U32 [outBZStride], %BZStride
MUL.U32.LOW %bidX32, %BXStride, %BXStride
MUL.U32.LOW %bidY32, %BYStride, %BYStride
MUL.U32.LOW %bidZ32, %BZStride, %BZStride
ADD.U32 %out, %BXStride, %out
ADD.U32 %out, %BYStride, %out
ADD.U32 %out, %BZStride, %out
LOAD [DVA], IA
##
STORECONT %out, IA
##