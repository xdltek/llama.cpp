//ENTRY:
.param .u32 input
.param .u32 output
.param .u32 blockXSizeIn
.param .u32 blockYSize
.param .u32 blockXSizeOut
.param .u16 inStrideY
.param .u16 inStrideZ
.param .u16 outStrideY
.param .u16 outStrideZ

.var .u32 %input
.var .u32 %out

.var .u32 %tmp0
.var .u32 %tmp1
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3
.var .u16 %one


LDPARAM.U16 [tid_xyz], %S0 
LDPARAM.U16 [tbDim.x], %S1
LDPARAM.U16 [tbDim.y], %S2
CFGXYZ %S1, %S2, %S0
MOVE.U32 0, VAB
MMOV 0, DVA
MOVE.U16 1, %one
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
//input
LDPARAM.U32 [input], %input
LDPARAM.U32 [output], %out
MUL.U16.WIDE %blockIdx.x, %one, %tmp0
LDPARAM.U32 [blockXSizeIn],   %tmp1
MUL.U32.LOW %tmp0, %tmp1, %tmp0
ADD.S32 %input, %tmp0, %input
MUL.U16.WIDE %blockIdx.y, %one, %tmp0
LDPARAM.U32 [blockYSize],   %tmp1
MUL.U32.LOW %tmp0, %tmp1, %tmp0
ADD.S32 %input, %tmp0, %input
ADD.S32 %out, %tmp0, %out
//output
MUL.U16.WIDE %blockIdx.x, %one, %tmp0
LDPARAM.U32 [blockXSizeOut],   %tmp1
MUL.U32.LOW %tmp0, %tmp1, %tmp0
ADD.S32 %out, %tmp0, %out
$BLOCK_ENTRY:
LDPARAM.U16 [inStrideY], %S0
LDPARAM.U16 [inStrideZ], %S1
LOADALN %input, %S0, %S1, IA
##
LDPARAM.U16 [outStrideY], %S2
LDPARAM.U16 [outStrideZ], %S3
STOREALN %out, %S2, %S3, IA
##