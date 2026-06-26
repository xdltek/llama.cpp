.param .u32 input
.param .u32 output
.param .u16 inStrideY
.param .u16 outStrideY
.param .u32 inStrideZ
.param .u32 outStrideZ
.param .u32 inBlockSize
.param .u32 outBlockSize

.var .u32 %input
.var .u32 %output
.var .u16 %inStrideY
.var .u16 %outStrideY
.var .u32 %inStrideZ
.var .u32 %outStrideZ
.var .u32 %inBlockSize
.var .u32 %outBlockSize
.var .u32 %tmp
.var .u16 %tmp0

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %tmp0
LDPARAM.U16 [tbDim.y], %inStrideY
LDPARAM.U16 [tid_xyz], %outStrideY
CFGXYZ %tmp0, %inStrideY, %outStrideY

$BLOCK_ENTRY:
LDPARAM.U32 [input], %input
LDPARAM.U32 [output], %output
LDPARAM.U16 [inStrideY], %inStrideY
LDPARAM.U16 [outStrideY], %outStrideY
LDPARAM.U32 [inStrideZ], %inStrideZ
LDPARAM.U32 [outStrideZ], %outStrideZ
LDPARAM.U32 [inBlockSize], %inBlockSize
LDPARAM.U32 [outBlockSize], %outBlockSize

CVT.U16.U32 %blockIdx, %tmp
MUL.U32 %tmp, %inBlockSize, %inBlockSize
ADD.U32 %inBlockSize, %input, %input
MUL.U32 %tmp, %outBlockSize, %outBlockSize
ADD.U32 %outBlockSize, %output, %output

MOVE.U16 0, %tmp0
LOADALN.Z32 %input, %inStrideY, %inStrideZ, MB
ADD.U16 MB, %tmp0, VA
##
STOREALN.Z32 %output, %outStrideY, %outStrideZ, VA
##