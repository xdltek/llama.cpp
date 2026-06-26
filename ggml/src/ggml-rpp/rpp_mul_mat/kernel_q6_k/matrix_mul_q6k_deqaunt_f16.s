.param   .U32 in_lsb
.param   .U32 in_msb
.param   .U32 in_scale
.param   .U32 out_addr
.param   .U32 lut_addr
.param   .U32 inStrideY0
.param   .U32 inStrideZ0
.param   .U32 inStrideY1
.param   .U32 inStrideZ1
.param   .U32 dequantStrideY
.param   .U32 dequantStrideZ
.param   .U32 outStrideY
.param   .U32 outStrideZ
.param   .U32 inUnrollStride
.param   .U32 outUnrollStride
.param   .U32 inBlockXSize0
.param   .U32 inBlockXSize1
.param   .U32 inBlockYSize0
.param   .U32 inBlockYSize1
.param   .U32 inDequantXSize
.param   .U32 inDequantYSize
.param   .U32 outBlockXSize
.param   .U32 outBlockYSize
.param   .U32 zero_addr
.param   .U16 blockX
.param   .U16 blockY


.var .u32  %in_lsb
.var .u32  %in_msb
.var .u32  %in_scale
.var .u32  %out_addr
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %strideZ
.var .u32  %blockSize0
.var .u32  %blockSize1
.var .u16  %cntX
.var .u16  %cntY
.var .u16  %one
.var .u16  %strideY
.var .u16  %loop0


LDPARAM.U16 [tbDim.x], %cntY
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %cntY, %one, %cntX
MOVE.U16 1, %one
MOVE.U16 2, %loop0
LDPARAM.U32 [in_lsb], %in_lsb
LDPARAM.U32 [in_msb], %in_msb
LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [out_addr], %out_addr
LDPARAM.U32 [inUnrollStride], %inUnrollStride
LDPARAM.U32 [outUnrollStride], %outUnrollStride
LDPARAM.U16 [blockX], %cntX
LDPARAM.U16 [blockY], %cntY
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
$BLOCK_ENTRY:
//-----------------------------------------------------------------------------------------------------
// in_scale   [K/256]      | [256/elements_per_thread] | [N]
// in_scale   [z]          | [y]                       | [x]
//            [grid.y]*[z] | [y]                       | [grid.x]*[x]
//-----------------------------------------------------------------------------------------------------
LDPARAM.U16 [dequantStrideY], %strideY
LDPARAM.U32 [dequantStrideZ], %strideZ
LOADALN.Z32 %in_scale, %strideY, %strideZ, IC
##
$BLOCK_0:
//----------------------------------------------------------------------------------------------------
// in_lsb     [K/256][256/16][4][4][N]
// in_lsb     [K/256]      | [256/elements_per_thread] | [elements_per_thread/wqlsb_per_word] | [wqlsb_per_word][N]
// in_lsb     [z]          | [y]                       | [unroll]                             | [x]
//            [grid.y]*[z] | [y]                                                              | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
LDPARAM.U16 [inStrideY0], %strideY
LDPARAM.U32 [inStrideZ0], %strideZ
LOADALN.Z32 %in_lsb, %strideY, %strideZ, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
LDPARAM.U16 [inStrideY1], %strideY
LDPARAM.U32 [inStrideZ1], %strideZ
LOADALN.Z32 %in_msb, %strideY, %strideZ, MB
COPYEXT MB, IH
SHL.U16 MB, 4, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
// element0
// out_addr   [N/32][K/256][256/16][16][32]
// out_addr   [N/32]   | [K/256]      | [256/elements_per_thread] | [elements_per_thread] | [32]
// out_addr            | [z]          | [y]                       | [unroll]              | [x]
//            [grid.x] | [grid.y]*[z] | [y]                                               | [x]
//-----------------------------------------------------------------------------------------------------
LDPARAM.U16 [outStrideY], %strideY
LDPARAM.U32 [outStrideZ], %strideZ
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
SHL.U16 IH, 2, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements1
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
AND.U16 IH, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements2
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
SHR.U16 IH, 2, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
#
//-----------------------------------------------------------------------------------------------------
//elements3
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
##

LDPARAM.U16 [inStrideY0], %strideY
LDPARAM.U32 [inStrideZ0], %strideZ
ADD.U32 %in_lsb, %inUnrollStride, %in_lsb
LOADALN.Z32 %in_lsb, %strideY, %strideZ, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
SHR.U16 IH, 4, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements4
//-----------------------------------------------------------------------------------------------------
LDPARAM.U16 [outStrideY], %strideY
LDPARAM.U32 [outStrideZ], %strideZ
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
SHR.U16 IH, 6, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements5
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
SHR.U16 IH, 8, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements6
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
SHR.U16 IH, 10, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements7
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
##
SUB.U16 %loop0, %one, %loop0
ADD.U32 %in_lsb, %inUnrollStride, %in_lsb
ADD.U32 %in_msb, %inUnrollStride, %in_msb
ADD.U32 %out_addr, %outUnrollStride, %out_addr
JMPC %loop0, $BLOCK_0

MOVE.U16 2, %loop0
SUB.U16 %cntX, %one, %cntX

LDPARAM.U32 [inBlockXSize0], %blockSize0
LDPARAM.U32 [inBlockXSize1], %blockSize1
ADD.U32 %in_lsb, %blockSize0, %in_lsb
ADD.U32 %in_msb, %blockSize1, %in_msb

LDPARAM.U32 [inDequantXSize], %blockSize0
LDPARAM.U32 [outBlockXSize], %blockSize1
ADD.U32 %in_scale, %blockSize0, %in_scale
ADD.U32 %out_addr, %blockSize1, %out_addr
JMPC %cntX, $BLOCK_ENTRY

LDPARAM.U16 [blockX], %cntX
SUB.U16 %cntY, %one, %cntY
LDPARAM.U32 [inBlockYSize0], %blockSize0
LDPARAM.U32 [inBlockYSize1], %blockSize1
ADD.U32 %in_lsb, %blockSize0, %in_lsb
ADD.U32 %in_msb, %blockSize1, %in_msb

LDPARAM.U32 [inDequantYSize], %blockSize0
LDPARAM.U32 [outBlockYSize], %blockSize1
ADD.U32 %in_scale, %blockSize0, %in_scale
ADD.U32 %out_addr, %blockSize1, %out_addr
JMPC %cntY, $BLOCK_ENTRY
##