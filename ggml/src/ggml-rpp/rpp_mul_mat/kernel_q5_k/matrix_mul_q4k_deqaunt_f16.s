.param   .U32 in_wq
.param   .U32 in_scale
.param   .U32 in_zero
.param   .U32 out_addr
.param   .U32 lut_addr
.param   .U32 inStrideY0
.param   .U32 inStrideZ0
.param   .U32 dequantStrideY
.param   .U32 dequantStrideZ
.param   .U32 outStrideY
.param   .U32 outStrideZ
.param   .U32 inUnrollStride
.param   .U32 outUnrollStride
.param   .U32 inBlockXSize0
.param   .U32 inBlockYSize0
.param   .U32 inDequantXSize
.param   .U32 inDequantYSize
.param   .U32 outBlockXSize
.param   .U32 outBlockYSize
.param   .U16 blockX
.param   .U16 blockY


.var .u32  %in_wq
.var .u32  %in_scale
.var .u32  %in_zero
.var .u32  %out_addr
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %inStrideZ
.var .u32  %outStrideZ
.var .u32  %blockSize0
.var .u16  %cntX
.var .u16  %cntY
.var .u16  %one
.var .u16  %inStrideY
.var .u16  %outStrideY
.var .u16  %loop0


LDPARAM.U16 [tbDim.x], %cntY
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %cntY, %one, %cntX
MOVE.U16 1, %one
MOVE.U16 8, %loop0
LDPARAM.U32 [in_wq], %in_wq
LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [in_zero], %in_zero
LDPARAM.U32 [out_addr], %out_addr
LDPARAM.U32 [inUnrollStride], %inUnrollStride
LDPARAM.U32 [outUnrollStride], %outUnrollStride
LDPARAM.U16 [blockX], %cntX
LDPARAM.U16 [blockY], %cntY
LDPARAM.U16 [outStrideY], %outStrideY
LDPARAM.U32 [outStrideZ], %outStrideZ
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
// in_scale   [K/256]      | [8]    | [N]
// in_scale   [z]          | [y]    | [x]
//            [grid.y]*[z] | [y]    | [grid.x]*[x]
//-----------------------------------------------------------------------------------------------------

LDPARAM.U16 [dequantStrideY], %inStrideY
LDPARAM.U32 [dequantStrideZ], %inStrideZ
LOADALN.Z32 %in_scale, %inStrideY, %inStrideZ, IC
##
LOADALN.Z32 %in_zero, %inStrideY, %inStrideZ, ID
##
$BLOCK_0:
//----------------------------------------------------------------------------------------------------
//group = 8
// in_lsb     [K/256]      |  [8]  |  [32/4]    |  [4][N]
// in_lsb     [z]          |  [y]  |  [unroll]  |  [x]
//            [grid.y]*[z] |  [y]               |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
// out_addr   [N/32]   | [K/256]      | [8]     | [32/4]    | [4]        | [32]
// out_addr            | [z]          | [y]     | [unroll0] | [unroll1]  | [x]
//            [grid.x] | [grid.y]*[z] | [y]                              | [x]
//-----------------------------------------------------------------------------------------------------
LDPARAM.U16 [inStrideY0], %inStrideY
LDPARAM.U32 [inStrideZ0], %inStrideZ
LOADALN.Z32 %in_wq, %inStrideY, %inStrideZ, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
CVT.U8.F16 VA, VA
##
COPYEXT VA, IE
FMA VA, IC, ID, VA
##
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
CVT.U8.F16 IB, VA
##
COPYEXT VA, IE
FMA VA, IC, ID, VA
##
//-----------------------------------------------------------------------------------------------------
//elements1
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
CVT.U8.F16 VA, VA
##
COPYEXT VA, IE
FMA VA, IC, ID, VA
##
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
CVT.U8.F16 IB, VA
##
COPYEXT VA, IE
FMA VA, IC, ID, VA
##
//-----------------------------------------------------------------------------------------------------
//elements3
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
##

SUB.U16 %loop0, %one, %loop0
ADD.U32 %in_wq, %inUnrollStride, %in_wq
ADD.U32 %out_addr, %outUnrollStride, %out_addr
JMPC %loop0, $BLOCK_0
MOVE.U16 8, %loop0
SUB.U16 %cntX, %one, %cntX

LDPARAM.U32 [inBlockXSize0], %blockSize0
ADD.U32 %in_wq, %blockSize0, %in_wq
LDPARAM.U32 [inDequantXSize], %blockSize0
LDPARAM.U32 [outBlockXSize], DS13
ADD.U32 %in_scale, %blockSize0, %in_scale
ADD.U32 %in_zero, %blockSize0, %in_zero
ADD.U32 %out_addr, DS13, %out_addr
JMPC %cntX, $BLOCK_ENTRY

LDPARAM.U16 [blockX], %cntX
SUB.U16 %cntY, %one, %cntY
LDPARAM.U32 [inBlockYSize0], %blockSize0
ADD.U32 %in_wq, %blockSize0, %in_wq

LDPARAM.U32 [inDequantYSize], %blockSize0
LDPARAM.U32 [outBlockYSize], DS13
ADD.U32 %in_scale, %blockSize0, %in_scale
ADD.U32 %in_zero, %blockSize0, %in_zero
ADD.U32 %out_addr, DS13, %out_addr
JMPC %cntY, $BLOCK_ENTRY
##