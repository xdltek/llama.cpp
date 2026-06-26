.param   .U32 in_wq
.param   .U32 in_sign
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
.param   .U32 inBlockYSize0
.param   .U32 inBlockXSize1
.param   .U32 inBlockYSize1
.param   .U32 inDequantXSize
.param   .U32 inDequantYSize
.param   .U32 outBlockXSize
.param   .U32 outBlockYSize
.param   .U16 blockX
.param   .U16 blockY


.var .u32  %in_wq
.var .u32  %in_sign
.var .u32  %in_scale
.var .u32  %lut_addr
.var .u32  %out_addr
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %inStrideZ
.var .u32  %outStrideZ
.var .u16  %cntX
.var .u16  %cntY
.var .u16  %one
.var .u16  %inStrideY
.var .u16  %outStrideY
.var .u16  %unroll1
.var .u16  %loop0

LDPARAM.U16 [tbDim.x], %cntY
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %cntY, %one, %cntX
MOVE.U16 1, %one
MOVE.U16 4, %loop0
MOVE.U16 8, %unroll1
LDPARAM.U32 [in_wq], %in_wq
LDPARAM.U32 [in_sign], %in_sign
LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [lut_addr], %lut_addr
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
$LOOP_0:
//----------------------------------------------------------------------------------------------------
// scale      [K/256]       |  [4]  |  [4]       |  [N]
// scale      [z]           |  [y]  |  [loop]    |  [x]
//            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
// qsign      [K/256]       |  [4]  |  [4]       |  [16][N]
// qsign      [z]           |  [y]  |  [loop]    |  [x]
//            [grid.y]*[z]  |  [y]               |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
LDPARAM.U16 [dequantStrideY], %inStrideY
LDPARAM.U32 [dequantStrideZ], %inStrideZ
LOADALN.Z32 %in_scale, %inStrideY, %inStrideZ, IC
##
LOADALN.Z32 %in_sign, %inStrideY, %inStrideZ, ID
##
$UNROLL_0:
//----------------------------------------------------------------------------------------------------
// codebook_nolut   [K/256]      |  [4]  |  [4]    | [2]   |  [8][N]
// codebook_nolut   [z]          |  [y]  |  [loop] |       |  [x]
//                  [grid.y]*[z] |  [y]  |                 |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
LDPARAM.U16 [inStrideY0], %inStrideY
LDPARAM.U32 [inStrideZ0], %inStrideZ
LOADALN.Z32 %in_wq, %inStrideY, %inStrideZ, MB
MOVE.U16 MB, VA
##

$FIRST_8:
COPYEXT VA, IG
AND.U16 VA, 0x3, VA
##
LOAD [%lut_addr + (VA << 1)], MA
COPYSIGN.F16 ID, MA, VA
##
MUL.F16 VA, IC, VA
##
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA
##
COPY VA, ID
SHR.U16 IG, 2, VA
##
SUB.U16 %unroll1, %one, %unroll1
ADD.U32 %out_addr, %outUnrollStride, %out_addr
JMPC %unroll1, $FIRST_8

MOVE.U16 8, %unroll1
ADD.U32 %in_wq, %inUnrollStride, %in_wq
TERM VA
LOADALN.Z32 %in_wq, %inStrideY, %inStrideZ, MB
MOVE.U16 MB, VA
##

$SECOND_8:
COPYEXT VA, IG
AND.U16 VA, 0x3, VA
##
LOAD [%lut_addr + (VA << 1)], MA
COPYSIGN.F16 ID, MA, VA
##
MUL.F16 VA, IC, VA
##
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA
##
COPY VA, ID
SHR.U16 IG, 2, VA
##
SUB.U16 %unroll1, %one, %unroll1
ADD.U32 %out_addr, %outUnrollStride, %out_addr
JMPC %unroll1, $SECOND_8
TERM VA
##

MOVE.U16 8, %unroll1
SUB.U16 %loop0, %one, %loop0
ADD.U32 %in_sign, %inUnrollStride, %in_sign
ADD.U32 %in_scale, %inUnrollStride, %in_scale
ADD.U32 %in_wq, %inUnrollStride, %in_wq
JMPC %loop0, $LOOP_0
MOVE.U16 4, %loop0


SUB.U16 %cntX, %one, %cntX
LDPARAM.U32 [inBlockXSize0], DS13
ADD.U32 %in_wq, DS13, %in_wq

LDPARAM.U32 [inDequantXSize], DS13
ADD.U32 %in_scale, DS13, %in_scale
ADD.U32 %in_sign, DS13, %in_sign

LDPARAM.U32 [outBlockXSize], DS13
ADD.U32 %out_addr, DS13, %out_addr
JMPC %cntX, $BLOCK_ENTRY

LDPARAM.U16 [blockX], %cntX
SUB.U16 %cntY, %one, %cntY
LDPARAM.U32 [inBlockYSize0], DS13
ADD.U32 %in_wq, DS13, %in_wq

LDPARAM.U32 [inDequantYSize], DS13
ADD.U32 %in_scale, DS13, %in_scale
ADD.U32 %in_sign, DS13, %in_sign

LDPARAM.U32 [outBlockYSize], DS13
ADD.U32 %out_addr, DS13, %out_addr
JMPC %cntY, $BLOCK_ENTRY
##