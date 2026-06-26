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
.var .u32  %out_addr
.var .u32  %lut_addr
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %inStrideZ
.var .u32  %outStrideZ
.var .u16  %cntX
.var .u16  %cntY
.var .u16  %one
.var .u16  %inStrideY
.var .u16  %outStrideY
.var .u16  %loop0
.var .u16  %unRoll

LDPARAM.U16 [tbDim.x], %cntY
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %cntY, %one, %cntX
MOVE.U16 1, %one
MOVE.U16 2, %loop0
MOVE.U16 2, %unRoll
LDPARAM.U32 [in_wq], %in_wq
LDPARAM.U32 [in_sign], %in_sign
LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [out_addr], %out_addr
LDPARAM.U32 [lut_addr], %lut_addr
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

$LOOP_0:
LDPARAM.U16 [inStrideY1], %inStrideY
LDPARAM.U32 [inStrideZ1], %inStrideZ
LOADALN.Z32 %in_sign, %inStrideY, %inStrideZ, ID
##
$UNROLL_0:
//----------------------------------------------------------------------------------------------------
// codebook   [K/256]      |  [8]   |  [2]     |  [2]      | [2][N]
// codebook   [z]          |  [y]   |  [loop]  |  [unroll] | [x]
//            [grid.y]*[z] |  [y]              |           | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
// qsign      [K/256]       |  [8]  |  [2]     |  [16][N]
// qsign      [z]           |  [y]  |  [loop]  |  [x]
//            [grid.y]*[z]  |  [y]  |          |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
LDPARAM.U16 [inStrideY0], %inStrideY
LDPARAM.U32 [inStrideZ0], %inStrideZ
LOADALN.Z32 %in_wq, %inStrideY, %inStrideZ, MB
UNPACK.U8 MB, VAB
##
COPY VAB, IAB
MSHIFTADD VA, %lut_addr, 2, DVA
##
LOAD [DVA], MA
UNPACK.U8 MA, VAB
MADD DVA, 2, DVA
##
COPYEXT VAB, IGH
CVT.U8.F16 VA, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements0
//-----------------------------------------------------------------------------------------------------
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA
##
COPY VA, ID
CVT.U8.F16 IH, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements1
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
LOAD [DVA], MA
UNPACK.U8 MA, VAB
MADD DVA, 2, DVA
##
COPYEXT VAB, IGH
CVT.U8.F16 VA, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements2
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
CVT.U8.F16 IH, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements3
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
MSHIFTADD IB, %lut_addr, 2, DVA
##
LOAD [DVA], MA
UNPACK.U8 MA, VAB
MADD DVA, 2, DVA
##
COPYEXT VAB, IGH
CVT.U8.F16 VA, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements4
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
CVT.U8.F16 IH, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements5
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
LOAD [DVA], MA
UNPACK.U8 MA, VAB
MADD DVA, 2, DVA
##
COPYEXT VAB, IGH
CVT.U8.F16 VA, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements6
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
CVT.U8.F16 IH, VA
##
COPYSIGN.F16 ID, VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements7
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_addr, %outUnrollStride, %out_addr
STOREALN.Z32  %out_addr, %outStrideY, %outStrideZ, VA
SHL.U16 ID, 1, VA 
##
COPY VA, ID
##
SUB.U16 %unRoll, %one, %unRoll
ADD.U32 %in_wq, %inUnrollStride, %in_wq
ADD.U32 %out_addr, %outUnrollStride, %out_addr
JMPC %unRoll, $UNROLL_0
MOVE.U16 2, %unRoll

SUB.U16 %loop0, %one, %loop0
ADD.U32 %in_sign, %inUnrollStride, %in_sign
JMPC %loop0, $LOOP_0
MOVE.U16 2, %loop0
SUB.U16 %cntX, %one, %cntX
LDPARAM.U32 [inBlockXSize0], DS13
ADD.U32 %in_wq, DS13, %in_wq
LDPARAM.U32 [inBlockXSize1], DS13
ADD.U32 %in_sign, DS13, %in_sign

LDPARAM.U32 [inDequantXSize], DS13
ADD.U32 %in_scale, DS13, %in_scale
LDPARAM.U32 [outBlockXSize], DS13
ADD.U32 %out_addr, DS13, %out_addr
JMPC %cntX, $BLOCK_ENTRY

LDPARAM.U16 [blockX], %cntX
SUB.U16 %cntY, %one, %cntY
LDPARAM.U32 [inBlockYSize0], DS13
ADD.U32 %in_wq, DS13, %in_wq
LDPARAM.U32 [inBlockYSize1], DS13
ADD.U32 %in_sign, DS13, %in_sign

LDPARAM.U32 [inDequantYSize], DS13
ADD.U32 %in_scale, DS13, %in_scale
LDPARAM.U32 [outBlockYSize], DS13
ADD.U32 %out_addr, DS13, %out_addr
JMPC %cntY, $BLOCK_ENTRY

##
