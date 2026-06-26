.param   .U32 in_scale
.param   .U32 in_super_scale
.param   .U32 out_scale
.param   .U32 LUT
.param   .U32 in_stride_y
.param   .U32 in_stride_z
.param   .U32 out_stride_y
.param   .U32 out_stride_z
.param   .U32 inUnrollStride
.param   .U32 outUnrollStride
.param   .U32 stride_per_block
.param   .U16 blockX

.var .u32  %in_scale
.var .u32  %in_super_scale
.var .u32  %out_scale
.var .u32  %LUT
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %stride_per_block
.var .u32  %in_stride_z
.var .u32  %out_stride_z
.var .u16  %cntX
.var .u16  %one
.var .u16  %in_stride_y
.var .u16  %out_stride_y

LDPARAM.U16 [tbDim.x], %in_stride_y
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %in_stride_y, %one, %cntX
MOVE.U16 1, %one

LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [LUT], %LUT
LDPARAM.U32 [in_super_scale], %in_super_scale
LDPARAM.U32 [out_scale], %out_scale
LDPARAM.U32 [outUnrollStride], %outUnrollStride
LDPARAM.U32 [inUnrollStride], %inUnrollStride
LDPARAM.U32 [stride_per_block], %stride_per_block
LDPARAM.U32 [in_stride_z], %in_stride_z
LDPARAM.U32 [out_stride_z], %out_stride_z
LDPARAM.U16 [in_stride_y], %in_stride_y
LDPARAM.U16 [out_stride_y], %out_stride_y


LDPARAM.U16 [blockX], %cntX
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
FORWARD DVA
##
//----------------------------------------------------------------------------------------------------
// super group = 256
//----------------------------------------------------------------------------------------------------
// in_super_scale   [K/256]  |     | [N]
// in_super_scale   [sg]     |     | [N]
// in_super_scale   [z]      | [1] | [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// out_scale   [K/256]     | [16]     |  [N]
// out_scale   [z]         | [16]     |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// in_scale   [K/256]  | [4]          |  [4][N]
// in_scale   [K/256]  | [4]          |  [4][N]
// in_scale   [z]      | [unroll]     |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
$BLOCK_X:
LOADCONT %in_super_scale, IC
##
LOADALN.Z32 %in_scale, %in_stride_y, %in_stride_z, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
LOAD [%LUT + (VA << 1)], MA
MUL.F16 MA, IC, VA
##
UNPACK.U4.HIGH IG, VAB
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
COPY VAB, IAB
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements1
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IA << 1)], MA
MUL.F16 MA, IC, VA
##
//--------------------------------------------------------------------------------------------
//elements2
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//--------------------------------------------------------------------------------------------
//elements3
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
ADD.U32 %in_scale, %inUnrollStride, %in_scale
LOADALN.Z32 %in_scale, %in_stride_y, %in_stride_z, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
LOAD [%LUT + (VA << 1)], MA
MUL.F16 MA, IC, VA
##
//--------------------------------------------------------------------------------------------
//elements4
//--------------------------------------------------------------------------------------------
UNPACK.U4.HIGH IG, VAB
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
COPY VAB, IAB
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements5
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IA << 1)], MA
MUL.F16 MA, IC, VA
##
//----------------------------------------------------------------------------------------------
//elements6
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//----------------------------------------------------------------------------------------------
//elements7
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
ADD.U32 %in_scale, %inUnrollStride, %in_scale
LOADALN.Z32 %in_scale, %in_stride_y, %in_stride_z, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
LOAD [%LUT + (VA << 1)], MA
MUL.F16 MA, IC, VA
##
//--------------------------------------------------------------------------------------------
//elements8
//--------------------------------------------------------------------------------------------
UNPACK.U4.HIGH IG, VAB
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
COPY VAB, IAB
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements9
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IA << 1)], MA
MUL.F16 MA, IC, VA
##
//----------------------------------------------------------------------------------------------
//elements10
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//----------------------------------------------------------------------------------------------
//elements11
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
ADD.U32 %in_scale, %inUnrollStride, %in_scale
LOADALN.Z32 %in_scale, %in_stride_y, %in_stride_z, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
LOAD [%LUT + (VA << 1)], MA
MUL.F16 MA, IC, VA
##
//--------------------------------------------------------------------------------------------
//elements12
//--------------------------------------------------------------------------------------------
UNPACK.U4.HIGH IG, VAB
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
COPY VAB, IAB
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements13
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IA << 1)], MA
MUL.F16 MA, IC, VA
##
//----------------------------------------------------------------------------------------------
//elements14
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
LOAD [%LUT + (IB << 1)], MA
MUL.F16 MA, IC, VA
##
//----------------------------------------------------------------------------------------------
//elements15
//--------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
SUB.U16 %cntX, %one, %cntX
ADD.S32 %in_super_scale, %stride_per_block, %in_super_scale
ADD.S32 %in_scale, %stride_per_block, %in_scale
ADD.S32 %out_scale, %stride_per_block, %out_scale
JMPC %cntX, $BLOCK_X
##