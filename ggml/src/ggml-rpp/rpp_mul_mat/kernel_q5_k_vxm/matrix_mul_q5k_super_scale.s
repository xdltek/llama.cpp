.param   .U32 in_scale_lsb
.param   .U32 in_scale_msb
.param   .U32 in_super_scale
.param   .U32 out_scale
.param   .U32 in_stride_y
.param   .U32 in_stride_z
.param   .U32 out_stride_y
.param   .U32 out_stride_z
.param   .U32 inUnrollStride
.param   .U32 outUnrollStride
.param   .U32 stride_per_block
.param   .U16 blockX
.param   .U16 sign


.var .u32  %in_scale_lsb
.var .u32  %in_scale_msb
.var .u32  %in_super_scale
.var .u32  %out_scale
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %stride_per_block
.var .u32  %in_stride_z
.var .u32  %out_stride_z
.var .u16  %cntX
.var .u16  %one
.var .u16  %in_stride_y
.var .u16  %out_stride_y
.var .u16  %sign

LDPARAM.U16 [tbDim.x], %in_stride_y
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %in_stride_y, %one, %cntX
MOVE.U16 1, %one

LDPARAM.U32 [in_scale_lsb], %in_scale_lsb
LDPARAM.U32 [in_scale_msb], %in_scale_msb
LDPARAM.U32 [in_super_scale], %in_super_scale
LDPARAM.U32 [out_scale], %out_scale
LDPARAM.U32 [outUnrollStride], %outUnrollStride
LDPARAM.U32 [inUnrollStride], %inUnrollStride
LDPARAM.U32 [stride_per_block], %stride_per_block
LDPARAM.U32 [in_stride_z], %in_stride_z
LDPARAM.U32 [out_stride_z], %out_stride_z
LDPARAM.U16 [in_stride_y], %in_stride_y
LDPARAM.U16 [out_stride_y], %out_stride_y
LDPARAM.U16 [sign], %sign


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
// group = 8
//----------------------------------------------------------------------------------------------------
// in_super_scale   [K/256]  |     | [N]
// in_super_scale   [sg]     |     | [N]
// in_super_scale   [z]      | [1] | [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// qscale_per_word = 4
// in_scale_lsb   [K/256]  | [2]                  |  [qscale_per_word][N]
// in_scale_lsb   [K/256]  | [2]                  |  [qscale_per_word][N]
// in_scale_lsb   [z]      | [unroll]             |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// qscale_per_word = 8
// in_scale_msb   [K/256]  | [1]                  |  [qscale_per_word][N]
// in_scale_msb   [K/256]  | [1]                  |  [qscale_per_word][N]
// in_scale_msb   [z]      | [1]                  |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// out_scale   [K/256]     | [8]                  |  [N]
// out_scale   [z]         | [8]                  |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------

//For super zero, need to mulitpy with -1
$BLOCK_X:
LOADCONT %in_super_scale, MB
MUL.F16 MB, %sign, VA
##
COPY VA, IC
LOADCONT %in_scale_msb, IH
##
LOADALN.Z32 %in_scale_lsb, %in_stride_y, %in_stride_z, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
SHL.U16 IH, 4, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
MUL.F16 VA, IC, VA
##
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
SHL.U16 IH, 2, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements1
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
AND.U16 IH, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements2
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
SHR.U16 IH, 2, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements3
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##

ADD.U32 %in_scale_lsb, %inUnrollStride, %in_scale_lsb
LOADALN.Z32 %in_scale_lsb, %in_stride_y, %in_stride_z, MB
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
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements4
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
SHR.U16 IH, 6, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements5
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
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
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements6
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
SHR.U16 IH, 10, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
MUL.F16 VA, IC, VA
##
//-----------------------------------------------------------------------------------------------------
//elements7
//-----------------------------------------------------------------------------------------------------
ADD.U32 %out_scale, %outUnrollStride, %out_scale
STOREALN.Z32  %out_scale, %out_stride_y, %out_stride_z, VA
##
SUB.U16 %cntX, %one, %cntX
ADD.S32 %in_super_scale, %stride_per_block, %in_super_scale
ADD.S32 %in_scale_lsb, %stride_per_block, %in_scale_lsb
ADD.S32 %in_scale_msb, %stride_per_block, %in_scale_msb
ADD.S32 %out_scale, %stride_per_block, %out_scale
JMPC %cntX, $BLOCK_X
##