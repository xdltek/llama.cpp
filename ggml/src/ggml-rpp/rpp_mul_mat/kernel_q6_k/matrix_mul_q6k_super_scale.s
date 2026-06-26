.param   .U32 in_scale
.param   .U32 in_super_scale
.param   .U32 out_scale
.param   .U32 in_stride_y
.param   .U32 in_stride_z
.param   .U32 out_stride_y
.param   .U32 out_stride_z
.param   .U32 stride_per_row
.param   .U32 stride_per_block
.param   .U16 blockX
.param   .U16 blockY

.var .u32  %in_scale
.var .u32  %in_super_scale
.var .u32  %out_scale
.var .u32  %in_scale_tmp
.var .u32  %in_super_scale_tmp
.var .u32  %out_scale_tmp
.var .u32  %stride_per_row
.var .u32  %stride_per_block
.var .u32  %in_stride_z
.var .u32  %out_stride_z
.var .u16  %stride_y
.var .u16  %cntX
.var .u16  %cntY
.var .u16  %one
.var .u16  %strideY

LDPARAM.U16 [tbDim.x], %cntY
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %cntY, %one, %cntX
MOVE.U16 1, %one

LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [in_super_scale], %in_super_scale
LDPARAM.U32 [out_scale], %out_scale
LDPARAM.U32 [stride_per_row], %stride_per_row
LDPARAM.U32 [stride_per_block], %stride_per_block
LDPARAM.U32 [in_stride_z], %in_stride_z
LDPARAM.U32 [out_stride_z], %out_stride_z
LDPARAM.U16 [in_stride_y], %stride_y
LDPARAM.U16 [blockX], %cntX
LDPARAM.U16 [blockY], %cntY
MOVE.U32 %in_super_scale, %in_super_scale_tmp
MOVE.U32 %in_scale, %in_scale_tmp
MOVE.U32 %out_scale, %out_scale_tmp

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
// group = 16
//----------------------------------------------------------------------------------------------------
// in_super_scale   [K/256]  |     | [N]
// in_super_scale   [sg]     |     | [N]
// in_super_scale   [z]      | [1] | [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// qscale_per_word = 2
// in_scale   [K/256]  | [256/16/qscale_per_word]  |  [qscale_per_word][N]
// in_scale   [K/256]  | [256/32]                  |  [qscale_per_word][N]     
// in_scale   [z]      | [unroll]                  |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
// out_scale   [K/256]  | [256/16]                 |  [N] 
// out_scale   [z]      | [unroll]                 |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------
$BLOCK_X:
LOADCONT %in_super_scale_tmp, IC
##
//----------------------------------------------------------------------------------------------------
//IA = out_scale[y][0][N]
//IB = out_scale[y][1][N]
//out_scale[0][N] = IA * IC
//out_scale[1][N] = IB * IC
//----------------------------------------------------------------------------------------------------
$BLOCK_Y:
LOADALN.Z32 %in_scale_tmp, %stride_y, %in_stride_z, MB
UNPACK.U8 MB, VAB
##
COPY VAB, IAB
CVT.S8.F16 VA, VA
##
COPY VA, IA
CVT.S8.F16 IB, VA
##
COPY VA, IB
MUL.F16 IA, IC, VA
##
STOREALN.Z32 %out_scale_tmp, %stride_y, %out_stride_z, VA
MUL.F16 IB, IC, VA
##
ADD.U32 %out_scale_tmp, %stride_per_row, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %stride_y, %out_stride_z, VA
##

SUB.U16 %cntY, %one, %cntY
ADD.U32 %out_scale_tmp, %stride_per_row, %out_scale_tmp
ADD.U32 %in_scale_tmp, %stride_per_row, %in_scale_tmp
JMPC %cntY, $BLOCK_Y

LDPARAM.U16 [blockY], %cntY
SUB.U16 %cntX, %one, %cntX
ADD.S32 %in_super_scale, %stride_per_block, %in_super_scale
ADD.S32 %in_scale, %stride_per_block, %in_scale

ADD.S32 %out_scale, %stride_per_block, %out_scale
MOVE.U32 %in_super_scale, %in_super_scale_tmp
MOVE.U32 %in_scale, %in_scale_tmp
MOVE.U32 %out_scale, %out_scale_tmp
JMPC %cntX, $BLOCK_X
##
