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

.var .u32  %in_scale_lsb
.var .u32  %in_scale_msb
.var .u32  %in_super_scale
.var .u32  %out_scale
.var .u32  %in_scale_lsb_tmp
.var .u32  %out_scale_tmp
.var .u32  %inUnrollStride
.var .u32  %outUnrollStride
.var .u32  %stride_per_block
.var .u32  %in_stride_z
.var .u32  %out_stride_z
.var .u16  %tbx
.var .u16  %cntX
.var .u16  %one
.var .u16  %in_stride_y
.var .u16  %out_stride_y

LDPARAM.U16 [tbDim.x], %tbx
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %tbx, %one, %cntX
MOVE.U16 1, %one

LDPARAM.U32 [in_scale_lsb], %in_scale_lsb
LDPARAM.U32 [in_scale_msb], %in_scale_msb
LDPARAM.U32 [in_super_scale], %in_super_scale
LDPARAM.U32 [out_scale], %out_scale
LDPARAM.U32 [inUnrollStride], %inUnrollStride
LDPARAM.U32 [outUnrollStride], %outUnrollStride
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

// super group = 256, group = 32, 8 scales per super-group
$BLOCK_X:
LOADCONT %in_super_scale, IC
##
LOADCONT %in_scale_msb, IH
##
MOVE.U32 %in_scale_lsb, %in_scale_lsb_tmp
MOVE.U32 %out_scale, %out_scale_tmp

// elements 0,1
LOADALN.Z32 %in_scale_lsb_tmp, %in_stride_y, %in_stride_z, MB
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
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

// elements 2,3
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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

// elements 4,5
ADD.U32 %in_scale_lsb_tmp, %inUnrollStride, %in_scale_lsb_tmp
LOADALN.Z32 %in_scale_lsb_tmp, %in_stride_y, %in_stride_z, MB
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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

// elements 6,7
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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA

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
ADD.U32 %out_scale_tmp, %outUnrollStride, %out_scale_tmp
STOREALN.Z32 %out_scale_tmp, %out_stride_y, %out_stride_z, VA
##

SUB.U16 %cntX, %one, %cntX
ADD.U32 %in_super_scale, %stride_per_block, %in_super_scale
ADD.U32 %in_scale_lsb, %stride_per_block, %in_scale_lsb
ADD.U32 %in_scale_msb, %stride_per_block, %in_scale_msb
ADD.U32 %out_scale, %stride_per_block, %out_scale
JMPC %cntX, $BLOCK_X
##
