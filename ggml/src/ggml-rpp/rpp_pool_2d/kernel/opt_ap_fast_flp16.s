//ENTRY:
.param .u32 input    //input
.param .u32 out    //output
.param .u32 inStrideY
.param .u32 outStrideY
.param .u32 inStrideZ
.param .u32 outStrideZ
.param .u32 inBlockSize
.param .u32 outBlockSize
.param .u32 offset0
.param .u32 offset1
.param .u16 rpt0
.param .u16 Un
.param .u32 inUnStride
.param .u32 outUnStride
.param .u16 Bn
.param .u16 rpt_m1
.param .u32 ap_scale
.param .u32 inBlockYStride
.param .u32 outBlockYStride
.param .u32 inBlockZStride
.param .u32 outBlockZStride
.param .u16 tail_block_y      //last block in dimension y

.var .u32 %input
.var .u32 %out
.var .u32 %offset0
.var .u32 %offset1
.var .u32 %offset2
.var .u32 %StrideY
.var .u32 %StrideZ

.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3
.var .u16 %S4
.var .u16 %S5
.var .u16 %S6

LDPARAM.U16 [tail_block_y], %S4
JMPNC %S4,$LAST_BLOCK_IN_Z
MOVE.U16 %blockIdx.y, %S5
LDPARAM.U16 [grid_dim_y], %S6
JMPC %blockIdx.y, $BLOCK_ENTRY_LAST
JMP $BLOCK_ENTRY
$LAST_BLOCK_IN_Z:
MOVE.U16 %blockIdx.z, %S5
LDPARAM.U16 [grid_dim_z], %S6
JMPC %blockIdx.z, $BLOCK_ENTRY_LAST
$BLOCK_ENTRY:
LDPARAM.U16 [tid_xyz], %S0
LDPARAM.U16 [tbDim.x], %S3
LDPARAM.U16 [tbDim.y], %S1
CFGXYZ %S3, %S1, %S0
JMP $BLOCK_ENTRY_1
$BLOCK_ENTRY_LAST:
MOVE.U16 1, %S2
ADD.U16 %S5, %S2, %S5
SUB.U16 %S6, %S5, %S6
JMPC %S6, $BLOCK_ENTRY_1
//config the block dimension for the last block
LDPARAM.U16 [xyzcfg_shadow], %S0
LDPARAM.U16 [dimx_shadow], %S3
LDPARAM.U16 [dimy_shadow], %S1
CFGXYZ %S3, %S1, %S0
$BLOCK_ENTRY_1:
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
MOVE.U16 1, %S4
LDPARAM.U32 [input], %input
//grid.x offset
LDPARAM.U32 [inBlockSize], %offset1
MUL.U16.WIDE %blockIdx.x, %S4, %offset0
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.U32 %input, %offset1, %input
//grid.y offset
LDPARAM.U32 [inBlockYStride], %offset1
MUL.U16.WIDE %blockIdx.y, %S4, %offset0
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.U32 %input, %offset1, %input
//grid.z offset
LDPARAM.U32 [inBlockZStride], %offset1
MUL.U16.WIDE %blockIdx.z, %S4, %offset0
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.U32 %input, %offset1, %input

LDPARAM.U16 [rpt0], %S0
LDPARAM.U16 [rpt_m1], %S1

LDPARAM.U32 [offset0], %offset0
LDPARAM.U32 [offset1], %offset1
LDPARAM.U32 [inStrideY], %StrideY
LDPARAM.U32 [inStrideZ], %StrideZ
MOVE.U32 0, VAB
##
$LOOP:
COPY VAB, IAB
LOADALN.Z32 %input, %StrideY, %StrideZ, MB
CVT.F16.F32 MB, VAB
##
ADD.S32 %input, %offset0, %input  //change
ADD.F32 VAB, IAB, VAB
##
SUB.S16 %S1, %S4, %S1
JMPC %S1, $LOOP
SUB.S16 %S0, %S4, %S0
ADD.S32 %input, %offset1, %input
LDPARAM.U16 [rpt_m1], %S1
JMPC %S0, $LOOP
LDPARAM.U32 [ap_scale], %offset2
MUL.F32 VAB, %offset2, VAB
##
CVT.F32.F16 VAB, VA
##

LDPARAM.U32 [out], %out
//grid.x offset c group
LDPARAM.U32 [outBlockSize], %offset1
MUL.U16.WIDE %blockIdx.x, %S4, %offset0
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.U32 %out, %offset1, %out
//grid.y offset
LDPARAM.U32 [outBlockYStride], %offset1
MUL.U16.WIDE %blockIdx.y, %S4, %offset0
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.U32 %out, %offset1, %out
//grid.z offset
LDPARAM.U32 [outBlockZStride], %offset1
MUL.U16.WIDE %blockIdx.z, %S4, %offset0
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.U32 %out, %offset1, %out

LDPARAM.U32 [outStrideY], %StrideY
LDPARAM.U32 [outStrideZ], %StrideZ
STOREALN.Z32 %out, %StrideY, %StrideZ, VA
##
