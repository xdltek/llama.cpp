// only one of shadowType set to 1
// shadowX:
// ┌───────┬───────┬───┐
// │       │       │###│
// ├───────┼───────┼───┤
// │       │       │###│
// └───────┴───────┴───┘
// shadowY:
// ┌───────────────────┐
// │                   │
// │                   │
// ├───────────────────┤
// │###################│
// └───────────────────┘

.param .u32 input
.param .u32 output
.param .u32 inputStrideY
.param .u32 blockStrideX
.param .u32 blockStrideY
.param .u16 ifShadowX
.param .u16 ifShadowY

.var .u32 %input
.var .u32 %output
.var .u32 %tmp0
.var .u32 %tmp1
.var .u32 %inputStrideY
.var .u32 %blockStrideX
.var .u32 %blockStrideY
.var .u32 %tmp32_1

.var .u16 %one
.var .u16 %tbDim.x
.var .u16 %tbDim.y
.var .u16 %tmp16_1
.var .u16 %tmp16_2
.var .u16 %tmp16_3
.var .u16 %ifShadowX
.var .u16 %ifShadowY
.var .u16 %mask
.var .u16 %seven

MOVE.U16 1, %one

LDPARAM.U16 [ifShadowX], %ifShadowX
LDPARAM.U16 [ifShadowY], %ifShadowY

JMPC %ifShadowX, $SHADOW_X

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tbDim.x], %tbDim.x
LDPARAM.U16 [tbDim.y], %tbDim.y
LDPARAM.U16 [tid_xyz], %tmp16_3
CFGXYZ %tbDim.x, %tbDim.y, %tmp16_3
JMPC %ifShadowY, $SHADOW_Y_CFG
JMP $BLOCK_ENTRY

$SHADOW_X:
JMPC %blockIdx.x, $SHADOW_X_CFG
LDPARAM.U16 [tbDim.x], %tbDim.x
LDPARAM.U16 [tbDim.y], %tbDim.y
LDPARAM.U16 [tid_xyz], %tmp16_3
CFGXYZ %tbDim.x, %tbDim.y, %tmp16_3
JMP $BLOCK_ENTRY

$SHADOW_X_CFG:
LDPARAM.U16 [grid_dim_x], %tmp16_1
ADD.U16 %blockIdx.x, %one, %tmp16_2
SUB.U16 %tmp16_1, %tmp16_2, %tmp16_3
JMPNC %tmp16_3, $SHADOW_CFG   // if blockIdx.x == grid_dim_x - 1, cfg shadowX
JMP $BLOCK_ENTRY

$SHADOW_Y_CFG:
LDPARAM.U16 [grid_dim_y], %tmp16_1
ADD.U16 %blockIdx.y, %one, %tmp16_2
SUB.U16 %tmp16_1, %tmp16_2, %tmp16_3
JMPNC %tmp16_3, $SHADOW_CFG    // if blockIdx.y == grid_dim_y - 1, cfg shadowY
JMP $BLOCK_ENTRY

$SHADOW_CFG:
LDPARAM.U16 [dimx_shadow], %tmp16_1
LDPARAM.U16 [dimy_shadow], %tmp16_2
LDPARAM.U16 [xyzcfg_shadow], %tmp16_3
CFGXYZ %tmp16_1, %tmp16_2, %tmp16_3

$BLOCK_ENTRY:
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
LDPARAM.U32 [input], %input
LDPARAM.U32 [output], %output
LDPARAM.U32 [inputStrideY], %inputStrideY
LDPARAM.U32 [blockStrideX], %blockStrideX
LDPARAM.U32 [blockStrideY], %blockStrideY

// output address = %output
//                  + blockIdx.y * blockStrideY
//                  + blockIdx.x * blockStrideX
CVT.U16.U32 %blockIdx.y, %tmp0
MUL.U32 %tmp0, %blockStrideY, %tmp0
ADD.U32 %output, %tmp0, %output
CVT.U16.U32 %blockIdx.x, %tmp0
MUL.U32 %tmp0, %blockStrideX, %tmp0
ADD.U32 %output, %tmp0, %output

// input address = %input
//                 + (threadIdx.x + blockIdx.x * blockDim.x) * inputStrideY
//                 + (threadIdx.y + blockIdx.y * blockDim.y) * sizeof(bf16)
LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, MB
COPYEXT MB, IE
DECTID.X MB, %tmp1, VA  // threadIdx.x
##
CVT.U16.U32 VA, VAB
MUL.U16.WIDE %blockIdx.x, %tbDim.x, %tmp32_1
##
ADD.U32 VAB, %tmp32_1, VAB
##
MUL.U32 VAB, %inputStrideY, VAB
##
ADD.U32 VAB, %input, IGH
##
DECTID.Y IE, %tmp1, VA  // threadIdx.y
##
CVT.U16.U32 VA, VAB
MUL.U16.WIDE %blockIdx.y, %tbDim.y, %tmp32_1
##
ADD.U32 VAB, %tmp32_1, VAB
##
SHL.U32 VAB, %one, VAB
##
ADD.U32 IGH, VAB, VAB
##
MMOV VAB, DVA
##

MOVE.U16 0, %tmp16_1
MOVE.U16 7, %seven
MOVE.U16 1, %mask
LOADMASK [DVA], %mask, MA
ADD.U16 MA, %tmp16_1, VA
##
$LOAD_LOOP:
SHL.U16 %mask, %one, %mask
LOADMASK [DVA], %mask, MA
MERGEMASK VA, MA, %mask, VA
##
SUB.U16 %seven, %one, %seven
JMPC %seven, $LOAD_LOOP

STORECONT %output, VA
##
