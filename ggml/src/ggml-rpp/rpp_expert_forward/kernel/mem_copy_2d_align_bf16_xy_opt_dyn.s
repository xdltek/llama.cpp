//ENTRY:
.param .u32 input      //input
.param .u32 out        //out
.param .u32 inStrideY  //input stride y
.param .u32 inStrideZ  //input stride z
.param .u32 outStrideY //output stride y
.param .u32 outStrideZ //output stride z
.param .u32 inBXStride     //input grid X stride
.param .u32 outBXStride    //output grid X stride
.param .u32 inBYStride     //input grid Y stride
.param .u32 outBYStride    //output grid Y stride
.param .u32 inBZStride     //input grid Z stride
.param .u32 outBZStride    //output grid Z stride
.param .u16 grid_y_tail    //tail block in grid y
.param .u16 grid_z_tail    //tail block in grid z
.param .u16 gridX

.var .u32 %input
.var .u32 %out
.var .u32 %inBXStride
.var .u32 %outBXStride
.var .u32 %blockStride
.var .u16 %inStrideY
.var .u16 %outStrideY
.var .u16 %one
.var .u16 %y
.var .u16 %x
.var .u16 %gridX
.var .u16 %S2
.var .u16 %S3


//----------------------------------------------------------------------------------------------------
// for HW to NHW32
// input    [1]      | [H]  | [W]
//                   | [y]  | [grid.x][x]
// output   [N]      | [H]  | [W32]      
// 			[grid.x] | [y]	| [x]
// H is dynamic, configure through FRAM register [cfg_fft_w4_6]
// Not support grid.y > 1
// For MoE case, the application make sure current epxpert tokens < 256
// The block-X stride is also derived from the dynamic H so the same graph can
// be reused across experts with different token counts.
//----------------------------------------------------------------------------------------------------
LDPARAM.U16 [tbDim.x], %x
LDPARAM.U16 [cfg_fft_w4_6], %y
MUL.U16.LOW %x, %y, %S2
CFGXYZ %x, %y, %S2

LDPARAM.U16 [gridX], %gridX
LDPARAM.U16 [outStrideY], %outStrideY
LDPARAM.U16 [inStrideY], %inStrideY
MOVE.U16 1, %one
MOVE.U16 2, %S2
MUL.U16.LOW %x, %y, %S3
MUL.U16.WIDE %S3, %S2, %blockStride

CMP.S16.EQ %inStrideY, %x, %S3
JMPC %S3, $INPUT_HW32
MUL.U16.WIDE %x, %S2, %inBXStride
JMP $INPUT_READY
$INPUT_HW32:
MOVE.U32 %blockStride, %inBXStride
$INPUT_READY:

CMP.S16.EQ %outStrideY, %x, %S3
JMPC %S3, $OUTPUT_HW32
MUL.U16.WIDE %x, %S2, %outBXStride
JMP $OUTPUT_READY
$OUTPUT_HW32:
MOVE.U32 %blockStride, %outBXStride
$OUTPUT_READY:

LDPARAM.U32 [input], %input
LDPARAM.U32 [out], %out

$BLOCK_ENTRY:
LOADALN %input, %inStrideY, %inStrideY, IA
##
STOREALN %out, %outStrideY, %outStrideY, IA
##
SUB.U16 %gridX, %one, %gridX
ADD.S32 %input, %inBXStride, %input
ADD.U32 %out, %outBXStride, %out
JMPC %gridX, $BLOCK_ENTRY
##
