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
.var .u32 %inBYStride
.var .u32 %outBYStride
.var .u16 %inStrideY
.var .u16 %outStrideY
.var .u16 %one
.var .u16 %M
.var .u16 %y
.var .u16 %x
.var .u16 %gridX
.var .u16 %gridY
.var .u16 %bidX 
.var .u16 %bidY
.var .u16 %S2
.var .u16 %S3


//----------------------------------------------------------------------------------------------------
// for HW to NHW32
// input    [1]      | [H]          | [W]
//                   | [grid.y][y]  | [grid.x][x]
// output   [N]      | [H]          | [W32]      
// 			[grid.x] | [grid.y][y] 	| [x]
// H is dynamic, configure through FRAM register [cfg_fft_w4_6]
// x = 32
// y = 128
// Hrnd = (H + 127) / 128 * 128
// %gridY = Hrnd / y
//----------------------------------------------------------------------------------------------------
LDPARAM.U16 [tbDim.x], %x
MOVE.U16 128, %y
MUL.U16.LOW %x, %y, %S2
CFGXYZ %x, %y, %S2

MOVE.U16 1, %one
SUB.U16 %y, %one, %S2 
LDPARAM.U16 [cfg_fft_w4_6], %M
ADD.U16 %M, %S2, %S2
MOVE.U16 7, %S3 
SHR.U16 %S2, %S3, %gridY
SHL.U16 %gridY, 7, %M

LDPARAM.U16 [gridX], %gridX
LDPARAM.U16 [outStrideY], %outStrideY
LDPARAM.U16 [inStrideY], %inStrideY

MOVE.U16 2, %S2
//----------------------------------------------------------------------------------------------------
// first check which branch
//
// HW to NHW32:
// inBXStride =  block_x * sizeof(short);
// outBXStride = M * block_x * sizeof(short);
// inBYStride = block_y * column * sizeof(short);
// outBYStride = block_y * block_x * sizeof(short);
// inStrideY = column;
//
// NHW32 to HW:
// inBXStride = M * block_x * sizeof(short);
// outBXStride = block_x * sizeof(short);
// inBYStride = block_y * block_x * sizeof(short);
// outBYStride = block_y * column * sizeof(short);
// outStrideY = column;
//----------------------------------------------------------------------------------------------------
CMP.S16.EQ %inStrideY, %x, %S3
JMPC %S3, $NHW32_TO_HW
MUL.U16.WIDE %x, %S2, %inBXStride
MUL.U16.LOW %x, %S2, %S3
MUL.U16.WIDE %S3, %M, %outBXStride
MUL.U16.LOW %y, %S2, %S3
MUL.U16.WIDE %S3, %inStrideY, %inBYStride
MUL.U16.LOW %x, %y, %S3
MUL.U16.WIDE %S3, %S2, %outBYStride
JMP $READY

$NHW32_TO_HW:
MUL.U16.LOW %x, %S2, %S3
MUL.U16.WIDE %S3, %M, %inBXStride
MUL.U16.WIDE %x, %S2, %outBXStride

MUL.U16.LOW %x, %y, %S3
MUL.U16.WIDE %S3, %S2, %inBYStride
MUL.U16.LOW %y, %S2, %S3
MUL.U16.WIDE %S3, %outStrideY, %outBYStride


$READY:
MOVE.U16 0, %bidX
MOVE.U16 0, %bidY
LDPARAM.U32 [input], %input
LDPARAM.U32 [out], %out

$BLOCK_ENTRY:
LOADALN %input, %inStrideY, %inStrideY, IA
##
STOREALN %out, %outStrideY, %outStrideY, IA
##
//----------------------------------------------------------------------------------------------------
// input  +=  bidX * inBXStride + bidY * inBYStride
// output +=  bidX * outBXStride + bidY * outBYStride
//----------------------------------------------------------------------------------------------------
SUB.U16 %gridX, %one, %gridX
ADD.U16 %bidX, %one, %bidX
LDPARAM.U32 [input], %input

CVT.U16.U32 %bidX, DS13
MUL.U32.LOW DS13, %inBXStride, DS13
ADD.S32 %input, DS13, %input

CVT.U16.U32 %bidY, DS13
MUL.U32.LOW DS13, %inBYStride, DS13
ADD.S32 %input, DS13, %input


LDPARAM.U32 [out], %out
CVT.U16.U32 %bidX, DS13
MUL.U32.LOW DS13, %outBXStride, DS13
ADD.S32 %out, DS13, %out

CVT.U16.U32 %bidY, DS13
MUL.U32.LOW DS13, %outBYStride, DS13
ADD.S32 %out, DS13, %out

JMPC %gridX, $BLOCK_ENTRY


LDPARAM.U16 [gridX], %gridX
MOVE.U16 0, %bidX
SUB.U16 %gridY, %one, %gridY
ADD.U16 %bidY, %one, %bidY
LDPARAM.U32 [input], %input
CVT.U16.U32 %bidY, DS13
MUL.U32.LOW DS13, %inBYStride, DS13
ADD.S32 %input, DS13, %input
LDPARAM.U32 [out], %out
CVT.U16.U32 %bidY, DS13
MUL.U32.LOW DS13, %outBYStride, DS13
ADD.S32 %out, DS13, %out
JMPC %gridY, $BLOCK_ENTRY
##
