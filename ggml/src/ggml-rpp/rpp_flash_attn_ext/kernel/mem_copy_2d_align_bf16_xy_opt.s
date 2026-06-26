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


.var .u32 %input
.var .u32 %out
.var .u32 %input_tmp
.var .u32 %out_tmp

.var .u32 %tmp0
.var .u32 %bidY32
.var .u32 %bidX32

.var .u32 %inBYStride
.var .u32 %outBYStride
.var .u32 %inBXStride
.var .u32 %outBXStride


.var .u16 %inStrideY
.var .u16 %outStrideY

.var .u16 %one


JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tid_xyz], %inStrideY
LDPARAM.U16 [tbDim.x], %outStrideY
LDPARAM.U16 [tbDim.y], %one
CFGXYZ %outStrideY, %one, %inStrideY
MOVE.U16 1, %one
LDPARAM.U16 [outStrideY], %outStrideY
LDPARAM.U16 [inStrideY], %inStrideY
LDPARAM.U32 [inBXStride], %inBXStride
LDPARAM.U32 [outBXStride], %outBXStride

LDPARAM.U32 [input], %input
LDPARAM.U32 [out], %out
LDPARAM.U32 [inBYStride], %inBYStride
LDPARAM.U32 [outBYStride], %outBYStride

$BLOCK_ENTRY:

MUL.U16.WIDE %blockIdx.x, %one, %bidX32
MUL.U32.LOW %bidX32, %inBXStride, %tmp0
ADD.S32 %input, %tmp0, %input_tmp

MUL.U16.WIDE %blockIdx.y, %one, %bidY32
MUL.U32.LOW %bidY32, %inBYStride, %tmp0
ADD.S32 %input_tmp, %tmp0, %input_tmp

LDPARAM.U32 [inStrideZ], %tmp0
LOADALN.Z32 %input_tmp, %inStrideY, %tmp0, IA
##
MUL.U32.LOW %bidX32, %outBXStride, %tmp0
ADD.U32 %out, %tmp0, %out_tmp
MUL.U32.LOW %bidY32, %outBYStride, %tmp0
ADD.U32 %out_tmp, %tmp0, %out_tmp

LDPARAM.U32 [outStrideZ], %tmp0
STOREALN.Z32 %out_tmp, %outStrideY, %tmp0, IA
##