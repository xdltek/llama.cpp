//ENTRY:
.param .u32 input
.param .u32 output
.param .u32 colorStride
.param .u32 batchNb
.param .u32 batchStride
.param .u16 channel
.param .u16 width
.param .u16 height


.var .u32 %input0
.var .u32 %output
.var .u16 %width
.var .u16 %height
.var .u32 %colorStride
.var .u32 %tmp0
.var .u32 %tmp1

.var .u16 %sreg0
.var .u16 %one
.var .u16 %tbDim.x
.var .u16 %tbDim.y
.var .u16 %channel
.var .u16 %two

.var .u32 %batchNb
.var .u32 %doneBatchNb
.var .u32 %batchStride

LDPARAM.U16 [tid_xyz], %sreg0 
LDPARAM.U16 [tbDim.x], %tbDim.x
LDPARAM.U16 [tbDim.y], %tbDim.y
CFGXYZ %tbDim.x, %tbDim.y, %sreg0
MOVE.U32 0, VAB
MMOV 0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
LDPARAM.U16 [width], %width
LDPARAM.U16 [height], %height
LDPARAM.U16 [channel], %channel
LDPARAM.U32 [colorStride], %colorStride
LDPARAM.U32 [batchNb], %batchNb
MOVE.U32 0, %doneBatchNb

MOVE.U16 1, %one
MOVE.U16 2, %two
LDPARAM.U32 [input], %input0
LDPARAM.U32 [output], %output

$BLOCK_ENTRY:
LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, IC
##
MUL.U16.LOW %blockIdx.x, %tbDim.x, %sreg0
DECTID.X IC, %tmp1, IA   
##
ADD.U16 IA, %sreg0, IA   //IA is dstx
##
MUL.U16.LOW %blockIdx.y, %tbDim.y, %sreg0
DECTID.Y IC, %tmp1, IB
##
ADD.U16 IB, %sreg0, IB   //IB is dsty
##
MUL.U16.WIDE  IB, %width, ICD  //inY*inWidth
##
CVT.U16.U32 IA, IAB
##
ADD.S32	IAB, ICD, ICD //inY*inWidth + inX
##
CVT.U16.U32 %channel, %tmp0
MUL.U32.LOW ICD, %tmp0, IEF // inY*inWidth + inX
##
SHL.U32	IEF, 1, IEF
##
SHL.U32 ICD, 1, ICD
##
$channelRpt:
MADD.SCA32 ICD, %input0, DVA
##
LOAD [DVA], IA
##
MADD.SCA32 IEF, %output, DVA
##
STORE IA, [DVA]
##
ADD.U32 %input0, %colorStride, %input0
CVT.U16.U32 %two, %tmp0
ADD.U32 %tmp0, %output, %output
SUB.U16 %channel, %one, %channel
JMPC %channel, $channelRpt
##
LDPARAM.U32 [input], %input0
LDPARAM.U32 [output], %output
LDPARAM.U32 [batchStride], %batchStride
CVT.U16.U32 %one, %tmp0
ADD.U32 %tmp0, %doneBatchNb, %doneBatchNb
MUL.U32 %doneBatchNb, %batchStride, %batchStride
ADD.U32 %input0, %batchStride, %input0
ADD.U32 %output, %batchStride, %output
SUB.U32 %batchNb, %tmp0, %batchNb
LDPARAM.U16 [channel], %channel
JMPC %batchNb, $BLOCK_ENTRY
##