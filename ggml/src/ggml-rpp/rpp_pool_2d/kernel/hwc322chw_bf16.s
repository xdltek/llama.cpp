//ENTRY:
.param .u32 input
.param .u32 output
.param .u32 blockSize
.param .u32 batchNb
.param .u32 inBlockSize
.param .u32 outBlockSize


.var .u32 %input0
.var .u32 %output
.var .u32 %blockSize
.var .u32 %bitShift
.var .u32 %res
.var .u32 %one
.var .u32 %tmp0
.var .u32 %tmp1
.var .u16 %sreg0

.var .u16 %tbDim.x
.var .u16 %tbDim.y

.var .u32 %batchNb
.var .u32 %inBlockSize
.var .u32 %outBlockSize

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
MOVE.U32 1, %one
MOVE.U32 5, %bitShift
MOVE.U32 31, %res
LDPARAM.U32 [input], %input0
LDPARAM.U32 [output], %output
LDPARAM.U32 [blockSize], %blockSize
LDPARAM.U32 [batchNb], %batchNb
LDPARAM.U32 [inBlockSize], %inBlockSize
LDPARAM.U32 [outBlockSize], %outBlockSize
$BLOCK_ENTRY:
LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, IC
##
DECTID.X IC, %tmp1, IA   
##
MUL.U16.WIDE %blockIdx.y, %tbDim.y, %tmp0
DECTID.Y IC, %tmp1, IB
##
CVT.U16.U32 IB, ICD
##
CVT.U16.U32 IA, IAB		//idx
##
ADD.U32 ICD, %tmp0, ICD	//idy
##
SHR.U32 IAB, %bitShift, IEF // dstId = idx / 32
##
MUL.U32 IEF, %blockSize, IEF // dstId * hwsize
##
ADD.U32 IEF, ICD, IEF		//IEF = hwsize * dstId + idy
##
SHL.U32 IEF, %bitShift, IEF	//IEF = IEF * 32
##
AND.U32 IAB, %res, IGH		//resIdx
##
OR.U32	IEF, IGH, IEF		//srcIdx = (hwsize * dstId + idy) * 32 + resIdx
##
SHL.U32 IEF, %one, IEF
##
MUL.U32 IAB, %blockSize, IAB
##
ADD.U32 IAB, ICD, IAB		//dstIdx = idx * hwsize + idy
##
SHL.U32 IAB, %one, IAB
##
ADD.U32 IEF, %input0, IEF
##
ADD.U32 IAB, %output, IAB
##
MMOV	IEF, DVA
##
LOAD	[DVA], IE			//src[srcIdx]
##
CMP.U32.LT	ICD, %blockSize, IC
##
MMOV	IAB, DVA
##
STOREMASK	IE, IC, [DVA]
##
ADD.U32 %input0, %inBlockSize, %input0
ADD.U32 %output, %outBlockSize, %output
SUB.U32 %batchNb, %one, %batchNb
JMPC %batchNb, $BLOCK_ENTRY
##