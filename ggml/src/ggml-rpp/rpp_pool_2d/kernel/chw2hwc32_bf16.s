//ENTRY:
.param .u32 input
.param .u32 output
.param .u32 blockSize
.param .u32 batchNb
.param .u32 inBlockSize
.param .u32 outBlockSize
.param .u32 inBlockXStride
.param .u32 outBlockXStride


.var .u32 %input0
.var .u32 %output
.var .u32 %blockSize
.var .u32 %bitShift
.var .u32 %res
.var .u32 %one
.var .u32 %tmp0
.var .u32 %tmp1
.var .u32 %tmp2

.var .u32 %batchNb
.var .u32 %inBlockSize
.var .u32 %outBlockSize

.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3

MOVE.U32 1, %one
JMPC %blockIdx.x, $BLOCK_ENTRY_LAST
LDPARAM.U16 [tid_xyz], %S0 
LDPARAM.U16 [tbDim.x], %S2
LDPARAM.U16 [tbDim.y], %S3
CFGXYZ %S2, %S3, %S0
JMP $BLOCK_ENTRY_1
$BLOCK_ENTRY_LAST:
MOVE.U16 %blockIdx.x, %S0
LDPARAM.U16 [grid_dim_x], %S1
ADD.U16 %S0, %one, %S2
SUB.U16 %S1, %S2, %S2
JMPC %S2, $BLOCK_ENTRY_1
//config the block dimension for the last block
LDPARAM.U16 [xyzcfg_shadow], %S0
LDPARAM.U16 [dimx_shadow], %S2
LDPARAM.U16 [dimy_shadow], %S1
CFGXYZ %S2, %S1, %S0
$BLOCK_ENTRY_1:
MOVE.U32 0, VAB
MMOV 0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
MOVE.U32 5, %bitShift
MOVE.U32 31, %res
LDPARAM.U32 [input], %input0
LDPARAM.U32 [output], %output
LDPARAM.U32 [blockSize], %blockSize
LDPARAM.U32 [batchNb], %batchNb
LDPARAM.U32 [inBlockSize], %inBlockSize
LDPARAM.U32 [outBlockSize], %outBlockSize
$BLOCK_ENTRY:
LDPARAM.U32 [inBlockXStride], %tmp0
MUL.U16.WIDE %blockIdx.x, %one, %tmp1
MUL.U32.LOW %tmp0, %tmp1, %tmp2

LDPARAM.U32 [tid_base], %tmp0
LDPARAM.U32 [tid_depack], %tmp1
LOADCONT %tmp0, IC
##
DECTID.X IC, %tmp1, IA   
##
MUL.U16.WIDE %blockIdx.y, %S3, %tmp0
DECTID.Y IC, %tmp1, IB
##
CVT.U16.U32 IB, ICD
##
CVT.U16.U32 IA, IAB		//idx
##
ADD.U32 IAB, %tmp2, IAB //idx + block offset (block_dim_x * block_index)
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
OR.U32	IEF, IGH, IEF		//dstIdx = (hwsize * dstId + idy) * 32 + resIdx
##
SHL.U32 IEF, %one, IEF
##
MUL.U32 IAB, %blockSize, IAB
##
ADD.U32 IAB, ICD, IAB		//srcIdx = idx * hwsize + idy
##
SHL.U32 IAB, %one, IAB
##
ADD.U32 IAB, %input0, IAB
##
ADD.U32 IEF, %output, IEF
##
MMOV	IAB, DVA
##
LOAD	[DVA], IA			//src[srcIdx]
##
CMP.U32.LT	ICD, %blockSize, IC
##
MMOV	IEF, DVA
##
STOREMASK	IA, IC, [DVA]
##
ADD.U32 %input0, %inBlockSize, %input0
ADD.U32 %output, %outBlockSize, %output
SUB.U32 %batchNb, %one, %batchNb
JMPC %batchNb, $BLOCK_ENTRY
##