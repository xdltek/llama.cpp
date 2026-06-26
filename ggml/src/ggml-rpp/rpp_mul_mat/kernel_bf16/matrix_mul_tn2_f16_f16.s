//ENTRY:
.param .u32 input       //input matrix A: CxMx32 (Cx32=K)
.param .u32 filter		//Input matrix B: nxKx32 (nx32=N)
.param .u32 out         //output matrix C: nxMx32
.param .u16 Cn
.param .u16 Un
.param .u32 inStrideY
.param .u32 outStrideY
.param .u32 inStrideZ
.param .u32 outStrideZ
.param .u32 inSwitchSize
.param .u32 gridXInBStride
.param .u32 gridXOutStride
.param .u32 gridYinAStride
.param .u32 gridYoutStride
.param .u32 gridZInAStride
.param .u32 gridZInBStride
.param .u32 gridZOutStride
.param .u32 outTnStride
.param .u32 filterOffset0
.param .u32 filterOffset1


.var .u32 %input
.var .u32 %offset0
.var .u32 %offset1
.var .u32 %filter
.var .u32 %out
.var .u32 %blockSize
.var .u32 %StrideY
.var .u32 %StrideZ
.var .u16 %Cn
.var .u16 %one
.var .u16 %vecLen
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %S3
.var .u16 %zero


//JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tid_xyz], %S0
LDPARAM.U16 [tbDim.x], %S2
LDPARAM.U16 [tbDim.y], %S1
CFGXYZ %S2, %S1, %S0

MOVE.U16 0, %zero
MOVE.U16 1, %one
MOVE.U16 8, %vecLen
//JMP $BLOCK_ENTRY_1
$BLOCK_ENTRY:
MOVE.U16 %blockIdx.y, %S0
LDPARAM.U16 [grid_dim_y], %S1
ADD.U16 %S0, %one, %S3
SUB.U16 %S1, %S3, %S3
JMPC %S3, $BLOCK_ENTRY_1
//config the block dimension for the last block in Y
LDPARAM.U16 [xyzcfg_shadow], %S0
LDPARAM.U16 [dimx_shadow], %S2
LDPARAM.U16 [dimy_shadow], %S1
CFGXYZ %S2, %S1, %S0

$BLOCK_ENTRY_1:
LDPARAM.U16 [Cn], %Cn
LDPARAM.U32 [inSwitchSize], %blockSize
LDPARAM.U32 [input], %input
//input offset set
//grid x, column grid, in 0 no grid x 
//grid y row grid
LDPARAM.U32 [gridYinAStride], %offset0
MUL.U16.WIDE %blockIdx.y, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %input, %offset1, %input
//grid z  outer dimension L     LxMxK  x LxKxN ->  LxMxN
LDPARAM.U32 [gridZInAStride], %offset0
MUL.U16.WIDE %blockIdx.z, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %input, %offset1, %input

LDPARAM.U32 [filter], %filter
LDPARAM.U32 [gridXInBStride], %offset0
MUL.U16.WIDE %blockIdx.x, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %filter, %offset1, %filter

LDPARAM.U32 [gridZInBStride], %offset0
MUL.U16.WIDE %blockIdx.z, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %filter, %offset1, %filter

LDPARAM.U32 [filterOffset0], %offset0
LDPARAM.U32 [filterOffset1], %offset1
MOVE.U32 0, VAB
MMOV 0, DVA
##
////////////////////////////////
///////INIT start////////////
////////////////////////////////
ADD.S32 %filter, %offset1, %filter
LDPARAM.U32 [inStrideY], %StrideY
LDPARAM.U32 [inStrideZ], %StrideZ
GLANEADDR.Z32 %input, %StrideY, %StrideZ, DVA
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
//////////////////////////////
//acc k0 ~ k7
//////////////////////////////
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, 0, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, 0, VAB
##
//////////////////////////////
//acc k8 ~ k15
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, ICD, VAB
##
//////////////////////////////
//acc k16~k23
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, ICD, VAB
##
//////////////////////////////
//acc k24~k31
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD.SCA32 DVA, %blockSize, DVA
ADD.F32 MAB, ICD, VAB
##
////////////////////////////////
///////COMBINE start////////////
////////////////////////////////
$COMBINE_ACC:
JMPNC %Cn, $END_ACC
//////////////////////////////
//combine k0 ~ k7
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, ICD, VAB
##
//////////////////////////////
//combine k8 ~ k15
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, ICD, VAB
##
//////////////////////////////
//combine k16 ~ k23
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, ICD, VAB
##
//////////////////////////////
//combine k24 ~ k31
//////////////////////////////
COPY VAB, ICD
SUB.S32 %filter, %offset1, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD.SCA32 DVA, %blockSize, DVA
ADD.F32 MAB, ICD, VAB
##
SUB.S16 %Cn, %one, %Cn
JMP $COMBINE_ACC
$END_ACC:
////////////////////////////////
///////Store tn0 start//////////
////////////////////////////////
$STORE_LAB0:
COPY VAB, ICD
LDPARAM.U32 [out], %out

LDPARAM.U32 [gridXOutStride], %offset0
MUL.U16.WIDE %blockIdx.x, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %out, %offset1, %out

LDPARAM.U32 [gridYoutStride], %offset0
MUL.U16.WIDE %blockIdx.y, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %out, %offset1, %out

LDPARAM.U32 [gridZOutStride], %offset1
MUL.U16.WIDE %blockIdx.z, %one, %offset0
MUL.U32.LOW %offset0, %offset1, %offset0
ADD.S32 %out, %offset0, %out

LDPARAM.U32 [outStrideY], %StrideY
LDPARAM.U32 [outStrideZ], %StrideZ
STOREALN.Z32 %out, %StrideY, %StrideZ, IB
##
LDPARAM.U32 [outTnStride], %offset0
ADD.U32 %out, %offset0, %out
STOREALN.Z32 %out, %StrideY, %StrideZ, ID
##
