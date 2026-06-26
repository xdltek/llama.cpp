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
.var .u16 %Cn
.var .u16 %one
.var .u16 %vecLen
.var .u16 %x
.var .u16 %M
.var .u16 %S2
.var .u16 %y
.var .u16 %gridY
.var .u16 %bidY
.var .u16 %zero
//----------------------------------------------------------------------------------------------------
// activation   [Kn]       | [M]    | [K32]      
// 			    [loop]     | [y]	| [loop]
// weights      [Nn]       | [K]    | [N32]      
// 			    [grid.x]   | [y]	| [x]
// output       [Nn]       | [M]    | [N32]
// 			    [grid.x]   | [y]	| [x]
// M is dynamic, configure through FRAM register [cfg_fft_w4_6]
// Not support grid.y > 1
// For MoE case, the application make sure current epxpert tokens < 256
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// below parameters realted to M, which also need to be updated
// inSwitchSize = ((M - 1) * block_x + 8) * 2
// out += M * block_x * 2 * tn_offset;
// gridx_out_stride = M * block_x * 2 * tn;
// outTnStride = (int)(in0_row * 32 * sizeof(short));
//-----------------------------------------------------------------------------------------------------

LDPARAM.U16 [tbDim.x], %x
MOVE.U16 128, %y
MUL.U16.LOW %x, %y, %S2
CFGXYZ %x, %y, %S2

MOVE.U16 1, %one
SUB.U16 %y, %one, %S2 
LDPARAM.U16 [cfg_fft_w4_6], %M
ADD.U16 %M, %S2, %S2
SHR.U16 %S2, 7, %gridY
SHL.U16 %gridY, 7, %M
MOVE.U16 0, %bidY

MOVE.U16 0, %zero
MOVE.U16 8, %vecLen

SUB.U16 %M, %one, %S2
MUL.U16.LOW %S2, %x, %S2
ADD.U16 %S2, %vecLen, %S2
SHL.U16 %S2, %one, %S2
CVT.U16.U32 %S2, %blockSize

LDPARAM.U16 [Cn], %Cn
LDPARAM.U32 [input], %input
LDPARAM.U32 [filter], %filter
LDPARAM.U32 [gridXInBStride], %offset0
MUL.U16.WIDE %blockIdx.x, %one, %offset1
MUL.U32.LOW %offset0,%offset1,%offset1
ADD.S32 %filter, %offset1, %filter

LDPARAM.U32 [filterOffset0], %offset0
LDPARAM.U32 [filterOffset1], %offset1
MOVE.U32 0, VAB
MMOV 0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
$LOOP0:
////////////////////////////////
///////INIT start////////////
////////////////////////////////
LDPARAM.U16 [Cn], %Cn
LDPARAM.U32 [input], %input
LDPARAM.U32 [gridYinAStride], %offset0
CVT.U16.U32 %bidY, DS13
MUL.U32.LOW DS13, %offset0, %offset0
ADD.S32 %input, %offset0, %input

LDPARAM.U32 [filter], %filter
LDPARAM.U32 [gridXInBStride], %offset0
MUL.U16.WIDE %blockIdx.x, %one, %offset1
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.S32 %filter, %offset1, %filter

LDPARAM.U32 [filterOffset0], %offset0
LDPARAM.U32 [filterOffset1], %offset1
LDPARAM.U32 [inStrideY], %StrideY
GLANEADDR %input, %StrideY, %StrideY, DVA
##
//////////////////////////////
//acc k0 ~ k7
//////////////////////////////
///////////////////////////////////////////////////////////
//To dispatch fast, below 4 register re-use as filter base
//filter & input & out & DS13
///////////////////////////////////////////////////////////
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, 0, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, 0, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, 0, VAB
##
COPYEXT VAB, IEF
SUB.S32 DS13, %offset1, %filter
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, 0, VAB
##
//////////////////////////////
//acc k8 ~ k15
//////////////////////////////
COPYEXT VAB, IGH
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13

LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
SUB.S32 DS13, %offset1, %filter
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, IGH, VAB
##
//////////////////////////////
//acc k16~k23
//////////////////////////////
COPYEXT VAB, IGH
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13

LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
SUB.S32 DS13, %offset1, %filter
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, IGH, VAB
##
//////////////////////////////
//acc k24~k31
//////////////////////////////
COPYEXT VAB, IGH
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
SUB.S32 DS13, %offset1, %filter
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD.SCA32 DVA, %blockSize, DVA
ADD.F32 MAB, IGH, VAB
##
////////////////////////////////
///////COMBINE start////////////
////////////////////////////////
$COMBINE_ACC:
JMPNC %Cn, $END_ACC
//////////////////////////////
//combine k0 ~ k7
//////////////////////////////
COPYEXT VAB, IGH
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13

LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
//ADD.S32 %filter, %offset0, %filter
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, IGH, VAB
##
//////////////////////////////
//combine k8 ~ k15
//////////////////////////////
COPYEXT VAB, IGH
SUB.S32 DS13, %offset1, %filter
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, IGH, VAB
##
//////////////////////////////
//combine k16 ~ k23
//////////////////////////////
COPYEXT VAB, IGH
SUB.S32 DS13, %offset1, %filter
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
SUB.S32 DS13, %offset1, %filter
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, IGH, VAB
##
//////////////////////////////
//combine k24 ~ k31
//////////////////////////////
COPYEXT VAB, IGH
ADD.S32 %filter, %offset0, %input
ADD.S32 %input, %offset0, %out
ADD.S32 %out, %offset0, DS13
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IAB, VAB
##
COPY VAB, IAB
LDWARP %input, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, ICD, VAB
##
COPY VAB, ICD
LDWARP %out, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
ADD.F32 MAB, IEF, VAB
##
COPYEXT VAB, IEF
LDWARP DS13, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD.SCA32 DVA, %blockSize, DVA
ADD.F32 MAB, IGH, VAB
##
SUB.S32 DS13, %offset1, %filter
SUB.S16 %Cn, %one, %Cn
JMP $COMBINE_ACC
$END_ACC:
////////////////////////////////
///////Store tn0 start//////////
////////////////////////////////
$STORE_LAB0:
COPYEXT VAB, IGH
LDPARAM.U32 [out], %out

LDPARAM.U32 [gridXOutStride], %offset0
CVT.U16.U32 %M, DS13
MUL.U32.LOW DS13, %offset0, %offset0
MUL.U16.WIDE %blockIdx.x, %one, %offset1
MUL.U32.LOW %offset0, %offset1, %offset1
ADD.S32 %out, %offset1, %out

LDPARAM.U32 [gridYoutStride], %offset0
CVT.U16.U32 %bidY, DS13
MUL.U32.LOW DS13, %offset0, %offset0
ADD.S32 %out, %offset0, %out

LDPARAM.U32 [outStrideY], %StrideY
STOREALN %out, %StrideY, %StrideY, IB
##
LDPARAM.U32 [outTnStride], %offset0
CVT.U16.U32 %M, DS13
MUL.U32.LOW DS13, %offset0, %offset0

ADD.U32 %out, %offset0, %out
STOREALN %out, %StrideY, %StrideY, ID
##
ADD.U32 %out, %offset0, %out
STOREALN %out, %StrideY, %StrideY, IF
##
ADD.U32 %out, %offset0, %out
STOREALN %out, %StrideY, %StrideY, IH
##
SUB.U16 %gridY, %one, %gridY
ADD.U16 %bidY, %one, %bidY

JMPC %gridY, $LOOP0
##
