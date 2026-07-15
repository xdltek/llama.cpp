//ENTRY:
.param .u32 input     //input
.param .u32 filter    //filter
.param .u32 bias      //bias
.param .u32 out       //out
.param .u16 Cn
.param .u16 isBias
.param .u16 alpha
.param .u16 isRelu
.param .u16 isPrelu
.param .u16 isLeaky
.param .u32 inStrideY
.param .u32 outStrideY
.param .u32 inStrideZ
.param .u32 outStrideZ
.param .u32 inSwitchSize
.param .u32 outBlockSize
.param .u32 inCombineOffset
.param .u32 comScale
.param .u32 bn_scale
.param .u32 prelu_scale
.param .u16 rpt_row
.param .u16 rpt_col_m1
.param .u16 tailLen
.param .u16 Un
.param .u32 inUnStride
.param .u32 outUnStride
.param .u32 rollBackOffset
.param .u16 tail_xyz
.param .u16 tail_rpt_col_m1
.param .u32 tailSwitchSize
.param .u32 tailBlockStride
.param .u32 tailInUnStride
.param .u32 tailFilterStride
.param .u32 tailStrideY
.param .u32 tailStrideZ
.param .u32 inGridYStride
.param .u32 tailInGridYStride
.param .u16 outGridYStride
.param .u16 biasOffset
.param .u32 filterOffset
.param .u16 act_others  // 128:swish

.var .u32 %input
.var .u32 %offset0
.var .u32 %blockSize
.var .u32 %filter
.var .u32 %out
.var .u32 %unStride
.var .u32 %StrideY
.var .u32 %StrideZ
.var .u32 %tailFilterStride
.var .u16 %Cn
.var .u16 %one
.var .u16 %vecLen
.var .u16 %type
.var .u16 %sreg0
.var .u16 %sreg1
.var .u16 %zero
.var .u16 %tailLen

JMPC %blockIdx, $BLOCK_ENTRY
LDPARAM.U16 [tid_xyz], %sreg0
LDPARAM.U16 [tbDim.x], %type
LDPARAM.U16 [tbDim.y], %sreg1
CFGXYZ %type, %sreg1, %sreg0
LDPARAM.U32 [tailFilterStride], %tailFilterStride
MOVE.U16 0, %zero
MOVE.U16 1, %one
MOVE.U16 8, %vecLen

$BLOCK_ENTRY:
MOVE.U16 %blockIdx.z, %sreg0
LDPARAM.U16 [grid_dim_z], %sreg1
ADD.U16 %sreg0, %one, %type
SUB.U16 %sreg1, %type, %type
JMPC %type, $BLOCK_ENTRY_1
//config the block dimension for the last block
LDPARAM.U16 [xyzcfg_shadow], %sreg0
LDPARAM.U16 [dimx_shadow], %type
LDPARAM.U16 [dimy_shadow], %sreg1
CFGXYZ %type, %sreg1, %sreg0
$BLOCK_ENTRY_1:

LDPARAM.U16 [tailLen], %tailLen
LDPARAM.U16 [Cn], %Cn

LDPARAM.U32 [input], %input
LDPARAM.U32 [inUnStride], %unStride
LDPARAM.U32 [inGridYStride], %blockSize
MUL.U16.WIDE %blockIdx.y, %one, %offset0
MUL.U32 %offset0, %blockSize, %filter
MUL.U16.WIDE %blockIdx.z, %one, %offset0
MUL.U32.LOW %offset0, %unStride, %offset0
ADD.S32 %input, %filter, %input
ADD.S32 %input, %offset0, %input


LDPARAM.U16 [rpt_row], %sreg0
LDPARAM.U32 [filter], %filter
LDPARAM.U32 [filterOffset], %blockSize
MUL.U16.WIDE %blockIdx.x,%one, %offset0
MUL.U32.LOW %offset0, %blockSize, %offset0
ADD.S32 %filter, %offset0, %filter

LDPARAM.U32 [inStrideY], %StrideY
LDPARAM.U32 [inStrideZ], %StrideZ
MOVE.U32 512, %offset0
LDPARAM.U16 [rpt_col_m1], %sreg1
SUB.S32 %filter, %offset0, %filter

////////////////////////////////
///////tn0 start////////////
////////////////////////////////
MOVE.U32 0, VAB
MMOV 0, DVA
##
LDPARAM.U32 [inSwitchSize], %blockSize
GLANEADDR.Z32 %input, %StrideY, %StrideZ, DVA
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
SUB.U32 VAB, 0, VAB
##
$BLK_LOOP:
JMPNC %Cn, $LAST_CN_BLK
REPEAT 2, %sreg0
REPEAT 1, %sreg1
ADD.S32 %filter, %offset0, %filter   //acc0
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, VAB, VAB
##
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD.SCA32 DVA, %blockSize, DVA
ADD.F32 MAB, VAB, VAB
##
SUB.U16 %Cn, %one, %Cn
LDPARAM.U32 [inCombineOffset], %blockSize
ADD.S32 %input, %blockSize, %input
LDPARAM.U32 [inSwitchSize], %blockSize
GLANEADDR.Z32 %input, %StrideY, %StrideZ, DVA
ADD.F32 VAB, 0, VAB
##
JMP $BLK_LOOP
//////////////////////////////////////
///tail block start//////////////////
//////////////////////////////////////
$LAST_CN_BLK:
LDPARAM.U32 [tailBlockStride], %blockSize
LDPARAM.U32 [input], %input
ADD.S32 %input, %blockSize, %input
LDPARAM.U32 [tailInGridYStride], %offset0
MUL.U16.WIDE %blockIdx.y, %one, %blockSize
MUL.U32 %blockSize, %offset0, %offset0
ADD.S32 %input, %offset0, %input
LDPARAM.U32 [tailInUnStride], %unStride
MUL.U16.WIDE %blockIdx.z, %one, %offset0
MUL.U32.LOW %offset0, %unStride, %offset0
ADD.S32 %input, %offset0, %input
MOVE.U32 512, %offset0
LDPARAM.U32 [tailStrideY], %StrideY
LDPARAM.U32 [tailStrideZ], %StrideZ
LDPARAM.U32 [tailSwitchSize], %blockSize
GLANEADDR.Z32 %input, %StrideY, %StrideZ, DVA
ADD.U32 VAB, 0, VAB
##
////////////////////////////////
///////tail block loop//////////
////////////////////////////////
$COL_ACC:
LDPARAM.U16 [tail_rpt_col_m1], %sreg1
$LINE_ACC:
JMPNC %sreg1, $LINE_SWITCH
SUB.S16 %sreg1, %one, %sreg1
ADD.S32 %filter, %offset0, %filter   //acc0
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %vecLen, MAB
MADD DVA, 16, DVA
ADD.F32 MAB, VAB, VAB
##
JMP $LINE_ACC
$LINE_SWITCH:
ADD.S32 %filter, %offset0, %filter
LDWARP %filter, %zero, %zero, WARPVEC
LDLANE_BFMAD [DVA], WARPVEC, %tailLen, MAB
MADD.SCA32 DVA, %blockSize, DVA
ADD.F32 MAB, VAB, VAB
##
SUB.S32 %filter, %tailFilterStride, %filter
SUB.S16 %sreg0, %one, %sreg0
JMPC %sreg0, $COL_ACC
////////////////////////////////
///////Store tn0 start//////////
////////////////////////////////
COPYEXT VAB, IEF
CVT.F32.F16 VAB, VA
##
LDPARAM.U16 [isBias], %type
JMPNC %type, $RELU_LAB0
LDPARAM.U32 [bias], %input
LDPARAM.U16 [biasOffset], %sreg0
MUL.U16.WIDE %blockIdx.x,%sreg0, %offset0
ADD.S32 %input, %offset0, %input
LOADALN %input, %zero, %zero, MB
ADD.F16 VA, MB, VA
##
$RELU_LAB0:
LDPARAM.U16 [isRelu], %type
JMPNC %type, $PRELU_LAB0
MAX.F16 VA, 0, VA
##
$PRELU_LAB0:
LDPARAM.U16 [isPrelu], %type
JMPNC %type, $LEAKY_LAB0
COPY VA, IB
MAX.F16 VA, 0, VA
##
COPY VA, IA
MIN.F16 IB, 0, VA
##
LDPARAM.U32 [prelu_scale], %input
LOADALN %input, %zero, %zero, MB
MUL.F16 VA, MB, VA
##
ADD.F16 VA, IA, VA
##

// if isLeaky:
//      do leakyrelu
// else:
//      goto act_others_lab
$LEAKY_LAB0:
LDPARAM.U16 [isLeaky], %type
JMPNC %type, $HSWISH_LAB0
LDPARAM.U16 [alpha], %sreg1
COPY VA, IA
MUL.F16 VA, %sreg1, VA
##
MAX.F16 VA, IA, VA
##

$HSWISH_LAB0:
LDPARAM.U16 [act_others], %type
MOVE.U16 32, %sreg0
CMP.U16.EQ %type, %sreg0, %sreg0
JMPNC %sreg0, $SIGMOID_C3_LAB
COPY VA, IA
ADD.F16 VA, 0x4040, VA
##
MAX.F16 VA, 0, VA
##
MIN.F16 VA, 0x40c0, VA
##
MUL.F16 VA, 0x3e2b, VA
##
MUL.F16 VA, IA, VA
##

$SIGMOID_C3_LAB:
LDPARAM.U16 [act_others], %type
MOVE.U16 128, %sreg0
CMP.U16.EQ %type, %sreg0, %sreg0
JMPNC %sreg0, $STORE_LAB0

COPYEXT VA, IF
##
ABS.F32 IEF, VAB
##
COPY VAB, IAB
MOVE.U32 0x3E968EAC, %out
MUL.F32 VAB, %out, VAB
##
MOVE.U32 0xBBC3F8A8, %out
ADD.F32 VAB, %out, VAB
##
COPYEXT VAB, IGH
MUL.F32 IAB, IAB, VAB
##
COPY VAB, ICD
MOVE.U32 0xBD721E0A, %out
MUL.F32 VAB, %out, VAB
##
ADD.F32 IGH, VAB, VAB
##
COPYEXT VAB, IGH
MUL.F32 ICD, IAB, VAB
##
MOVE.U32 0x3B847D7E, %out
MUL.F32 VAB, %out, VAB
##
ADD.F32 IGH, VAB, VAB
##
MOVE.U32 0x3F000000, %out
MIN.S32 VAB, %out, VAB
##
COPYSIGN.F32 IEF, VAB, VAB
##
MOVE.U32 0x3F000000, %out
ADD.F32 VAB, %out, VAB
##
MUL.F32 VAB, IEF, VAB
##
CVT.F32.F16 VAB, VA
##

$STORE_LAB0:
LDPARAM.U32 [out], %out
LDPARAM.U32 [outStrideY], %StrideY
LDPARAM.U32 [outStrideZ], %StrideZ
LDPARAM.U32 [outUnStride], %unStride
LDPARAM.U16 [outGridYStride], %sreg0
MUL.U16.WIDE %blockIdx.y, %sreg0, %filter
MUL.U16.WIDE %blockIdx.z, %one, %offset0
MUL.U32.LOW %offset0, %unStride, %offset0
ADD.S32 %out, %offset0, %out
ADD.S32 %out, %filter, %out
//block x is the channel group
LDPARAM.U32 [outBlockSize], %blockSize
MUL.U16.WIDE %blockIdx.x,%one, %offset0
MUL.U32.LOW %offset0, %blockSize, %offset0
ADD.S32 %out, %offset0, %out
STOREALN.Z32 %out, %StrideY, %StrideZ, VA
##
