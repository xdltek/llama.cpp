.param   .U32 in_act
.param   .U32 in_wq
.param   .U32 in_sign
.param   .U32 in_scale
.param   .U32 out_addr
.param   .U32 lut_addr
.param   .U32 inUnrollStride
.param   .U32 inLoopStride0
.param   .U32 inLoopStride1
.param   .U32 scaleLoopStride
.param   .U32 blockXSize
.param   .U32 in_wq_blockz_size
.param   .U32 in_sign_blockz_size
.param   .U32 in_scale_blockz_size
.param   .U32 in_a_blockz_size
.param   .U32 input_acc_addr
.param   .U32 input_acc_addr_hi
.param   .U32 hilo_offset
.param   .U16 loop
.param   .U16 combine


.var .u32  %in_wq
.var .u32  %in_sign
.var .u32  %in_scale
.var .u32  %lut_addr
.var .u32  %out_addr
.var .u32  %inUnrollStride
.var .u32  %blockXSize
.var .u32  %Tmp0
.var .u32  %Tmp1
.var .u32  %bidz
.var .u16  %one
.var .u16  %unroll0
.var .u16  %unroll1
.var .u16  %loop
.var .u16  %combine


LDPARAM.U16 [tbDim.x], %unroll0
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %loop
CFGXYZ %unroll0, %one, %loop
MOVE.U16 1, %one
MOVE.U16 4, %unroll0
MOVE.U16 8, %unroll1
LDPARAM.U32 [in_wq], %in_wq
LDPARAM.U32 [in_sign], %in_sign
LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [lut_addr], %lut_addr
LDPARAM.U32 [inUnrollStride], %inUnrollStride
LDPARAM.U32 [blockXSize], %blockXSize
LDPARAM.U16 [loop], %loop
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
CVT.U16.U32 S30, %bidz
LDPARAM.U32 [in_scale_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_scale, %Tmp0, %in_scale
CVT.U16.U32 S28, DS13
MUL.U32 DS13, %blockXSize, %Tmp1
ADD.U32 %in_scale, %Tmp1, %in_scale

LDPARAM.U32 [in_wq_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_wq, %Tmp0, %in_wq
ADD.U32 %in_wq, %Tmp1, %in_wq


LDPARAM.U32 [in_sign_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_sign, %Tmp0, %in_sign
ADD.U32 %in_sign, %Tmp1, %in_sign

LDPARAM.U32 [out_addr], %out_addr
ADD.U32 %out_addr, %Tmp1, %out_addr

LDPARAM.U32 [in_a_blockz_size], %Tmp0
LDPARAM.U32 [in_act], DS13
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 DS13, %Tmp0, DS13
MMOV DS13, DVA
##
//----------------------------------------------------------------------------------------------------
// scale      [K/256]     |  [4]     |  [4]          |  [N]
// scale      [grid.z]    |  [loop]  |  [unroll0]    |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
// qsign      [K/256]     |  [4]     |  [4]          |  [16][N]
// qsign      [grid.z]    |  [loop]  |  [unroll0]    |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
// codebook   [K/256]      |  [4]     |  [4]       [2] |  [8][N]
// codebook   [grid.z]     |  [loop]  |  [unroll0]     |  [grid.x]*[x]
//----------------------------------------------------------------------------------------------------


$LOOP_0:
$UNROLL_0:
LOADCONT %in_sign, IB
##
LOADCONT %in_wq, MB
MOVE.U16 MB, VA
##
$FIRST_8:
COPYEXT VA, IG
AND.U16 VA, 0x3, VA
##
LOAD [%lut_addr + (VA << 1)], MA
COPYSIGN.F16 IB, MA, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
COPYEXT VAB, IEF
SHL.U16 IB, 1, VA
##
COPY VA, IB
SHR.U16 IG, 2, VA
##
SUB.U16 %unroll1, %one, %unroll1
JMPC %unroll1, $FIRST_8

MOVE.U16 8, %unroll1
ADD.U32 %in_wq, %inUnrollStride, %in_wq

TERM VA
LOADCONT %in_wq, MB
MOVE.U16 MB, VA
##
$SECOND_8:
COPYEXT VA, IG
AND.U16 VA, 0x3, VA
##
LOAD [%lut_addr + (VA << 1)], MA
COPYSIGN.F16 IB, MA, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
COPYEXT VAB, IEF
SHL.U16 IB, 1, VA
##
COPY VA, IB
SHR.U16 IG, 2, VA
##
SUB.U16 %unroll1, %one, %unroll1
JMPC %unroll1, $SECOND_8

//---------------------------------------------------------------------------------------
//result = scale * acc( a * wq)
//acc(a * wq) = IEF
//----------------------------------------------------------------------------------------
COPYEXT VA, IG
LOADCONT %in_scale, MB
CVT.F16.F32 MB, VAB
##
MUL.F32 VAB, IEF, VAB
##
ADD.F32 VAB, ICD, VAB
##
COPY VAB, ICD
MOVE.U32 0, VAB
##
COPYEXT VAB, IEF
##
SUB.U16 %unroll0, %one, %unroll0
MOVE.U16 8, %unroll1
ADD.U32 %in_scale, %inUnrollStride, %in_scale
ADD.U32 %in_wq, %inUnrollStride, %in_wq
ADD.U32 %in_sign, %inUnrollStride, %in_sign
JMPC %unroll0, $UNROLL_0

MOVE.U16 4, %unroll0
SUB.U16 %loop, %one, %loop
JMPC %loop, $LOOP_0
LDPARAM.U16 [combine], %combine
OR.U16 %combine, S30, %combine
LDPARAM.U32 [hilo_offset], %Tmp0
ADD.U32 %out_addr, %Tmp0, %Tmp0
JMPNC %combine, $STORE
LOADCONT %out_addr, IB
##
//low word
LOADCONT %Tmp0, MB
ADD.F32.R0MB IB, ICD, VAB
##
COPY VAB, ICD
MOVE.U32 0, VAB
##
COPYEXT VAB, IEF
##
$STORE:
//high word
STORECONT %out_addr, ID
##
//low word
STORECONT %Tmp0, IC
##16
DMB_ORI
##
