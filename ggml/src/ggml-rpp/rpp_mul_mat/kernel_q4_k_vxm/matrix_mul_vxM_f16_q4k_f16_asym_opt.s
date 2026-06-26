.param   .U32 in_act
.param   .U32 in_lsb
.param   .U32 in_scale
.param   .U32 in_zero
.param   .U32 out_addr
.param   .U32 lut_addr
.param   .U32 inUnrollStride
.param   .U32 inLoopStride0
.param   .U32 scaleLoopStride
.param   .U32 blockXSize
.param   .U32 in_lsb_blockz_size
.param   .U32 in_scale_blockz_size
.param   .U32 in_a_blockz_size
.param   .U32 input_acc_addr
.param   .U32 input_acc_addr_hi
.param   .U32 hilo_offset
.param   .U16 loop
.param   .U16 combine


.var .u32  %in_lsb
.var .u32  %in_scale
.var .u32  %in_zero
.var .u32  %out_addr
.var .u32  %inUnrollStride
.var .u32  %blockXSize
.var .u32  %Tmp0
.var .u32  %Tmp1
.var .u32  %bidx
.var .u32  %bidz
.var .u16  %one
.var .u16  %unroll
.var .u16  %loop
.var .u16  %combine
.var .u16  %sreg0


LDPARAM.U16 [tbDim.x], %unroll
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %loop
CFGXYZ %unroll, %one, %loop
MOVE.U16 1, %one
MOVE.U16 8, %unroll
LDPARAM.U32 [in_lsb], %in_lsb
LDPARAM.U32 [in_scale], %in_scale
LDPARAM.U32 [in_zero], %in_zero
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
CVT.U16.U32 S28, %bidx
CVT.U16.U32 S30, %bidz
LDPARAM.U32 [in_scale_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_scale, %Tmp0, %in_scale
ADD.U32 %in_zero, %Tmp0, %in_zero
MUL.U32 %bidx, %blockXSize, %Tmp1
ADD.U32 %in_scale, %Tmp1, %in_scale
ADD.U32 %in_zero, %Tmp1, %in_zero


LDPARAM.U32 [in_lsb_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_lsb, %Tmp0, %in_lsb
ADD.U32 %in_lsb, %Tmp1, %in_lsb

LDPARAM.U32 [out_addr], %out_addr
ADD.U32 %out_addr, %Tmp1, %out_addr

LDPARAM.U32 [in_a_blockz_size], %Tmp0
LDPARAM.U32 [in_act], DS13
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 DS13, %Tmp0, DS13
MMOV DS13, DVA
MOVE.U32 0, VAB
##
//-----------------------------------------------------------------------------------------------------
// in_scale   [K/256]           | [8]       | [N]
// in_scale   [grid.z]          | [loop]    | [x]
//-----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// in_lsb  [K/256]      | [8]      | [32/4]         | [4][N]
// in_lsb  [1]          | [loop]   | [unroll]       | [x]
//         [grid.z]     | [loop]                    | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
$BLOCK_0:
COPYEXT VAB, IEF
##
LOADCONT %in_lsb, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
CVT.U8.F16 VA, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
COPYEXT VAB, IEF
CVT.U8.F16 IB, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
COPYEXT VAB, IEF
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
CVT.U8.F16 VA, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
COPYEXT VAB, IEF
CVT.U8.F16 IB, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
SUB.U16 %unroll, %one, %unroll
ADD.U32 %in_lsb, %inUnrollStride, %in_lsb
JMPC %unroll, $BLOCK_0
//---------------------------------------------------------------------------------------
//w = scale * wq - zero
//acc += a * w
//acc = acc(a * wq) * scale - acc(a) * zero 
//acc(a * wq) = IEF
//acc(a) = IAB
//----------------------------------------------------------------------------------------
COPYEXT VAB, IEF
LOADCONT %in_scale, MB
CVT.F16.F32 MB, VAB
##
MUL.F32 VAB, IEF, VAB
##
//---------------------------------------------------------------------------------------
//InputAcc [K/256]      | [8]
//         [grid.z]     | [loopidx]
//accIdx = blockIdx.z * 8 + loopidx
//----------------------------------------------------------------------------------------
COPYEXT VAB, IEF
LDPARAM.U32 [input_acc_addr], %Tmp0
LDPARAM.U32 [input_acc_addr_hi], %Tmp1
MOVE.U16 8, S26
MUL.U16.LOW S30, S26, %sreg0
SUB.U16 S26, %loop, S27
ADD.U16 %sreg0, S27, %sreg0
MOVE.U16 2, S26
MUL.U16.WIDE %sreg0, S26, DS13
ADD.U32 %Tmp0, DS13, %Tmp0
ADD.U32 %Tmp1, DS13, %Tmp1
LOAD [%Tmp0 + (IH << 1)], IA
##
LOADCONT %in_zero, MB
LOAD [%Tmp1 + (IH << 1)], IB
CVT.F16.F32 MB, VAB
##
MUL.F32 IAB, VAB, VAB
##
ADD.F32 IEF, VAB, VAB
##
ADD.F32 VAB, ICD, VAB
##
COPY VAB, ICD
MOVE.U32 0, VAB
##
MOVE.U16 8, %unroll
SUB.U16 %loop, %one, %loop
ADD.U32 %in_scale, %inUnrollStride, %in_scale
ADD.U32 %in_zero, %inUnrollStride, %in_zero
JMPC %loop, $BLOCK_0

LDPARAM.U16 [combine], %combine
OR.U16 %combine, S30, %combine
LDPARAM.U32 [hilo_offset], %Tmp0
ADD.U32 %out_addr, %Tmp0, %Tmp0
JMPNC %combine, $STORE
LOADCONT %out_addr, IB
COPYEXT VAB, IEF
##
//low word
LOADCONT %Tmp0, MB
ADD.F32.R0MB IB, ICD, VAB
##
COPY VAB, ICD
MOVE.U32 0, VAB
##
$STORE:
//high word
COPYEXT VAB, IEF
STORECONT %out_addr, ID
##
//low word
STORECONT %Tmp0, IC
##16
DMB_ORI
##