.param   .U32 in_act
.param   .U32 in_lsb
.param   .U32 in_msb
.param   .U32 in_scale
.param   .U32 out_addr
.param   .U32 lut_addr
.param   .U32 inUnrollStride
.param   .U32 inLoopStride0
.param   .U32 inLoopStride1
.param   .U32 scaleLoopStride
.param   .U32 blockXSize
.param   .U32 in_msb_blockz_size
.param   .U32 in_lsb_blockz_size
.param   .U32 in_scale_blockz_size
.param   .U32 in_a_blockz_size
.param   .U32 zero_addr
.param   .U32 hilo_offset
.param   .U16 loop
.param   .U16 combine

.var .u32  %in_lsb
.var .u32  %in_msb
.var .u32  %in_scale
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

LDPARAM.U16 [tbDim.x], %unroll
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %loop
CFGXYZ %unroll, %one, %loop
MOVE.U16 1, %one
MOVE.U16 2, %unroll
LDPARAM.U32 [in_lsb], %in_lsb
LDPARAM.U32 [in_msb], %in_msb
LDPARAM.U32 [in_scale], %in_scale
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
MUL.U32 %bidx, %blockXSize, %Tmp1
ADD.U32 %in_scale, %Tmp1, %in_scale

LDPARAM.U32 [in_msb_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_lsb, %Tmp0, %in_lsb
ADD.U32 %in_lsb, %Tmp1, %in_lsb


LDPARAM.U32 [in_lsb_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_msb, %Tmp0, %in_msb
ADD.U32 %in_msb, %Tmp1, %in_msb

LDPARAM.U32 [out_addr], %out_addr
ADD.U32 %out_addr, %Tmp1, %out_addr

LDPARAM.U32 [in_a_blockz_size], %Tmp0
LDPARAM.U32 [in_act], DS13
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 DS13, %Tmp0, DS13
MMOV DS13, DVA
MOVE.U32 0, VAB
##

//----------------------------------------------------------------------------------------------------
// in_lsb  [K/256][256/16][4][4][N]
// in_lsb  [K/256]      | [256/elements_per_thread] | [elements_per_thread/wqlsb_per_word] | [wqlsb_per_word][N]
// in_lsb  [1]          | [loop]                    | [unroll]                             | [x]
//         [grid.z]     | [loop]                                                           | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// in_msb  [K/256][256/16][2][8][N]
// in_msb  [K/256]      | [256/elements_per_thread] | [elements_per_thread/wqmsb_per_word] | [wqmsb_per_word][N]
// in_msb  [1]          | [loop]                    | [unroll]                             | [x]
//         [grid.z]     | [loop]                                                           | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// super group = 256
// group = 16
// in_scale   [K/256]  | [256/16]                 |  [N] 
// in_scale   [z]      | [loop]                   |  [grid.x] * [x]
//----------------------------------------------------------------------------------------------------

$BLOCK_0:
COPYEXT VAB, IEF
LOADCONT %in_lsb, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
LOADCONT %in_msb, MB
COPYEXT MB, IH
SHL.U16 MB, 4, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
// element0
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
SHL.U16 IH, 2, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements1
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
AND.U16 IH, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements2
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
SHR.U16 IH, 2, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements3
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
ADD.U32 %in_lsb, %inUnrollStride, %in_lsb
LOADCONT %in_lsb, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##
COPY VAB, IAB
SHR.U16 IH, 4, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements4
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
SHR.U16 IH, 6, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements5
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, IAB
SHR.U16 IH, 8, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IA, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements6
//-----------------------------------------------------------------------------------------------------
COPYEXT VAB, IEF
SHR.U16 IH, 10, VA
##
AND.U16 VA, 0x30, VA
##
OR.U16 VA, IB, VA
##
CVT.U8.F16 VA, VA
##
SUB.F16 VA, 0x4200, VA
##
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA MA, VA, IEF, VAB
##
//-----------------------------------------------------------------------------------------------------
//elements7
//-----------------------------------------------------------------------------------------------------
SUB.U16 %unroll, %one, %unroll
ADD.U32 %in_lsb, %inUnrollStride, %in_lsb
ADD.U32 %in_msb, %inUnrollStride, %in_msb
JMPC %unroll, $BLOCK_0
COPYEXT VAB, IEF
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
MOVE.U16 2, %unroll
SUB.U16 %loop, %one, %loop
ADD.U32 %in_scale, %inUnrollStride, %in_scale
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