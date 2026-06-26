.param   .U32 in_act
.param   .U32 in_lsb
.param   .U32 out_lsb
.param   .U32 out_msb
.param   .U32 inUnrollStride
.param   .U32 out_stride_y
.param   .U32 out_stride_z
.param   .U32 in_lsb_blockz_size
.param   .U32 out_blockz_size
.param   .U32 in_a_blockz_size
.param   .U32 input_acc_addr
.param   .U32 input_acc_addr_hi
.param   .U32 hilo_offset
.param   .U32 blockXSize
.param   .U16 act_stridey
.param   .U16 in_lsb_stridey
.param   .U16 combine


.var .u32  %in_lsb
.var .u32  %out_lsb
.var .u32  %out_msb
.var .u32  %inUnrollStride
.var .u32  %Tmp0
.var .u32  %Tmp1
.var .u32  %bidz
.var .u32  %blockXSize
.var .u16  %one
.var .u16  %zero
.var .u16  %unroll
.var .u16  %act_stridey
.var .u16  %out_stridey
.var .u16  %in_lsb_stridey
.var .u16  %combine
.var .u16  %sreg0


LDPARAM.U16 [tbDim.x], %unroll
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %zero
CFGXYZ %unroll, %one, %zero
MOVE.U16 1, %one
MOVE.U16 8, %unroll
LDPARAM.U32 [in_lsb], %in_lsb
LDPARAM.U32 [inUnrollStride], %inUnrollStride
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
MOVE.U16 0, %zero
//act_stridey is elements * bytes_of_elements
LDPARAM.U16 [act_stridey], %act_stridey
//in_lsb_stridey is elements, do not apply bytes_of_elements
LDPARAM.U16 [in_lsb_stridey], %in_lsb_stridey
LDPARAM.U32 [blockXSize], %blockXSize

LDPARAM.U32 [tid_base], %Tmp0
LDPARAM.U32 [tid_depack], %Tmp1
LOADCONT %Tmp0, MB
DECTID.Y MB, %Tmp1, VA
##
MUL.U16.LOW VA, %act_stridey, VA
##
LDPARAM.U32 [in_lsb_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %in_lsb, %Tmp0, %in_lsb
// grid.x splits N across x-chunks, so move the packed input/output bases
// to this block's N-slice before the per-row loads/stores.
CVT.U16.U32 S28 DS13
MUL.U32 DS13, %blockXSize, %Tmp1
ADD.U32 %in_lsb, %Tmp1, %in_lsb

LDPARAM.U32 [out_lsb], %out_lsb
LDPARAM.U32 [out_msb], %out_msb
LDPARAM.U32 [out_blockz_size], %Tmp0
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 %out_lsb, %Tmp0, %out_lsb
ADD.U32 %out_msb, %Tmp0, %out_msb
ADD.U32 %out_lsb, %Tmp1, %out_lsb
ADD.U32 %out_msb, %Tmp1, %out_msb


LDPARAM.U32 [in_a_blockz_size], %Tmp0
LDPARAM.U32 [in_act], DS13
MUL.U32 %bidz, %Tmp0, %Tmp0
ADD.U32 DS13, %Tmp0, DS13
MSHIFTADD VA, DS13, 0, DVA
MOVE.U32 0, VAB
##
//-----------------------------------------------------------------------------------------------------
// in_scale   [K/256]           | [8]       | [N]
// in_scale   [grid.z]          | [y]       | [x]
//-----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// in_lsb  [K/256]      | [8]      | [32/4]      | [4][N]
// in_lsb  [1]          | [y]      | [unroll]    | [x]
//         [grid.z]     | [y]                    | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// in_act  [1]      | [K / 256]  | [8]      | [256 / 8]
//                               | [y]      | [unroll]
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
// output             | [8]      | [N]
//                    | [y]      | [grid.x]*[x]
//----------------------------------------------------------------------------------------------------
$BLOCK_0:
COPYEXT VAB, IEF
##
LOADALN %in_lsb, %in_lsb_stridey, %zero, MB
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
$STORE:
// Store the 8 x block_x partial tile into the logical [8][N] temp buffer.
// STORECONT uses the launch-local row width, which aliases rows when grid.x > 1.
LDPARAM.U16 [out_stride_y], %out_stridey
LDPARAM.U32 [out_stride_z], %Tmp0
STOREALN %out_lsb, %out_stridey, %Tmp0, VA
MOVE.U16 VB, VA
##
STOREALN %out_msb, %out_stridey, %Tmp0, VA
##
