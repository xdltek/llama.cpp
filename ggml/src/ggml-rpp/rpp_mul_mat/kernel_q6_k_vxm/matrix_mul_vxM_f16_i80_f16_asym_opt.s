.param   .U32  in_a_addr 
.param   .U32  in_b_addr
.param   .U32  out_addr
.param   .U32  hilo_offset
.param   .U32  scale_addr
.param   .U32  in_b_row_size
.param   .U32  rsvd0
.param   .U32  in_a_stride
.param   .U32  in_b_stride
.param   .U32  scale_stride
.param   .U32  lut_addr
.param   .U32  zero_addr
.param   .U32  inAcc_addr_lo
.param   .U32  inAcc_addr_hi
.param   .U16  rsvd1
.param   .U16  rsvd2
.param   .U16  Kd
.param   .U16  Kd_size
.param   .U16  combine


LDPARAM.U16 [tbDim.x], S0
LDPARAM.U16 [tbDim.y], S1
LDPARAM.U16 [tid_xyz], S2
CFGXYZ S0, S1, S2
MMOV 0, DVA
MOVE.U16 0, VA
##0
COPY VA, IA
COPY VA, IB
COPY VA, IC
COPY VA, ID
COPYEXT VA, IE
COPYEXT VA, IF
COPYEXT VA, IG
COPYEXT VA, IH
##

/////////////////////////////////////////////////////////////////////////////
//Let G = K / 32
//A     [1][K]
//B     [K][N]
//Scale [K/32][N]
//---------------------------------------------------------------------------
//A     [1][G][32]
//B     [G][32][N]
//Scale [G][N]
//grid.z = G;
//grid.x = N;
/////////////////////////////////////////////////////////////////////////////
///scale_addr += (blockIdx.z * scale_stride + blockIdx.x * blockDim.x) * 2
/////////////////////////////////////////////////////////////////////////////
LDPARAM.U32 [scale_addr], DS3
LDPARAM.U32 [scale_stride], DS6
MOVE.U16 1, S1
MUL.U16.WIDE S1, S30, DS1
MUL.U32 DS1, DS6, DS6
ADD.U32 DS3, DS6, DS3
MUL.WIDE.U16 S28, S0, DS2
SHL.U32 DS2, S1, DS2
ADD.U32 DS3, DS2, DS3
LOADCONT DS3, IE
##
/////////////////////////////////////////////////////////////////////////////////
///in_a_stride = 32 * sizeof(short)
///in_b_stride = 32 * N
///in_a_addr += (blockIdx.z * in_a_stride)
///in_b_addr += blockIdx.z * in_b_stride + blockIdx.x * blockDim.x * sizeof(short)
///////////////////////////////////////////////////////////////////////////////////
LDPARAM.U16 [Kd_size], S1
LDPARAM.U32 [in_b_row_size], DS5


LDPARAM.U32 [in_a_addr], DS3
LDPARAM.U32 [in_b_addr], DS4
LDPARAM.U32 [in_a_stride], DS6
MUL.U32 DS1, DS6, DS6
ADD.U32 DS3, DS6, DS12
LDPARAM.U32 [in_b_stride], DS6
MUL.U32 DS1, DS6, DS6
ADD.U32 DS4, DS6, DS4
ADD.S32 DS4, DS2, DS4


/////////////////////////////////////////////////////////////////////////////
//Let G = K / 32
//Kd = G
/////////////////////////////////////////////////////////////////////////////
LDPARAM.U16 [Kd], S14

MOVE.U16 0, S2
MOVE.U16 1, S20
MOVE.U32 2, DS3
MOVE.U32 0, VAB
MMOV DS12, DVA
##
$L__BB2_2:
COPYEXT VAB, IGH
LOAD [DVA], IA
MADD DVA, 2, DVA
LOADCONT DS4, MB
COPYEXT MB, IF
UNPACK.U8 MB, VAB
##
COPY VAB, ICD
CVT.S8.F16 VA, VA
##
HFMA IA, VA, IGH, VAB
##
COPYEXT VAB, IGH
LOAD [DVA], IA
MADD DVA, 2, DVA
CVT.S8.F16 ID, VA
##
HFMA IA, VA, IGH, VAB
##
/////////////////////////////////////////////////////////////////////////////////
//in_b_row_size = N * sizeof(short)
///in_b_addr += in_b_row_size
///////////////////////////////////////////////////////////////////////////////////

ADD.S32 DS4, DS5, DS4
SUB.S16 S14, S20, S14
CMP.EQ.S16 S14, S2, S19
JMPC  S19, $L__BB2_3
JMP  $L__BB2_2
$L__BB2_3:
COPYEXT VAB, IGH
CVT.F16.F32 IE, VAB
##
MUL.F32 VAB, IGH, VAB
##
COPY VAB, IAB
##
LDPARAM.U32 [out_addr], DS1
LDPARAM.U16 [combine], S8
OR.U16 S8, S30,S8
ADD.S32 DS1, DS2, DS1
LDPARAM.U32 [hilo_offset], DS10
ADD.U32 DS1, DS10, DS10
JMPNC S8, $STORE
LOADCONT DS1, ID
##
//low word
LOADCONT DS10, MB
ADD.F32.R0MB ID, IAB, VAB
##
COPY VAB, IAB
##

$STORE:
//high word
STORECONT DS1, IB
##
//low word
STORECONT DS10, IA
##16
DMB_ORI
##