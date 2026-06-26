.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_0 //in_a_addr
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_1 //in_b_addr
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_2 //out_addr
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_21 //offset of high low
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_3 //scale_addr
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_7 //B_Nd_size
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_8 //C_Nd_size
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_12 //stride of input a address
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_13 //stride of input b address
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_14 //stride of scale address
.param   .U32 _Z6MatMulPsS_S_S_tttttt_param_11 //lut address
.param   .U32 _zero_addr
.param   .U32 _inAcc_addr_lo
.param   .U32 _inAcc_addr_hi
.param   .U16 _Z6MatMulPsS_S_S_tttttt_param_4 //Md
.param   .U16 _Z6MatMulPsS_S_S_tttttt_param_5 //Nd
.param   .U16 _Z6MatMulPsS_S_S_tttttt_param_6 //Kd
.param   .U16 _Z6MatMulPsS_S_S_tttttt_param_9 //Kd_size
.param   .U16 _Z6MatMulPsS_S_S_tttttt_param_10 //combine, 0, no 1 yes

//x: column
//y: row
//z: group
//weights format: pack 4 rows into 2 bytes
// c0(r0,r1,r2,r3),c1(r0,r1,r2,r3), ... ,cn(r0,r1,r2,r3)
// c0(r4,r5,r6,r7),c1(r4,r5,r6,r7), ... ,cn(r4,r5,r6,r7)

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
LDPARAM.U32 [_zero_addr], DS4
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_3], DS3
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_14], DS6
MOVE.U16 1, S1
MUL.U16.WIDE S1, S30, DS1
MUL.U32 DS1, DS6, DS6
ADD.U32 DS3, DS6, DS3

//SHR.U32 DS6, S1, DS6
ADD.U32 DS4, DS6, DS4

MUL.WIDE.U16 S28, S0, DS2
SHL.U32 DS2, S1, DS2
ADD.U32 DS4, DS2, DS4
ADD.U32 DS3, DS2, DS3
LDPARAM.U32 [tid_base], DS5
LDPARAM.U32 [tid_depack], DS6
LOADCONT DS5, MB
DECTID.X MB, DS6, VA
##
LOAD [DS4 + (VA << 1)], IB
##
LOADCONT DS3, IE
##
//IB = scale * zero
LDPARAM.U16 [_Z6MatMulPsS_S_S_tttttt_param_9], S1
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_7], DS5
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_0], DS3
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_1], DS4
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_12], DS6
//ina = base + stride_a * block_z_idx
MUL.U32 DS1, DS6, DS6
ADD.U32 DS3, DS6, DS12

//inb (weights) = base + stride_b * block_z_idx
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_13], DS6
MUL.U32 DS1, DS6, DS6
ADD.U32 DS4, DS6, DS4

LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_11], DS8
//loop number = Kd/4 (handle 4 acc per loop)
LDPARAM.U16 [_Z6MatMulPsS_S_S_tttttt_param_6], S14
//weight address = blockIdx.x * blockDim.x * 4/2 + inb base address
ADD.S32 DS4, DS2, DS4

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
UNPACK.U4.LOW MB, VAB
##
COPY VAB, ICD
LOAD [DS8 + (VA << 1)], MA
HFMA IA, MA, IGH, VAB
##14
COPYEXT VAB, IGH
LOAD [DVA], IA
MADD DVA, 2, DVA
UNPACK.U4.HIGH IF, VAB
##
COPY VAB, ICD
LOAD [DS8 + (ID << 1)], MA
HFMA IA, MA, IGH, VAB
##
COPYEXT VAB, IGH
LOAD [DVA], IA
MADD DVA, 2, DVA
##
//LOAD W2
LOAD [DS8 + (IC << 1)], MA
HFMA IA, MA, IGH, VAB
##
COPYEXT VAB, IGH
LOAD [DVA], IA
MADD DVA, 2, DVA
##
LOAD [DS8 + (ID << 1)], MA
HFMA IA, MA, IGH, VAB
##
ADD.S32 DS4, DS5, DS4
SUB.S16 S14, S20, S14
CMP.EQ.S16 S14, S2, S19
JMPC  S19, $L__BB2_3
JMP  $L__BB2_2
$L__BB2_3:
COPYEXT VAB, IGH
##
LDPARAM.U32 [_inAcc_addr_lo], DS5
LDPARAM.U32 [_inAcc_addr_hi], DS6
MOVE.U16 2, S1
MUL.U16.WIDE S30, S1, DS13
ADD.U32 DS5, DS13, DS5
ADD.U32 DS6, DS13, DS6
MMOV DS5, DVA
CVT.F16.F32 IB, VAB
##
COPY VAB, IAB
LOAD [DVA], IC
MMOV DS6, DVA
CVT.F16.F32 IE, VAB
##
LOAD [DVA], ID
MUL.F32 VAB, IGH, VAB
##

//IGH = scale * wei * x
COPYEXT VAB, IGH
//VAB = scale * zero * x
MUL.F32 IAB, ICD, VAB
##
ADD.F32 IGH, VAB, VAB
##
COPY VAB, IAB
##
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_2], DS1
LDPARAM.U16 [_Z6MatMulPsS_S_S_tttttt_param_10], S8
OR.U16 S8, S30,S8
ADD.S32 DS1, DS2, DS1
LDPARAM.U32 [_Z6MatMulPsS_S_S_tttttt_param_21], DS10
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