.param   .U32 sparse_act
.param   .U32 dense_act
.param   .U32 token_id_base
.param   .U32 row_stride


.var .u32  %sparse_act
.var .u32  %dense_act
.var .u32  %out_row_stride
.var .u32  %token_id_base
.var .u32  %token_id_beg
.var .u32  %temp0
.var .u32  %temp1
.var .u16  %one
.var .u16  %in_row_stride
.var .u16  %loop

LDPARAM.U16 [tbDim.x], %in_row_stride
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %loop
CFGXYZ %in_row_stride, %one, %loop
MOVE.U16 1, %one

//---------------------------------------------------------------------------------------------------
// sparse_act[B][K]
// dense_act[M][K]
// B is fix parameter related to ubatch
// M is tokens number for current expert
// M is dynamic and configure through FRAM register [cfg_fft_w4_6]
// beg_offset is  dynamic and configure through FRAM register [cfg_fft_w4_7]
//----------------------------------------------------------------------------------------------------
// sparse_act    [B]      |  [K]
// 					      |  [x]
// dense_act     [M]      |  [K]
// 				 [loop]	  |  [x]
// token_id      [B]
// token_id_beg  [M]
// token_id_beg  = token_id + beg_offset
//----------------------------------------------------------------------------------------------------

LDPARAM.U32 [sparse_act], %sparse_act
LDPARAM.U32 [dense_act], %dense_act
LDPARAM.U32 [token_id_base], %token_id_base
LDPARAM.U32 [row_stride], %out_row_stride
LDPARAM.U16 [cfg_fft_w4_6], %loop
LDPARAM.U32 [cfg_fft_w4_7], %token_id_beg
SHL.U32 %token_id_beg, %one, %token_id_beg
ADD.U32 %token_id_base, %token_id_beg, %token_id_base
 CVT.U32.U16 %out_row_stride, %in_row_stride
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
FORWARD DVA
##

LDPARAM.U32 [tid_base], %temp0
LDPARAM.U32 [tid_depack], %temp1
LOADCONT %temp0, MB
DECTID.X MB, %temp1, VA   
##
MUL.U16.WIDE VA, 2, VAB
##
MOVE.U32 2, %temp0
COPYEXT VAB, IEF
##
$BLOCK_X:
LOAD [%token_id_base + (IG << 1)], MA
MUL.U16.WIDE MA, %in_row_stride, VAB
##
ADD.U32 IEF, VAB, VAB
##
MADD.SCA32 VAB, %sparse_act, DVA
##
LOAD [DVA], IA
##
STORECONT %dense_act, IA
##
SUB.U16 %loop, %one, %loop
ADD.U32 %token_id_base, %temp0, %token_id_base
ADD.U32 %dense_act, %out_row_stride, %dense_act
JMPC %loop, $BLOCK_X
##
