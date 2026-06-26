//ENTRY:
.param .u32 input0
.param .u32 weight
.param .u32 slot_id_base
.param .u32 token_id_base
.param .u32 out
.param .u32 offset0
.param .u16 slots
.param .u16 row_stride


.var .u32 %input0
.var .u32 %weight
.var .u32 %offset0
.var .u32 %slot_id_base
.var .u32 %token_id_base
.var .u32 %out
.var .u32 %two
.var .u16 %loop
.var .u16 %begin
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %one
.var .u16 %slots
.var .u16 %row_stride


//----------------------------------------------------------------------------------------------------   
// input0        [M]      |      [N]    
// 			     [loop]   |      [x]
//
// weight        [M]      |   [slot]
// 			     [loop]   |   [8]
//
// slot_id_base  [ubatch * 8]
//
// token_id_base  [ubatch * 8]
//
// output        [ubatch]      |    [N]
// M is dynamic, configure through FRAM register [cfg_fft_w4_6]
// begin is dynamic, configure through FRAM register [cfg_fft_w4_7]
// slot_id = slot_id_base[begin]
// row_stride = N * sizeof(short)
//----------------------------------------------------------------------------------------------------

LDPARAM.U16 [tid_xyz], %S0 
LDPARAM.U16 [tbDim.x], %S1
LDPARAM.U16 [tbDim.y], %S2
CFGXYZ %S1, %S2, %S0
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
LDPARAM.U32 [tid_depack], %input0        
LDPARAM.U32 [tid_base], %weight
LOADCONT %weight, MB
DECTID.X MB, %input0, VA
##
COPY VA, IA
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [weight], %weight
LDPARAM.U32 [slot_id_base], %slot_id_base
LDPARAM.U32 [token_id_base], %token_id_base
LDPARAM.U32 [out], %out
LDPARAM.U16 [cfg_fft_w4_6], %loop
LDPARAM.U16 [cfg_fft_w4_7], %begin
LDPARAM.U16 [slots], %slots
LDPARAM.U32 [offset0], %offset0
LDPARAM.U16 [row_stride], %row_stride
MOVE.U16 2, %S0
CVT.U16.U32 %S0, %two
MOVE.U16 1, %one
MUL.U16.WIDE %S0, %begin, DS13
ADD.U32 DS13, %token_id_base, %token_id_base
ADD.U32 DS13, %slot_id_base, %slot_id_base
##
$BLOCK_ENTRY:
//----------------------------------------------------------------------------------------------------   
// tok_id = token_id_base[begin]
// slot_id = slot_id_base[begin]
// weight        [M]      |   [slot]
// 			     [loop]   |   [8]
// weight = weights + tok_id * slots + slots_id

// M is dynamic, configure through FRAM register [cfg_fft_w4_6]
// begin is dynamic, configure through FRAM register [cfg_fft_w4_7]
// slot_id = slot_id_base[begin]
//----------------------------------------------------------------------------------------------------
LOAD [%token_id_base + (IG << 1)], MA
COPYEXT MA, IH
MUL.U16.LOW MA, %slots, VA
##
LOAD [%slot_id_base + (IG << 1)], MA
ADD.U16 VA, MA, VA
##
LOADCONT %input0, MB
COPY MB, IC
LOAD [%weight + (VA << 1)], MA
COPY MA, ID
MUL.F16 MB, MA, VA
##
COPYEXT VA, IF
MUL.U16.WIDE IH, %row_stride, VAB
##
MADD.SCA32 VAB, %out, DVA
##
LOAD [DVA + (IA << 1)], MA
COPY MA, IC
ADD.F16 MA, IF, VA
##
STORE VA, [DVA + (IA << 1)]
##
ADD.U32 %input0, %offset0, %input0
ADD.U32 %token_id_base, %two, %token_id_base
ADD.U32 %slot_id_base, %two, %slot_id_base
SUB.U16 %loop, %one, %loop
JMPC  %loop, $BLOCK_ENTRY
##