//ENTRY:
.param .u32 input0
.param .u32 input1
.param .u32 lut
.param .u32 out
.param .u32 offset0
.param .u32 loop
.param .u32 mode

.var .u32 %input0
.var .u32 %input1
.var .u32 %lut
.var .u32 %offset0
.var .u32 %out
.var .u16 %loop
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %one
.var .u16 %mode


//----------------------------------------------------------------------------------------------------   
// In/Out        [M]         | [N]    
// 			     [loop]      | [x]
// M is dynamic, configure through FRAM register [cfg_fft_w4_6]
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
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [lut], %lut
LDPARAM.U32 [offset0], %offset0
LDPARAM.U32 [out], %out
LDPARAM.U16 [cfg_fft_w4_6], %loop
LDPARAM.U16 [mode], %mode
MOVE.U16 1, %one

$BLOCK_ENTRY:
LOADCONT %input0, IA
##
LOAD [%lut + (IA << 1)], IB
##
JMPNC %mode, $END
LOADCONT %input1, MB
MUL.F16 IB, MB, VA
##
COPY VA, IB
##
$END:
STORECONT %out, IB
##
ADD.U32 %input0, %offset0, %input0
ADD.U32 %input1, %offset0, %input1
ADD.U32 %out, %offset0, %out
SUB.U16 %loop, %one, %loop
JMPC  %loop, $BLOCK_ENTRY
##