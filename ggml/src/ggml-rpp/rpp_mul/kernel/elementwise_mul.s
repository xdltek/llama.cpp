//ENTRY:
.param .u32 input0
.param .u32 input1
.param .u32 out
.param .u32 offset0
.param .u32 offset1
.param .u32 loop
.param .u32 stridez0
.param .u32 stridez1

.var .u32 %input0
.var .u32 %input1
.var .u32 %offset0
.var .u32 %offset1
.var .u32 %stridez0
.var .u32 %stridez1
.var .u32 %out
.var .u16 %loop
.var .u16 %S0
.var .u16 %S1
.var .u16 %S2
.var .u16 %one


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
LDPARAM.U32 [offset0], %offset0
LDPARAM.U32 [offset1], %offset1
LDPARAM.U32 [stridez0], %stridez0
LDPARAM.U32 [stridez1], %stridez1
LDPARAM.U32 [out], %out
LDPARAM.U16 [loop], %loop
MOVE.U16 1, %one

$BLOCK_ENTRY:
LOADALN.Z32 %input0, %stridez0, %stridez0, MB
CVT.F16.F32 MB, VAB
##
COPY VAB, IAB
LOADALN.Z32 %input1, %stridez1, %stridez1, MB
CVT.F16.F32 MB, VAB
##
MUL.F32 IAB, VAB, VAB
##
CVT.F32.F16 VAB, VA
##
STOREALN.Z32 %out, %stridez0, %stridez0, VA
##
ADD.U32 %input0, %offset0, %input0
ADD.U32 %input1, %offset1, %input1
ADD.U32 %out, %offset0, %out
SUB.U16 %loop, %one, %loop
JMPC  %loop, $BLOCK_ENTRY
##