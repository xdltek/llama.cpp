//ENTRY:
.param .u32 input0
.param .u32 out_lo
.param .u32 out_hi
.param .u16 group

.var .u32 %input0
.var .u32 %out_lo
.var .u32 %out_hi
.var .u32 %tid_depack
.var .u32 %tid_base
.var .u16 %group
.var .u16 %sreg0
.var .u16 %sreg1

LDPARAM.U16 [tid_xyz], %sreg0
LDPARAM.U16 [tbDim.x], %group
LDPARAM.U16 [tbDim.y], %sreg1
CFGXYZ %group, %sreg1, %sreg0

MOVE.U32 0, VAB
MMOV 0, DVA
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
MOVE.U16 0x3f80, VA
##
COPY VA, IA
LDPARAM.U16 [group], %group
LDPARAM.U32 [tid_depack], %tid_depack        
LDPARAM.U32 [tid_base], %tid_base
LOADCONT %tid_base, MB
DECTID.X MB, %tid_depack, VA
##
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [out_lo], %out_lo
LDPARAM.U32 [out_hi], %out_hi
MUL.U16.LOW VA, %group, VA
##
MSHIFTADD VA, %input0, 1, DVA
MOVE.U32 0, VAB
##
REPEAT 1, %group
LOAD [DVA], MA
MADD DVA, 2, DVA
HFMA IA, MA, VAB, VAB
##
COPY VAB, ICD
STORECONT %out_lo, VA
##
STORECONT %out_hi, ID
##