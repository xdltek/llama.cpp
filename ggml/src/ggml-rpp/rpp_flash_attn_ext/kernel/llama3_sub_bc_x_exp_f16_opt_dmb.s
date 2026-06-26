.param .u32 input0
.param .u32 input1
.param .u32 expTable
.param .u32 output
.param .u32 in0BlockStride
.param .u32 in1BlockStride
.param .u16 blocks

.var .u32 %input0
.var .u32 %input1
.var .u32 %expTable
.var .u32 %output
.var .u32 %in0BlockStride
.var .u32 %in1BlockStride
.var .u32 %tmp
.var .u16 %tmp0
.var .u16 %tmp1
.var .u16 %blocks


LDPARAM.U16 [tbDim.x], %tmp0
LDPARAM.U16 [tbDim.y], %tmp1
LDPARAM.U16 [tid_xyz], %blocks
CFGXYZ %tmp0, %tmp1, %blocks


MMOV 0, DVA
MOVE.U32 0, VAB
##
COPY VAB, IAB
COPY VAB, ICD
COPYEXT VAB, IEF
COPYEXT VAB, IGH
##
LDPARAM.U32 [input1], %input1
LDPARAM.U32 [tid_base], %in0BlockStride
LDPARAM.U32 [tid_depack], %in1BlockStride
LOADCONT %in0BlockStride, MB
DECTID.Y MB, %in1BlockStride, VA
##
LOAD [%input1 + (VA << 1)], ID
LDPARAM.U32 [input0], %input0
LDPARAM.U32 [expTable], %expTable
LDPARAM.U32 [output], %output
LDPARAM.U32 [in0BlockStride], %in0BlockStride
MOVE.U16 0, %tmp0
MOVE.U16 1, %tmp1
LDPARAM.U16 [blocks], %blocks
##

$BLOCK_ENTRY:
LOADCONT %input0, IA
##
SUB.F16 IA, ID, VA
##
LOAD [%expTable + (VA << 1)], IA
##
STORECONT %output, IA
##
DMB_ORI
##
ADD.U32 %input0, %in0BlockStride, %input0
ADD.U32 %output, %in0BlockStride, %output
SUB.U16 %blocks, %tmp1, %blocks
JMPC %blocks, $BLOCK_ENTRY
##
