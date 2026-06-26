.param   .U32 input_addr       //4bits weights address
.param   .U32 dequant_tb_addr  //dequant table address
.param   .U32 output_addr      //out_addr
.param   .U32 output_stride    //out_addr stride of multi stores
.param   .U32 lut_addr         //lookup table I4 to BF16 or i8
.param   .U32 inStrideY	       //load stride Y of weights
.param   .U32 inStrideZ        //load stride Z of weights
.param   .U32 dequantStrideY   //load stride Y of weights
.param   .U32 dequantStrideZ   //load stride Z of weights
.param   .U32 outStrideY	   //store stride Y of weights
.param   .U32 outStrideZ       //store stride Z
.param   .U32 inBlockXSize	   //input block X size
.param   .U32 inBlockYSize	   //input block Y size
.param   .U32 inDequantXSize   //input block X size
.param   .U32 inDequantYSize   //input block Y size
.param   .U32 outBlockXSize	   //output block X size
.param   .U32 outBlockYSize	   //output block Y size
.param   .U32 zero_addr
.param   .U16 blockX
.param   .U16 blockY

.var .u32  %in_addr
.var .u32  %dequant_addr
.var .u32  %out_addr
.var .u32  %zero_addr
.var .u32  %lut_addr

.var .u32  %inBlockXSize
.var .u32  %inBlockYSize


.var .u32  %strideZ

.var .u32  %outBlockXSize
.var .u32  %outBlockYSize
.var .u32  %output_stride

.var .u16  %cntX
.var .u16  %cntY
.var .u16  %one
.var .u16  %strideY

LDPARAM.U16 [tbDim.x], %cntY
LDPARAM.U16 [tbDim.y], %one
LDPARAM.U16 [tid_xyz], %cntX
CFGXYZ %cntY, %one, %cntX
MOVE.U16 1, %one

LDPARAM.U32 [input_addr], %in_addr
LDPARAM.U32 [dequant_tb_addr], %dequant_addr
LDPARAM.U32 [zero_addr], %zero_addr
LDPARAM.U32 [output_addr], %out_addr
LDPARAM.U32 [inBlockXSize], %inBlockXSize
LDPARAM.U32 [inBlockYSize], %inBlockYSize
LDPARAM.U32 [outBlockXSize], %outBlockXSize
LDPARAM.U32 [outBlockYSize], %outBlockYSize
LDPARAM.U32 [output_stride], %output_stride
LDPARAM.U32 [lut_addr], %lut_addr

LDPARAM.U16 [blockX], %cntX
LDPARAM.U16 [blockY], %cntY


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
$BLOCK_ENTRY:
LDPARAM.U16 [inStrideY], %strideY
LDPARAM.U32 [inStrideZ], %strideZ
LOADALN.Z32 %in_addr, %strideY, %strideZ, MB
COPYEXT MB, IG
UNPACK.U4.LOW MB, VAB
##6
COPY VAB, ICD
LDPARAM.U16 [dequantStrideY], %strideY
LDPARAM.U32 [dequantStrideZ], %strideZ
LOADALN.Z32 %zero_addr, %strideY, %strideZ, IE
CVT.U8.F16 VA, VA
##
COPY VA, IC
LOADALN.Z32 %dequant_addr, %strideY, %strideZ, IB
CVT.U8.F16 ID, VA
##
COPY VA, ID
FMA IC, IB, IE, VA
##
LDPARAM.U16 [outStrideY], %strideY
LDPARAM.U32 [outStrideZ], %strideZ
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
FMA ID, IB, IE, VA
##
ADD.U32 %out_addr, %output_stride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
UNPACK.U4.HIGH IG, VAB
##
COPY VAB, ICD
CVT.U8.F16 VA, VA
##
COPY VA, IC
CVT.U8.F16 ID, VA
##
COPY VA, ID
FMA IC, IB, IE, VA
##
ADD.U32 %out_addr, %output_stride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
FMA ID, IB, IE, VA
##
ADD.U32 %out_addr, %output_stride, %out_addr
STOREALN.Z32  %out_addr, %strideY, %strideZ, VA
##
LDPARAM.U32 [inDequantXSize], %strideZ
SUB.U16 %cntX, %one, %cntX
ADD.U32 %in_addr, %inBlockXSize, %in_addr
ADD.U32 %dequant_addr, %strideZ, %dequant_addr
//SHR.U32 %strideZ, %one, DS13
ADD.U32 %zero_addr, %strideZ, %zero_addr
ADD.U32 %out_addr, %outBlockXSize, %out_addr
JMPC %cntX, $BLOCK_ENTRY

LDPARAM.U32 [inDequantYSize], %strideZ
LDPARAM.U16 [blockX], %cntX
SUB.U16 %cntY, %one, %cntY
ADD.U32 %in_addr, %inBlockYSize, %in_addr
ADD.U32 %dequant_addr, %strideZ, %dequant_addr
//SHR.U32 %strideZ, %one, DS13
ADD.U32 %zero_addr, %strideZ, %zero_addr
ADD.U32 %out_addr, %outBlockYSize, %out_addr

JMPC %cntY, $BLOCK_ENTRY
##