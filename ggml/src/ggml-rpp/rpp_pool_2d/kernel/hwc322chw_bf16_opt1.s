
	.param	.U32	input
	.param	.U32	output
	.param	.U32	src_x_stribe
	.param	.U32	store_jump_size
	.param	.U32	loop_n
	.param  .U32    in_block_size
	.param  .U32    out_block_size

hwc322chw_opt1:
    LDPARAM.U16 [tid_xyz], S0 
    LDPARAM.U16 [tbDim.x], S1
    LDPARAM.U16 [tbDim.y], S2
    CFGXYZ S1, S2, S0
    MOVE.U32 0, VAB
    MMOV 0, DVA
    ##
    COPY VAB, IAB
    COPY VAB, ICD
    COPYEXT VAB, IEF
    COPYEXT VAB, IGH
    ##
	LDPARAM.U32 [input], DS11
	LDPARAM.U32 [output], DS12
	//calculate the base address = block_idx * block size
	LDPARAM.U32 [in_block_size], DS2
	MOVE.U16 1, S0
	MUL.U16.WIDE %blockIdx, S0, DS3
	MUL.U32.LOW DS3,DS2, DS2
	ADD.U32 DS11, DS2, DS11
	LDPARAM.U32 [out_block_size], DS2
	MUL.U32.LOW DS3,DS2, DS2
	ADD.U32 DS12, DS2, DS12
	LDPARAM.U32 [tid_base], DS0	
	LDPARAM.U32 [tid_depack], DS1
	LOADCONT DS0, MB
	COPYEXT MB, IG
	DECTID.X MB, DS1, VA
	##
	CVT.U16.U32 VA, VAB
	##
	LDPARAM.U32 [src_x_stribe], DS10
	MUL.U32.LOW VAB, DS10, VAB
	##
	COPY VAB, IAB
	LDPARAM.U32 [loop_n], DS8
	LDPARAM.U32 [store_jump_size], DS4
	MMOV DS11, DVA
	DECTID.Z IG, DS1, IC
	##
	MUL.U16.LOW IC, 1056, IC
	##
	DECTID.Y IG, DS1, VA
	##
	ADD VA, IC, VA
	##
	CVT.U16.U32 VA, VAB
	##
	MOVE.U32 2, DS0	//src_y_stribe
	MADD DVA, IAB, DVA
	MUL.U32.LOW VAB, DS0, VAB
	##
	MADD DVA, VAB, DVA
	##
	MOVE.U16 3, S0
	MOVE.U16 12, S1
	MOVE.U16 48, S2
	MOVE.U16 192, S3
	MOVE.U16 32, S17
	MOVE.U32 256, DS6
	MOVE.U16 1, S4
$LOOP:
	LOADMASK [DVA], S0, MA
	ADD.U16 MA, 0, VA
	##
	LOADMASK [DVA], S1, MA
	MERGEMASK VA, MA, S1, VA
	##
	LOADMASK [DVA], S2, MA
	MERGEMASK VA, MA, S2, VA
	##
	LOADMASK [DVA], S3, MA
	MERGEMASK VA, MA, S3, VA
	##
	MADD DVA, 8448, DVA
	STOREALN DS12, DS4, S17, VA
	##
	ADD.U32 DS12, DS6, DS12
	##	
	MOVE.U16 S0, S0
	##	
	MOVE.U16 S0, S0
	##
	SUB.U16 S16, S4, S16
	JMPC S16, $LOOP
	RET
.Lfunc_end0:
	.size	test_hwc322chw
//////////////////////////////////////////////////////////
