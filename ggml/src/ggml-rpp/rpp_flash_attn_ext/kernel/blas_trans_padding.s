	.param	.U32	input
	.param	.U32	output
	.param	.U32	inBlkStride
	.param	.U32	outBlkPadStride
	.param  .U32    outYStride
	.param	.U16	nMat
	.param	.U16	nBlockX
	.param	.U16	nBlockY

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
	LDPARAM.U32 [input], DS0	
	LDPARAM.U32 [output], DS1
	LDPARAM.U32 [inBlkStride], DS2
	LDPARAM.U32 [outBlkPadStride], DS3
	LDPARAM.U32 [outYStride], DS5
	LDPARAM.U16 [nMat], S24
	LDPARAM.U16 [nBlockX], S20
	LDPARAM.U16 [nBlockY], S21
	MOVE.U16 1, S22

$LOOP_BLK:
    LOADCONT DS0, IA
	##
	STOREALN DS1, S10, DS5, IA
	##
	ADD.U32 DS0, DS2, DS0
	ADD.U32 DS1, DS2, DS1
	SUB.U16 S20, S22, S20
	JMPC S20, $LOOP_BLK
	LDPARAM.U16 [nBlockX], S20
    SUB.U16 S21, S22, S21
	ADD.U32 DS1, DS3, DS1
	JMPC S21, $LOOP_BLK
	LDPARAM.U16 [nBlockX], S20
	LDPARAM.U16 [nBlockY], S21
	SUB.U16 S24, S22, S24
    JMPC S24, $LOOP_BLK
    RET
.Lfunc_end0:
	.size	blas_trans_padding